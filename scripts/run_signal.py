"""1コマンドで特定日のシグナルレポートを生成する統合パイプライン。

データはローカルの CSV/Parquet を優先し、不足分のみ自動スクレイプする。
HTML（recommend_today.py のような）は使わず、安定した LZH + odds CSV ベース。

使い方:
    # 今日のシグナル（--date 省略で today）
    python -m scripts.run_signal --autofill

    # 特定日（昨日や過去日）
    python -m scripts.run_signal --date 2026-05-11

    # 不足データを自動取得（races.parquet に該当日が無ければ scrape）
    python -m scripts.run_signal --date 2026-05-11 --autofill

    # 特定オプション
    python -m scripts.run_signal --date 2026-05-11 \\
        --ev-threshold 1.05 --max-odds 10.0 \\
        --exclude-venues 04 03 02 14 01 24

挙動:
    1. races.parquet に該当日のデータがあるか確認
    2. odds_win.csv に該当日のオッズが揃っているか確認
    3. features.parquet を最新化（モデル学習はスキップ）
    4. backtest を該当日に絞って実行（候補抽出）
    5. ベット候補のみ HTML 結果取得 → payouts に追加（候補だけなので高速）
    6. backtest 再実行（結果が反映される）
    7. signal_report で HTML 出力
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.config import MODELS_DIR, PROCESSED_DIR, RAW_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---- データ存在チェック ----

def _races_has_date(races_path: Path, target: date) -> int:
    """races.parquet に該当日のレース数があるか返す。0 なら未取得。"""
    if not races_path.exists():
        return 0
    df = pd.read_parquet(races_path, columns=["race_id", "race_date"])
    df["d"] = df["race_date"].astype(str).str.replace("-", "").str[:8]
    return int(df[df["d"] == target.strftime("%Y%m%d")]["race_id"].nunique())


def _odds_has_date(odds_path: Path, target: date) -> int:
    """odds_win.csv に該当日の「有効オッズあり」 race_id 数を返す。

    0.00 等の無効値しか持たない race_id はカウントしない。--autofill 時に
    自動再取得されるようにするため。
    """
    if not odds_path.exists():
        return 0
    df = pd.read_csv(odds_path, usecols=["race_id", "odds_win"])
    df = df[df["race_id"].astype(str).str.startswith(target.strftime("%Y%m%d"))]
    df["odds_win"] = pd.to_numeric(df["odds_win"], errors="coerce")
    valid = df[df["odds_win"] >= 1.0]
    return int(valid["race_id"].nunique())


# ---- サブプロセス起動 ----

def _run(cmd: list[str]) -> None:
    logger.info("実行: %s", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        logger.error("コマンド失敗 (rc=%d): %s", res.returncode, " ".join(cmd))
        sys.exit(res.returncode)


def _autofill_races(target: date) -> None:
    """races.parquet に該当日のデータが無ければ取得（公式LZH優先・HTML補完）。"""
    _run([sys.executable, "-m", "scripts.scrape_dataset",
          "--from", target.isoformat(), "--to", target.isoformat()])


def _autofill_odds(target: date, workers: int, *, force: bool = False,
                   skip_past_deadline: Optional[Path] = None) -> None:
    """odds_win.csv に該当日のオッズを取得。

    force=True なら --resume を付けず既存レースも再スクレイプする（オッズは
    締切に向けて動くため、watch_signal の定期実行では force=True が必要）。
    skip_past_deadline に schedule CSV を渡すと、締切過ぎのレース（オッズは
    既に確定済み）はスクレイプ対象から除外される。
    """
    cmd = [sys.executable, "-m", "scripts.scrape_odds",
           "--from", target.isoformat(), "--to", target.isoformat(),
           "--races-from", str(RAW_DIR / "races.parquet"),
           "--workers", str(workers)]
    if not force:
        cmd.append("--resume")
    if skip_past_deadline is not None and skip_past_deadline.exists():
        cmd += ["--skip-past-deadline", str(skip_past_deadline)]
    _run(cmd)


def _schedule_today_coverage(schedule_path: Path, target: date) -> tuple[int, int]:
    """schedule_path にある target 日の deadline 数と、races.parquet 上の総レース数を返す。

    (have, expected) のタプル。races.parquet が無ければ (0, 0)。
    """
    races_path = RAW_DIR / "races.parquet"
    if not races_path.exists():
        return (0, 0)
    today_str = target.strftime("%Y%m%d")
    expected = int(
        pd.read_parquet(races_path, columns=["race_id"])["race_id"]
        .astype(str).str.startswith(today_str).sum()
    )
    if not schedule_path.exists():
        return (0, expected)
    try:
        sched = pd.read_csv(schedule_path, usecols=["race_id"])
    except Exception:
        return (0, expected)
    have = int(sched["race_id"].astype(str).str.startswith(today_str).nunique())
    return (have, expected)


def _autofill_schedule_all(target: date, workers: int) -> Path:
    """その日の全場スケジュールを取得（既に80%以上揃っていればスキップ）。

    取得済みなら HTTP リクエスト無しで race_schedule.csv のパスを返す。
    オッズスクレイプの「締切過ぎ除外」フィルタに使う。
    """
    schedule_path = RAW_DIR / "race_schedule.csv"
    have, expected = _schedule_today_coverage(schedule_path, target)
    if expected > 0 and have >= expected * 0.8:
        logger.info("schedule キャッシュ有効: %d/%d レース（fetch スキップ）",
                    have, expected)
        return schedule_path
    logger.info("schedule 取得開始: 既存 %d/%d レース（不足のため全場fetch）",
                have, expected)
    _run([sys.executable, "-m", "scripts.scrape_schedule",
          "--date", target.isoformat(),
          "--workers", str(workers)])
    return schedule_path


def _build_results_target_csv(target: date, bets_csv: Path) -> Optional[Path]:
    """結果スクレイプ対象 race_id を1つのCSVに集約。

    現在のベット候補 (bets_csv) と当日 signal_log に登場した全 race_id を
    まとめる。これで「シグナル発生したが候補外になったレース」も結果を取得し、
    HTML レポートの「本日のシグナル履歴」に反映できる。

    出力: data/processed/signal_log/_results_target_<date>.csv
    対象なし時は None を返す。
    """
    race_ids: set[str] = set()
    if bets_csv.exists():
        try:
            df = pd.read_csv(bets_csv, usecols=["race_id"])
            race_ids.update(df["race_id"].astype(str).dropna().unique())
        except Exception as e:
            logger.warning("bets_csv 読込失敗: %s", e)

    log_path = (PROCESSED_DIR / "signal_log"
                / f"signal_snapshots_{target.strftime('%Y%m%d')}.csv")
    if log_path.exists():
        try:
            df = pd.read_csv(log_path, usecols=["race_id"])
            race_ids.update(df["race_id"].astype(str).dropna().unique())
        except Exception as e:
            logger.warning("signal_log 読込失敗: %s", e)

    if not race_ids:
        return None

    out_path = (PROCESSED_DIR / "signal_log"
                / f"_results_target_{target.strftime('%Y%m%d')}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"race_id": sorted(race_ids)}).to_csv(out_path, index=False)
    return out_path


def _refresh_results_html(target: date, workers: int,
                          only_bets: Optional[Path] = None) -> None:
    """既に終わったレースの結果を HTML から取得して payouts に追加。

    only_bets が指定されると、その bets CSV にある race_id だけを取得する
    （= 4-6レースに絞れて高速・成功率高）。
    指定しないと当日の全レースを試行する。
    """
    cmd = [sys.executable, "-m", "scripts.scrape_results_html",
           "--date", target.isoformat(),
           "--workers", str(workers)]
    if only_bets is not None:
        cmd += ["--only-bets", str(only_bets)]
    _run(cmd)


def _refresh_schedule(target: date, workers: int,
                      only_bets: Optional[Path] = None) -> None:
    """ベット候補の場の締切時刻を取得して race_schedule.csv に保存。"""
    cmd = [sys.executable, "-m", "scripts.scrape_schedule",
           "--date", target.isoformat(),
           "--workers", str(workers)]
    if only_bets is not None:
        cmd += ["--only-bets", str(only_bets)]
    _run(cmd)


def _rebuild_features() -> None:
    """features.parquet を再生成（モデル学習はスキップ）。"""
    _run([sys.executable, "-m", "scripts.train_model",
          "--input", str(RAW_DIR / "races.parquet"),
          "--target", "is_win",
          "--features-only"])


def _run_backtest(target: date, *, ev_threshold: float, max_odds: float,
                  min_odds: Optional[float], kelly_fraction: float,
                  excluded_venues: list[str],
                  odds_shrinkage_max: float = 0.0,
                  watchlist_ev_threshold: Optional[float] = None) -> Path:
    """指定日1日分のバックテスト → bets_lane1_kelly.csv のパスを返す。

    target が当日（今日）の場合は --skip-post-deadline を自動付与し、
    既に締切過ぎたレースを候補から除外する（事後のみシグナルの誤検知防止）。
    過去日 backtest には影響しない。
    """
    cmd = [
        sys.executable, "-m", "scripts.backtest",
        "--features", str(PROCESSED_DIR / "features.parquet"),
        "--payouts", str(RAW_DIR / "races_payouts.parquet"),
        "--odds", str(RAW_DIR / "odds_win.csv"),
        "--strategy", "lane1_kelly",
        "--since", target.isoformat(),
        "--until", target.isoformat(),
        "--kelly-fraction", str(kelly_fraction),
        "--ev-threshold", str(ev_threshold),
        "--max-odds", str(max_odds),
    ]
    if min_odds is not None:
        cmd += ["--min-odds", str(min_odds)]
    if excluded_venues:
        cmd += ["--exclude-venues", *excluded_venues]
    if odds_shrinkage_max > 0:
        cmd += ["--odds-shrinkage-max", str(odds_shrinkage_max)]
    if watchlist_ev_threshold is not None:
        cmd += ["--watchlist-ev-threshold", str(watchlist_ev_threshold)]
    if target == date.today():
        cmd += ["--skip-post-deadline"]
    _run(cmd)
    return MODELS_DIR / "backtest" / "bets_lane1_kelly.csv"


def _run_historical_backtest(
    target: date, *, days: int, ev_threshold: float, max_odds: float,
    min_odds: Optional[float], kelly_fraction: float,
    excluded_venues: list[str],
    persisted_ev_threshold: float = 1.20,
) -> Optional[tuple[Path, str, str, Optional[Path]]]:
    """target 当日を含まない過去 days 日のバックテストを実行。

    2つのバックテストを実行:
      1. 通常 (--ev-threshold で指定): 全シグナルを買った場合
      2. 持続のみ近似 (--persisted-ev-threshold ≥ 1.20): 高EVシグナルのみ買った場合
         （オッズが 10-20% 動いても EV>1.05 を維持するレベル = 持続しやすい）

    成功時は (出力ディレクトリ, since, until, 持続版出力ディレクトリ) を返す。
    失敗時は None。持続版が ev_threshold と同じなら持続版は None。
    """
    since = (target - timedelta(days=days)).isoformat()
    until = (target - timedelta(days=1)).isoformat()
    out_dir = PROCESSED_DIR / "backtest_history"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _build_cmd(ev: float, out: Path) -> list[str]:
        cmd = [
            sys.executable, "-m", "scripts.backtest",
            "--features", str(PROCESSED_DIR / "features.parquet"),
            "--payouts", str(RAW_DIR / "races_payouts.parquet"),
            "--odds", str(RAW_DIR / "odds_win.csv"),
            "--strategy", "lane1_kelly",
            "--since", since,
            "--until", until,
            "--kelly-fraction", str(kelly_fraction),
            "--ev-threshold", str(ev),
            "--max-odds", str(max_odds),
            "--out", str(out),
        ]
        if min_odds is not None:
            cmd += ["--min-odds", str(min_odds)]
        if excluded_venues:
            cmd += ["--exclude-venues", *excluded_venues]
        return cmd

    # 1. 通常バックテスト
    cmd_std = _build_cmd(ev_threshold, out_dir)
    logger.info("実行: %s", " ".join(cmd_std))
    res = subprocess.run(cmd_std, check=False)
    if res.returncode != 0:
        logger.warning("過去バックテスト失敗 rc=%d (HTMLには載せず続行)", res.returncode)
        return None

    # 2. 持続シグナル近似（EV閾値を上げて再実行）
    persisted_out: Optional[Path] = None
    if persisted_ev_threshold > ev_threshold:
        persisted_out = PROCESSED_DIR / "backtest_history_persisted"
        persisted_out.mkdir(parents=True, exist_ok=True)
        cmd_p = _build_cmd(persisted_ev_threshold, persisted_out)
        logger.info("実行(持続版): %s", " ".join(cmd_p))
        res_p = subprocess.run(cmd_p, check=False)
        if res_p.returncode != 0:
            logger.warning("持続バックテスト失敗 rc=%d (省略して続行)", res_p.returncode)
            persisted_out = None

    return out_dir, since, until, persisted_out


def _append_signal_snapshot(target: date, bets_csv: Path) -> Optional[Path]:
    """各 iteration の bets を long-form snapshot CSV に追記。

    Schema: race_id, snapshot_at, ev, odds_win, stake, blended_win_prob,
            race_finished, hit, pnl, winner_lane

    候補から外れた race_id は追記されない（CSV 側で「最後に出現した時刻」を
    last_seen として扱える）。signal_report で集計 → HTML レンダリング。
    """
    log_dir = PROCESSED_DIR / "signal_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"signal_snapshots_{target.strftime('%Y%m%d')}.csv"

    if not bets_csv.exists():
        return log_path if log_path.exists() else None

    try:
        cur = pd.read_csv(bets_csv)
    except Exception as e:
        logger.warning("bets CSV 読込失敗: %s", e)
        return log_path if log_path.exists() else None

    if cur.empty:
        return log_path if log_path.exists() else None

    from datetime import datetime as _dt
    now = _dt.now().strftime("%Y-%m-%d %H:%M:%S")

    cols = ["race_id"]
    snap = cur[["race_id"]].copy()
    snap["snapshot_at"] = now
    for col in ["ev", "odds_win", "stake", "blended_win_prob",
                "race_finished", "hit", "pnl", "winner_lane"]:
        if col in cur.columns:
            snap[col] = cur[col]
            cols.append(col)

    if log_path.exists():
        snap.to_csv(log_path, mode="a", header=False, index=False)
    else:
        snap.to_csv(log_path, index=False)

    return log_path


def _make_report(bets_csv: Path, target: date, ev_threshold: float,
                 history: Optional[tuple[Path, str, str, Optional[Path]]] = None,
                 signal_log: Optional[Path] = None) -> Path:
    out = PROCESSED_DIR / f"signals_{target.strftime('%Y%m%d')}.html"
    cmd = [sys.executable, "-m", "scripts.signal_report",
           "--bets", str(bets_csv),
           "--out", str(out),
           "--ev-threshold", str(ev_threshold),
           "--date-label", target.isoformat()]
    if signal_log is not None and signal_log.exists():
        cmd += ["--signal-log", str(signal_log)]
    if history is not None:
        hist_dir, since, until, persisted_dir = history
        summary_file = hist_dir / "summary_lane1_kelly.json"
        equity_file = hist_dir / "equity_lane1_kelly.csv"
        bets_file = hist_dir / "bets_lane1_kelly.csv"
        if summary_file.exists():
            cmd += ["--history-summary", str(summary_file),
                    "--history-since", since,
                    "--history-until", until]
            if equity_file.exists():
                cmd += ["--history-equity", str(equity_file)]
            if bets_file.exists():
                cmd += ["--history-bets", str(bets_file)]
        # 持続シグナル近似 (高EVのみ買った場合) のサマリも渡す
        if persisted_dir is not None:
            p_summary = persisted_dir / "summary_lane1_kelly.json"
            if p_summary.exists():
                cmd += ["--persisted-summary", str(p_summary)]
    _run(cmd)
    return out


# ---- メイン ----

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", default=None,
                   help="シグナル対象日 (YYYY-MM-DD)。省略時は今日")
    p.add_argument("--autofill", action="store_true",
                   help="races.parquet / odds_win.csv に該当日が無ければ自動でスクレイプ")
    p.add_argument("--refresh-odds", action="store_true",
                   help="当日のオッズを強制再取得（締切直前にオッズが動いた時用）")
    p.add_argument("--skip-features", action="store_true",
                   help="features.parquet の再生成をスキップ（既に最新の場合）")
    p.add_argument("--ev-threshold", type=float, default=1.05)
    p.add_argument("--watchlist-ev-threshold", type=float, default=0.95,
                   help="準シグナル(事前予告)の EV 下限。--ev-threshold より低い値。"
                        "EV がこの値〜閾値の間のレースを watchlist_lane1_kelly.csv に出力。"
                        "watch_signal の pre-alert に使う。0で無効")
    p.add_argument("--min-odds", type=float, default=None,
                   help="1号艇オッズ下限。指定すると本命過ぎを除外。デフォルトは未設定")
    p.add_argument("--max-odds", type=float, default=10.0)
    p.add_argument("--kelly-fraction", type=float, default=0.25)
    p.add_argument("--exclude-venues", nargs="*",
                   default=["04", "03", "02", "14", "01", "24"],
                   help="除外場コード（デフォルト: 平和島/江戸川/戸田/鳴門/桐生/大村）")
    p.add_argument("--workers", type=int, default=4,
                   help="スクレイプの並列数（autofill 時のみ使う）")
    p.add_argument("--no-open", action="store_true",
                   help="生成後にブラウザを自動で開かない（デフォルトは開く）")
    p.add_argument("--no-refresh-results", action="store_true",
                   help="HTMLから日中の確定済みレース結果を取得しない（デフォルトは取得）")
    p.add_argument("--no-backtest-summary", action="store_true",
                   help="HTMLレポートに過去バックテスト集計を含めない（watch_signal の定期実行で時短）")
    p.add_argument("--backtest-days", type=int, default=365,
                   help="過去バックテストの期間（日数）。デフォルト365日。データが無い分は自動で短くなる")
    p.add_argument("--odds-shrinkage-max", type=float, default=0.0,
                   help="時間ベース shrinkage の最大係数 (0で無効、推奨0.3)。"
                        "Kelly のステーク計算で「予想確定オッズ」を使い、早めの通知でも保守的に。"
                        "EVフィルタは元 odds で判定するので候補数は変わらない")
    p.add_argument("--persisted-ev-threshold", type=float, default=1.20,
                   help="過去バックテストで「持続シグナル」近似に使う EV 閾値（デフォルト 1.20）。"
                        "通常 EV 閾値より大きいときだけ第2バックテストを実行して比較表示")
    args = p.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()
    print(f"=== シグナル生成: {target} ===")

    races_path = RAW_DIR / "races.parquet"
    odds_path = RAW_DIR / "odds_win.csv"

    # 1. races チェック
    n_races = _races_has_date(races_path, target)
    print(f"  [1/5] races.parquet: {n_races} 件")
    if n_races == 0:
        if args.autofill:
            print("       → 不足のため scrape_dataset を実行")
            _autofill_races(target)
            n_races = _races_has_date(races_path, target)
            print(f"       再取得後: {n_races} 件")
        else:
            print(f"       ⚠ {target} のレースデータがありません。--autofill を付けるか手動で scrape してください")
            sys.exit(1)

    # 2. 当日の全場スケジュール（締切過ぎのレースを odds スクレイプから除外するため）
    schedule_path: Optional[Path] = None
    if args.autofill:
        schedule_path = _autofill_schedule_all(target, args.workers)

    # 3. odds チェック
    n_odds = _odds_has_date(odds_path, target)
    print(f"  [2/5] odds_win.csv: {n_odds} 件")
    # --refresh-odds 指定時は強制再取得
    if args.refresh_odds and args.autofill:
        print("       → --refresh-odds 指定のため強制再取得（--resume 無効化）")
        _autofill_odds(target, args.workers, force=True,
                       skip_past_deadline=schedule_path)
        n_odds = _odds_has_date(odds_path, target)
        print(f"       再取得後: {n_odds} 件")
    elif n_odds < n_races * 0.8:  # 80%未満なら不足とみなす（一部レース欠損は許容）
        if args.autofill:
            print("       → 不足のため scrape_odds を実行")
            _autofill_odds(target, args.workers,
                           skip_past_deadline=schedule_path)
            n_odds = _odds_has_date(odds_path, target)
            print(f"       再取得後: {n_odds} 件")
        else:
            print(f"       ⚠ {target} のオッズが不足。--autofill 推奨")

    # 3. features
    if args.skip_features:
        print("  [3/6] features 再生成: スキップ")
    else:
        print("  [3/6] features 再生成中...")
        _rebuild_features()

    # 4. backtest（一回目: ベット候補を抽出する目的）
    print("  [4/6] バックテスト実行中（候補抽出）...")
    bets_csv = _run_backtest(
        target,
        ev_threshold=args.ev_threshold,
        max_odds=args.max_odds,
        min_odds=args.min_odds,
        kelly_fraction=args.kelly_fraction,
        excluded_venues=args.exclude_venues,
        odds_shrinkage_max=args.odds_shrinkage_max,
        watchlist_ev_threshold=(args.watchlist_ev_threshold
                                if args.watchlist_ev_threshold > 0 else None),
    )

    # 5. 結果取得 + 締切時刻取得 → backtest 再実行
    # 対象 = 現候補 + 当日 signal_log の race_ids (= 1度でも候補化したレース)。
    # これにより、候補外になったレースの結果も payouts に追加され、HTML の
    # 「本日のシグナル履歴」セクションに 1着/外れ/PnL が反映される。
    bets_count = len(pd.read_csv(bets_csv)) if bets_csv.exists() else 0
    if not args.no_refresh_results:
        combined_target = _build_results_target_csv(target, bets_csv)
        if combined_target is not None:
            n_target = len(pd.read_csv(combined_target))
            print(f"  [5/6] 結果取得 + 締切時刻取得中: {n_target}件 (現候補+履歴)...")
            _refresh_results_html(target, args.workers, only_bets=combined_target)
            _refresh_schedule(target, args.workers, only_bets=combined_target)
            # 現候補の hit/PnL を反映するため backtest を再実行（候補ありの場合のみ）
            if bets_count:
                print(f"        → 結果取得後にバックテスト再実行（PnL反映）")
                bets_csv = _run_backtest(
                    target,
                    ev_threshold=args.ev_threshold,
                    max_odds=args.max_odds,
                    min_odds=args.min_odds,
                    kelly_fraction=args.kelly_fraction,
                    excluded_venues=args.exclude_venues,
                    odds_shrinkage_max=args.odds_shrinkage_max,
                )
        else:
            print(f"  [5/6] 結果取得対象なし（現候補ゼロ + 履歴空）")
    else:
        print("  [5/6] HTML結果取得: スキップ")

    # 6. 過去バックテスト集計（オプション・通常は手動実行時のみ）
    history: Optional[tuple[Path, str, str, Optional[Path]]] = None
    if not args.no_backtest_summary:
        print(f"  [6/7] 過去 {args.backtest_days}日のバックテスト集計中（通常+持続）...")
        history = _run_historical_backtest(
            target,
            days=args.backtest_days,
            ev_threshold=args.ev_threshold,
            max_odds=args.max_odds,
            min_odds=args.min_odds,
            kelly_fraction=args.kelly_fraction,
            excluded_venues=args.exclude_venues,
            persisted_ev_threshold=args.persisted_ev_threshold,
        )

    # 6.5. 今日のシグナル履歴に現在の bets を追記（候補から外れたものも追跡できるように）
    signal_log = _append_signal_snapshot(target, bets_csv)

    # 7. HTML レポート
    step = "7/7" if not args.no_backtest_summary else "6/6"
    print(f"  [{step}] HTML レポート生成中...")
    html_path = _make_report(bets_csv, target, args.ev_threshold,
                              history=history, signal_log=signal_log)

    print()
    print("=" * 60)
    print(f"✓ 完了: {html_path}")
    print(f"  open in browser: file://{html_path.resolve()}")
    print("=" * 60)

    if not args.no_open:
        url = html_path.resolve().as_uri()
        try:
            webbrowser.open(url)
        except Exception as e:
            logger.warning("ブラウザ自動起動に失敗: %s (手動で開いてください)", e)


if __name__ == "__main__":
    main()
