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


def _autofill_odds(target: date, workers: int, *, force: bool = False) -> None:
    """odds_win.csv に該当日のオッズを取得。

    force=True なら --resume を付けず既存レースも再スクレイプする（オッズは
    締切に向けて動くため、watch_signal の定期実行では force=True が必要）。
    """
    cmd = [sys.executable, "-m", "scripts.scrape_odds",
           "--from", target.isoformat(), "--to", target.isoformat(),
           "--races-from", str(RAW_DIR / "races.parquet"),
           "--workers", str(workers)]
    if not force:
        cmd.append("--resume")
    _run(cmd)


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
                  excluded_venues: list[str]) -> Path:
    """指定日1日分のバックテスト → bets_lane1_kelly.csv のパスを返す。"""
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
    _run(cmd)
    return MODELS_DIR / "backtest" / "bets_lane1_kelly.csv"


def _run_historical_backtest(
    target: date, *, days: int, ev_threshold: float, max_odds: float,
    min_odds: Optional[float], kelly_fraction: float,
    excluded_venues: list[str],
) -> Optional[tuple[Path, str, str]]:
    """target 当日を含まない過去 days 日のバックテストを実行。

    成功時は (出力ディレクトリ, since, until) を返す。失敗時は None。
    出力先は data/processed/backtest_history/ で、当日用 bets と分離する。
    """
    since = (target - timedelta(days=days)).isoformat()
    until = (target - timedelta(days=1)).isoformat()
    out_dir = PROCESSED_DIR / "backtest_history"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "scripts.backtest",
        "--features", str(PROCESSED_DIR / "features.parquet"),
        "--payouts", str(RAW_DIR / "races_payouts.parquet"),
        "--odds", str(RAW_DIR / "odds_win.csv"),
        "--strategy", "lane1_kelly",
        "--since", since,
        "--until", until,
        "--kelly-fraction", str(kelly_fraction),
        "--ev-threshold", str(ev_threshold),
        "--max-odds", str(max_odds),
        "--out", str(out_dir),
    ]
    if min_odds is not None:
        cmd += ["--min-odds", str(min_odds)]
    if excluded_venues:
        cmd += ["--exclude-venues", *excluded_venues]
    logger.info("実行: %s", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        logger.warning("過去バックテスト失敗 rc=%d (HTMLには載せず続行)", res.returncode)
        return None
    return out_dir, since, until


def _make_report(bets_csv: Path, target: date, ev_threshold: float,
                 history: Optional[tuple[Path, str, str]] = None) -> Path:
    out = PROCESSED_DIR / f"signals_{target.strftime('%Y%m%d')}.html"
    cmd = [sys.executable, "-m", "scripts.signal_report",
           "--bets", str(bets_csv),
           "--out", str(out),
           "--ev-threshold", str(ev_threshold),
           "--date-label", target.isoformat()]
    if history is not None:
        hist_dir, since, until = history
        summary_file = hist_dir / "summary_lane1_kelly.json"
        equity_file = hist_dir / "equity_lane1_kelly.csv"
        if summary_file.exists():
            cmd += ["--history-summary", str(summary_file),
                    "--history-since", since,
                    "--history-until", until]
            if equity_file.exists():
                cmd += ["--history-equity", str(equity_file)]
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

    # 2. odds チェック
    n_odds = _odds_has_date(odds_path, target)
    print(f"  [2/5] odds_win.csv: {n_odds} 件")
    # --refresh-odds 指定時は強制再取得
    if args.refresh_odds and args.autofill:
        print("       → --refresh-odds 指定のため強制再取得（--resume 無効化）")
        _autofill_odds(target, args.workers, force=True)
        n_odds = _odds_has_date(odds_path, target)
        print(f"       再取得後: {n_odds} 件")
    elif n_odds < n_races * 0.8:  # 80%未満なら不足とみなす（一部レース欠損は許容）
        if args.autofill:
            print("       → 不足のため scrape_odds を実行")
            _autofill_odds(target, args.workers)
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
    )

    # 5. ベット候補だけ結果取得 + 締切時刻取得 → backtest 再実行
    if not args.no_refresh_results and bets_csv.exists():
        bets_count = len(pd.read_csv(bets_csv))
        if bets_count:
            print(f"  [5/6] ベット候補 {bets_count}件 の結果＋締切時刻を取得中...")
            _refresh_results_html(target, args.workers, only_bets=bets_csv)
            _refresh_schedule(target, args.workers, only_bets=bets_csv)
            # payouts が更新されたので backtest を再実行して hit/PnL を反映
            print(f"        → 結果取得後にバックテスト再実行（PnL反映）")
            bets_csv = _run_backtest(
                target,
                ev_threshold=args.ev_threshold,
                max_odds=args.max_odds,
                min_odds=args.min_odds,
                kelly_fraction=args.kelly_fraction,
                excluded_venues=args.exclude_venues,
            )
        else:
            print(f"  [5/6] ベット候補なし → 結果取得スキップ")
    else:
        print("  [5/6] HTML結果取得: スキップ")

    # 6. 過去バックテスト集計（オプション・通常は手動実行時のみ）
    history: Optional[tuple[Path, str, str]] = None
    if not args.no_backtest_summary:
        print(f"  [6/7] 過去 {args.backtest_days}日のバックテスト集計中...")
        history = _run_historical_backtest(
            target,
            days=args.backtest_days,
            ev_threshold=args.ev_threshold,
            max_odds=args.max_odds,
            min_odds=args.min_odds,
            kelly_fraction=args.kelly_fraction,
            excluded_venues=args.exclude_venues,
        )

    # 7. HTML レポート
    step = "7/7" if not args.no_backtest_summary else "6/6"
    print(f"  [{step}] HTML レポート生成中...")
    html_path = _make_report(bets_csv, target, args.ev_threshold, history=history)

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
