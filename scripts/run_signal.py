"""1コマンドで特定日のシグナルレポートを生成する統合パイプライン。

データはローカルの CSV/Parquet を優先し、不足分のみ自動スクレイプする。
HTML（recommend_today.py のような）は使わず、安定した LZH + odds CSV ベース。

使い方:
    # 昨日のシグナルを1コマンドで（既にキャッシュ済みなら数秒で終わる）
    python -m scripts.run_signal --date 2026-05-11

    # 不足データを自動取得（races.parquet に該当日が無ければ scrape）
    python -m scripts.run_signal --date 2026-05-11 --autofill

    # 特定オプション
    python -m scripts.run_signal --date 2026-05-11 \\
        --ev-threshold 1.05 --max-odds 10.0 \\
        --exclude-venues 04 03 02 14 01 24

挙動:
    1. races.parquet に該当日のデータがあるか確認
       - 無ければ --autofill 指定時に scrape_dataset を自動実行
    2. odds_win.csv に該当日のオッズが揃っているか確認
       - 不足分があれば --autofill 指定時に scrape_odds を自動実行
    3. features.parquet を最新化（モデル学習はスキップ）
    4. backtest を該当日に絞って実行
    5. HTML レポートを生成
    6. レポートのパスを表示
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from datetime import date
from pathlib import Path

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
    """odds_win.csv に該当日の race_id 数を返す。"""
    if not odds_path.exists():
        return 0
    df = pd.read_csv(odds_path, usecols=["race_id"])
    return int(df[df["race_id"].astype(str).str.startswith(target.strftime("%Y%m%d"))]["race_id"].nunique())


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


def _autofill_odds(target: date, workers: int) -> None:
    """odds_win.csv に該当日のオッズが無ければ取得。"""
    _run([sys.executable, "-m", "scripts.scrape_odds",
          "--from", target.isoformat(), "--to", target.isoformat(),
          "--resume",
          "--races-from", str(RAW_DIR / "races.parquet"),
          "--workers", str(workers)])


def _rebuild_features() -> None:
    """features.parquet を再生成（モデル学習はスキップ）。"""
    _run([sys.executable, "-m", "scripts.train_model",
          "--input", str(RAW_DIR / "races.parquet"),
          "--target", "is_win",
          "--features-only"])


def _run_backtest(target: date, *, ev_threshold: float, max_odds: float,
                  kelly_fraction: float, excluded_venues: list[str]) -> Path:
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
    if excluded_venues:
        cmd += ["--exclude-venues", *excluded_venues]
    _run(cmd)
    return MODELS_DIR / "backtest" / "bets_lane1_kelly.csv"


def _make_report(bets_csv: Path, target: date, ev_threshold: float) -> Path:
    out = PROCESSED_DIR / f"signals_{target.strftime('%Y%m%d')}.html"
    _run([sys.executable, "-m", "scripts.signal_report",
          "--bets", str(bets_csv),
          "--out", str(out),
          "--ev-threshold", str(ev_threshold),
          "--date-label", target.isoformat()])
    return out


# ---- メイン ----

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", required=True, help="シグナル対象日 (YYYY-MM-DD)")
    p.add_argument("--autofill", action="store_true",
                   help="races.parquet / odds_win.csv に該当日が無ければ自動でスクレイプ")
    p.add_argument("--skip-features", action="store_true",
                   help="features.parquet の再生成をスキップ（既に最新の場合）")
    p.add_argument("--ev-threshold", type=float, default=1.05)
    p.add_argument("--max-odds", type=float, default=10.0)
    p.add_argument("--kelly-fraction", type=float, default=0.25)
    p.add_argument("--exclude-venues", nargs="*",
                   default=["04", "03", "02", "14", "01", "24"],
                   help="除外場コード（デフォルト: 平和島/江戸川/戸田/鳴門/桐生/大村）")
    p.add_argument("--workers", type=int, default=4,
                   help="スクレイプの並列数（autofill 時のみ使う）")
    p.add_argument("--no-open", action="store_true",
                   help="生成後にブラウザを自動で開かない（デフォルトは開く）")
    args = p.parse_args()

    target = date.fromisoformat(args.date)
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
    if n_odds < n_races * 0.8:  # 80%未満なら不足とみなす（一部レース欠損は許容）
        if args.autofill:
            print("       → 不足のため scrape_odds を実行")
            _autofill_odds(target, args.workers)
            n_odds = _odds_has_date(odds_path, target)
            print(f"       再取得後: {n_odds} 件")
        else:
            print(f"       ⚠ {target} のオッズが不足。--autofill 推奨")

    # 3. features
    if args.skip_features:
        print("  [3/5] features 再生成: スキップ")
    else:
        print("  [3/5] features 再生成中...")
        _rebuild_features()

    # 4. backtest
    print("  [4/5] バックテスト実行中...")
    bets_csv = _run_backtest(
        target,
        ev_threshold=args.ev_threshold,
        max_odds=args.max_odds,
        kelly_fraction=args.kelly_fraction,
        excluded_venues=args.exclude_venues,
    )

    # 5. HTML レポート
    print("  [5/5] HTML レポート生成中...")
    html_path = _make_report(bets_csv, target, args.ev_threshold)

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
