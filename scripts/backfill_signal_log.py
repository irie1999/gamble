"""指定期間の過去日について、backtest 結果を signal_snapshots 形式で生成。

watch_signal が稼働していなかった日は signal_log/signal_snapshots_*.csv が
存在しないため、show_signals_today で表示できない。本スクリプトは backtest を
日ごとに実行し、ベット候補を「合成 snapshot」として書き出す。

注意:
  - 合成 snapshot は 1 race = 1 行のみ (実時系列ではない)
  - snapshot_at = "<date> 23:59:00" 固定 → 持続判定では「事後のみ」になる
  - 「ライブで賭けたシグナル履歴」ではなく「もし当日この戦略で運用していたら」
    に近い意味合い

使い方:
    python -m scripts.backfill_signal_log --from 2026-05-01 --to 2026-05-22
    python -m scripts.backfill_signal_log --days 30           # 過去30日
    python -m scripts.backfill_signal_log --days 30 --force   # 既存も上書き
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.utils.config import MODELS_DIR, PROCESSED_DIR, RAW_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# v2 (B 設定) と同等のデフォルト
DEFAULT_KELLY_FRACTION = 0.5
DEFAULT_EV_THRESHOLD = 1.05
DEFAULT_MIN_ODDS = 3.0
DEFAULT_MAX_ODDS = 10.0
DEFAULT_EXCLUDE_VENUES = ["04", "03", "02", "14", "01", "24", "10"]


def _snapshot_path(target: date) -> Path:
    return PROCESSED_DIR / "signal_log" / f"signal_snapshots_{target:%Y%m%d}.csv"


def _run_one_day_backtest(target: date) -> Path | None:
    """target 1日分の backtest を実行し bets CSV のパスを返す。失敗時 None。"""
    out_dir = MODELS_DIR / "backtest" / "_backfill"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "scripts.backtest",
        "--features", str(PROCESSED_DIR / "features.parquet"),
        "--payouts", str(RAW_DIR / "races_payouts.parquet"),
        "--odds", str(RAW_DIR / "odds_win.csv"),
        "--strategy", "lane1_kelly",
        "--since", target.isoformat(),
        "--until", target.isoformat(),
        "--kelly-fraction", str(DEFAULT_KELLY_FRACTION),
        "--ev-threshold", str(DEFAULT_EV_THRESHOLD),
        "--min-odds", str(DEFAULT_MIN_ODDS),
        "--max-odds", str(DEFAULT_MAX_ODDS),
        "--exclude-venues", *DEFAULT_EXCLUDE_VENUES,
        "--out", str(out_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if res.returncode != 0:
        logger.warning("backtest 失敗 date=%s rc=%d", target, res.returncode)
        return None
    return out_dir / "bets_lane1_kelly.csv"


def _bets_to_snapshot(bets_csv: Path, target: date) -> pd.DataFrame:
    """backtest の bets CSV を signal_snapshots フォーマットに変換。"""
    if not bets_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(bets_csv)
    if df.empty:
        return pd.DataFrame()
    snap = pd.DataFrame({
        "race_id": df["race_id"].astype(str),
        "snapshot_at": f"{target.isoformat()} 23:59:00",  # 事後扱い
        "ev": df.get("ev", 0.0),
        "odds_win": df.get("odds_win", 0.0),
        "stake": df.get("stake", 0).astype(int) if "stake" in df.columns else 0,
        "blended_win_prob": df.get("blended_win_prob", df.get("pred_win_prob", 0.0)),
        "race_finished": df.get("race_finished", False),
        "hit": df.get("hit", False),
        "pnl": df.get("pnl", 0).astype(int) if "pnl" in df.columns else 0,
        "winner_lane": df.get("winner_lane", ""),
    })
    return snap


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="date_from", default=None, help="YYYY-MM-DD")
    p.add_argument("--to", dest="date_to", default=None, help="YYYY-MM-DD")
    p.add_argument("--days", type=int, default=None,
                   help="過去N日（今日含む）。--from/--to が無い時に使う")
    p.add_argument("--force", action="store_true",
                   help="既に signal_snapshots がある日も上書きする")
    args = p.parse_args()

    if args.date_from and args.date_to:
        start = date.fromisoformat(args.date_from)
        end = date.fromisoformat(args.date_to)
    elif args.days:
        end = date.today()
        start = end - timedelta(days=args.days - 1)
    else:
        p.error("--from と --to のペア、または --days を指定してください")
        return

    targets: list[date] = []
    cur = start
    while cur <= end:
        targets.append(cur)
        cur += timedelta(days=1)

    print(f"backfill 対象: {start} 〜 {end} ({len(targets)}日)")
    skipped = 0
    succeeded = 0
    failed = 0
    empty = 0
    for t in targets:
        snap_path = _snapshot_path(t)
        if snap_path.exists() and not args.force:
            skipped += 1
            continue
        print(f"  {t}: backtest 実行中...", end=" ", flush=True)
        bets_csv = _run_one_day_backtest(t)
        if bets_csv is None:
            failed += 1
            print("✗ 失敗")
            continue
        snap = _bets_to_snapshot(bets_csv, t)
        if snap.empty:
            empty += 1
            print("候補ゼロ → スキップ")
            continue
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        snap.to_csv(snap_path, index=False)
        succeeded += 1
        print(f"✓ {len(snap)}件 → {snap_path.name}")

    print()
    print(f"=== 完了 ===")
    print(f"  成功:      {succeeded} 日")
    print(f"  候補ゼロ:  {empty} 日")
    print(f"  失敗:      {failed} 日")
    print(f"  スキップ:  {skipped} 日（既存・--force で上書き可）")
    print()
    print("次のコマンドで履歴を確認:")
    print(f"  python -m scripts.show_signals_today --from {start} --to {end}")


if __name__ == "__main__":
    main()
