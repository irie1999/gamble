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


def _run_range_backtest(start: date, end: date) -> Path | None:
    """期間 [start, end] を1回の backtest で実行し bets CSV パスを返す。

    日ごとに個別実行するより数十倍速い（features 読込・model 推論が1回で済む）。
    """
    out_dir = MODELS_DIR / "backtest" / "_backfill_range"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "scripts.backtest",
        "--features", str(PROCESSED_DIR / "features.parquet"),
        "--payouts", str(RAW_DIR / "races_payouts.parquet"),
        "--odds", str(RAW_DIR / "odds_win.csv"),
        "--strategy", "lane1_kelly",
        "--since", start.isoformat(),
        "--until", end.isoformat(),
        "--kelly-fraction", str(DEFAULT_KELLY_FRACTION),
        "--ev-threshold", str(DEFAULT_EV_THRESHOLD),
        "--min-odds", str(DEFAULT_MIN_ODDS),
        "--max-odds", str(DEFAULT_MAX_ODDS),
        "--exclude-venues", *DEFAULT_EXCLUDE_VENUES,
        "--out", str(out_dir),
    ]
    print(f"  期間 backtest 実行中 ({start} 〜 {end})...", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if res.returncode != 0:
        logger.warning("期間 backtest 失敗 rc=%d", res.returncode)
        if res.stderr:
            print(res.stderr[-2000:])
        return None
    return out_dir / "bets_lane1_kelly.csv"


def _split_bets_by_date(bets_csv: Path, force: bool) -> tuple[int, int, int]:
    """bets CSV を日付ごとに分割し signal_snapshots ファイルに書き出す。

    Returns:
        (succeeded_days, skipped_days, empty_days)
    """
    if not bets_csv.exists():
        return (0, 0, 0)
    df = pd.read_csv(bets_csv)
    if df.empty or "race_date" not in df.columns:
        return (0, 0, 0)
    df["_d"] = pd.to_datetime(df["race_date"]).dt.date

    succeeded = 0
    skipped = 0
    empty = 0
    for d, g in df.groupby("_d"):
        snap_path = _snapshot_path(d)
        if snap_path.exists() and not force:
            skipped += 1
            continue
        if g.empty:
            empty += 1
            continue
        snap = pd.DataFrame({
            "race_id": g["race_id"].astype(str),
            "snapshot_at": f"{d.isoformat()} 23:59:00",
            "ev": g.get("ev", 0.0),
            "odds_win": g.get("odds_win", 0.0),
            "stake": g["stake"].astype(int) if "stake" in g.columns else 0,
            "blended_win_prob": g.get("blended_win_prob", g.get("pred_win_prob", 0.0)),
            "race_finished": g.get("race_finished", False),
            "hit": g.get("hit", False),
            "pnl": g["pnl"].astype(int) if "pnl" in g.columns else 0,
            "winner_lane": g.get("winner_lane", ""),
        })
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        snap.to_csv(snap_path, index=False)
        succeeded += 1
    return (succeeded, skipped, empty)


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

    n_days = (end - start).days + 1
    print(f"backfill 対象: {start} 〜 {end} ({n_days}日)")

    # 期間が長い場合は1回の backtest で全部処理（数十倍速い）
    if n_days >= 7:
        bets_csv = _run_range_backtest(start, end)
        if bets_csv is None:
            print("✗ 期間 backtest 失敗")
            return
        succeeded, skipped, empty = _split_bets_by_date(bets_csv, args.force)
        failed = 0
        # 候補ゼロ日（bets CSV に出てこない日）を計算
        all_target_days = {start + timedelta(days=i) for i in range(n_days)}
        df = pd.read_csv(bets_csv)
        if not df.empty and "race_date" in df.columns:
            bet_days = set(pd.to_datetime(df["race_date"]).dt.date)
        else:
            bet_days = set()
        no_bet_days = len(all_target_days - bet_days)
        empty = max(empty, no_bet_days)
    else:
        # 日数少ない時は個別実行（旧フロー）
        skipped = succeeded = failed = empty = 0
        cur = start
        while cur <= end:
            snap_path = _snapshot_path(cur)
            if snap_path.exists() and not args.force:
                skipped += 1
                cur += timedelta(days=1)
                continue
            print(f"  {cur}: backtest 実行中...", end=" ", flush=True)
            bets_csv = _run_one_day_backtest(cur)
            if bets_csv is None:
                failed += 1
                print("✗ 失敗")
                cur += timedelta(days=1)
                continue
            snap = _bets_to_snapshot(bets_csv, cur)
            if snap.empty:
                empty += 1
                print("候補ゼロ → スキップ")
            else:
                snap_path.parent.mkdir(parents=True, exist_ok=True)
                snap.to_csv(snap_path, index=False)
                succeeded += 1
                print(f"✓ {len(snap)}件 → {snap_path.name}")
            cur += timedelta(days=1)

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
