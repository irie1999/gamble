"""バックテスト結果の bets_*.csv を分析して、勝ちパターン・負けパターンを抽出。

使い方:
    # backtest 実行後に
    python -m scripts.analyze_bets --bets data/models/backtest/bets_lane1_value.csv

    # 同じレース範囲を切って2分割（前半/後半）して挙動比較
    python -m scripts.analyze_bets --bets data/models/backtest/bets_lane1_value.csv --split-time
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _summarize(bets: pd.DataFrame, label: str) -> None:
    if bets.empty:
        print(f"\n=== {label} ===\n  (no bets)")
        return
    n = len(bets)
    wins = int(bets["hit"].sum()) if "hit" in bets.columns else 0
    stake = float(bets["stake"].sum())
    ret = float(bets["return_yen"].sum())
    pnl = ret - stake
    roi = pnl / stake if stake > 0 else 0
    print(f"\n=== {label} ===")
    print(f"  n_bets: {n}")
    print(f"  hit_rate: {wins/n:.3f} ({wins}勝)")
    print(f"  stake_total: ¥{stake:,.0f}")
    print(f"  return_total: ¥{ret:,.0f}")
    print(f"  pnl: ¥{pnl:,.0f}")
    print(f"  roi: {roi:+.3f}")


def _bucket_by_odds(bets: pd.DataFrame) -> None:
    if "odds_win" not in bets.columns:
        return
    print("\n--- オッズ帯別 ---")
    bins = [1.0, 2.0, 3.0, 5.0, 10.0, 100.0]
    labels = ["~2.0", "2.0-3.0", "3.0-5.0", "5.0-10.0", "10.0+"]
    bets = bets.copy()
    bets["odds_bucket"] = pd.cut(bets["odds_win"], bins=bins, labels=labels, include_lowest=True)
    for bucket in labels:
        sub = bets[bets["odds_bucket"] == bucket]
        if not len(sub):
            continue
        n = len(sub)
        wins = int(sub["hit"].sum())
        roi = (sub["return_yen"].sum() - sub["stake"].sum()) / max(sub["stake"].sum(), 1)
        print(f"  {bucket:>10}: n={n:3d}, hit={wins/n:.2f}, roi={roi:+.3f}")


def _bucket_by_venue(bets: pd.DataFrame) -> None:
    if "race_id" not in bets.columns:
        return
    bets = bets.copy()
    bets["venue"] = bets["race_id"].str.split("-").str[1]
    print("\n--- 場別 ---")
    for venue, sub in bets.groupby("venue"):
        n = len(sub)
        wins = int(sub["hit"].sum())
        roi = (sub["return_yen"].sum() - sub["stake"].sum()) / max(sub["stake"].sum(), 1)
        print(f"  場{venue}: n={n:3d}, hit={wins/n:.2f}, roi={roi:+.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bets", required=True, help="bets_*.csv")
    p.add_argument("--split-time", action="store_true", help="時系列で2分割して前半/後半比較")
    args = p.parse_args()

    bets = pd.read_csv(args.bets)
    print(f"Loaded: {args.bets}")
    print(f"columns: {list(bets.columns)}")

    _summarize(bets, "Overall")
    _bucket_by_odds(bets)
    _bucket_by_venue(bets)

    if args.split_time and "race_date" in bets.columns:
        bets["race_date"] = pd.to_datetime(bets["race_date"])
        bets = bets.sort_values("race_date").reset_index(drop=True)
        cut = len(bets) // 2
        first = bets.iloc[:cut]
        second = bets.iloc[cut:]
        print(f"\n--- 時系列2分割 (cut={cut}) ---")
        _summarize(first, "前半")
        _summarize(second, "後半（OOS的）")


if __name__ == "__main__":
    main()
