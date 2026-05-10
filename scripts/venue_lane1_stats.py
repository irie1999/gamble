"""場ごとの1コース勝率と lane1_value 戦略でのROIを集計し、除外候補を提示する。

背景: 場によって1コース有利/不利が大きく異なる（広い水面・気象・1マークの
構造などが原因）。lane1_value 戦略は1号艇に賭けるので、構造的に1コースが弱い
場を除外できれば効率が上がる。

使い方:
    # 全場の1コース勝率（地のベース）
    python -m scripts.venue_lane1_stats --payouts data/raw/races_payouts.parquet

    # lane1_value で実際に賭けた場合の場別ROIまで一括（要 odds_win.csv）
    python -m scripts.venue_lane1_stats \\
        --payouts data/raw/races_payouts.parquet \\
        --features data/processed/features.parquet \\
        --odds data/raw/odds_win.csv \\
        --since 2026-03-01 --ev-threshold 1.05
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.config import VENUE_CODES


def _read(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def _venue_from_race_id(race_id: pd.Series) -> pd.Series:
    return race_id.astype(str).str.split("-").str[1]


def lane1_base_rate(payouts: pd.DataFrame, since: Optional[str]) -> pd.DataFrame:
    """払戻データから「実勝者=1コース」の頻度を場別に集計。"""
    if since and "race_date" in payouts.columns:
        payouts = payouts[pd.to_datetime(payouts["race_date"]) >= pd.Timestamp(since)]
    df = payouts.copy()
    df["venue"] = _venue_from_race_id(df["race_id"])
    df["lane1_won"] = (df["winner_lane"] == 1).astype(int)
    g = df.groupby("venue").agg(
        n_races=("race_id", "count"),
        lane1_win_rate=("lane1_won", "mean"),
    ).reset_index()
    g["venue_name"] = g["venue"].map(VENUE_CODES)
    return g.sort_values("lane1_win_rate")


def lane1_value_roi(
    features: pd.DataFrame,
    payouts: pd.DataFrame,
    odds: pd.DataFrame,
    since: Optional[str],
    ev_threshold: float,
) -> pd.DataFrame:
    """lane1_value 相当のベットを再現し、場別ROIを集計。"""
    from src.models.predict import predict_win_probability
    from src.strategy.backtest import BacktestConfig, run_backtest

    feats = features
    if since and "race_date" in feats.columns:
        feats = feats[pd.to_datetime(feats["race_date"]) >= pd.Timestamp(since)]
    pred = predict_win_probability(feats)

    cfg = BacktestConfig(strategy="lane1_value", ev_threshold=ev_threshold)
    result = run_backtest(pred, payouts, odds_df=odds, config=cfg)
    bets = result.bets.copy()
    if bets.empty:
        return pd.DataFrame(columns=["venue", "venue_name", "n_bets", "hit_rate", "roi", "pnl"])
    bets["venue"] = _venue_from_race_id(bets["race_id"])
    g = bets.groupby("venue").agg(
        n_bets=("race_id", "count"),
        hit=("hit", "sum"),
        stake=("stake", "sum"),
        ret=("return_yen", "sum"),
    ).reset_index()
    g["hit_rate"] = g["hit"] / g["n_bets"]
    g["roi"] = (g["ret"] - g["stake"]) / g["stake"].clip(lower=1)
    g["pnl"] = g["ret"] - g["stake"]
    g["venue_name"] = g["venue"].map(VENUE_CODES)
    return g.sort_values("roi")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--payouts", required=True)
    p.add_argument("--features", default=None, help="指定すると ROI 集計も行う")
    p.add_argument("--odds", default=None)
    p.add_argument("--since", default=None)
    p.add_argument("--ev-threshold", type=float, default=1.05)
    p.add_argument("--min-bets", type=int, default=3,
                   help="ROI集計でこのベット数未満の場は除外候補判断から除く")
    args = p.parse_args()

    payouts = _read(Path(args.payouts))
    base = lane1_base_rate(payouts, args.since)
    print("\n=== 場別 1コース勝率（ベースレート） ===")
    print(f"{'場':>4} {'名前':<6} {'n_races':>8} {'1コース勝率':>10}")
    for _, r in base.iterrows():
        print(f"  {r['venue']:>2} {r['venue_name']:<6} {int(r['n_races']):>8d}  {r['lane1_win_rate']:.3f}")

    if not args.features or not args.odds:
        print("\n(--features と --odds を渡すと lane1_value のROI集計もできます)")
        return

    features = _read(Path(args.features))
    odds = _read(Path(args.odds))
    roi = lane1_value_roi(features, payouts, odds, args.since, args.ev_threshold)
    print(f"\n=== 場別 lane1_value ROI (ev>={args.ev_threshold}, since={args.since}) ===")
    print(f"{'場':>4} {'名前':<6} {'n_bets':>7} {'hit':>6} {'roi':>8} {'pnl':>10}")
    for _, r in roi.iterrows():
        print(f"  {r['venue']:>2} {r['venue_name']:<6} {int(r['n_bets']):>7d}  "
              f"{r['hit_rate']:.3f}  {r['roi']:>+7.3f}  ¥{r['pnl']:>+8,.0f}")

    # 除外候補
    candidates = roi[(roi["n_bets"] >= args.min_bets) & (roi["roi"] < 0)]
    if len(candidates):
        codes = " ".join(candidates["venue"].tolist())
        print(f"\n→ 除外推奨 (n_bets>={args.min_bets} かつ ROI<0): {codes}")
        print(f"   --exclude-venues {codes}")
    else:
        print("\n→ 除外推奨候補なし")


if __name__ == "__main__":
    main()
