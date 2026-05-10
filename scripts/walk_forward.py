"""Walk-forward / Out-of-sample 検証。

評価期間を時系列で2分割し、前半でしきい値をチューニング、後半で純粋な
out-of-sample 評価を行う。バックテストでパラメータを少し変えて高ROIを
出してしまうハインドサイト・バイアスを排除するための検証手順。

使い方:
    python -m scripts.walk_forward \
        --features data/processed/features.parquet \
        --payouts data/raw/races_payouts.parquet \
        --odds data/raw/odds_win.csv \
        --strategy lane1_value \
        --since 2026-03-01

複数の ev_threshold をスキャンし、前半で最良だった閾値を後半に適用する。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.models.predict import predict_win_probability
from src.strategy.backtest import BacktestConfig, run_backtest
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _read_table(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def _eval_threshold(
    pred: pd.DataFrame,
    payouts: pd.DataFrame,
    odds: Optional[pd.DataFrame],
    *,
    strategy: str,
    ev_threshold: float,
    blend_alpha: float,
    excluded_venues: tuple[str, ...] = (),
) -> dict:
    cfg = BacktestConfig(
        strategy=strategy,
        ev_threshold=ev_threshold,
        blend_alpha=blend_alpha,
        excluded_venues=excluded_venues,
    )
    result = run_backtest(pred, payouts, odds_df=odds, config=cfg)
    return result.summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--payouts", required=True)
    parser.add_argument("--odds", default=None)
    parser.add_argument("--strategy", default="lane1_value",
                        choices=["lane1_value", "lane1_kelly", "kelly", "flat"])
    parser.add_argument("--since", default=None, help="この日付以降に評価範囲を絞る")
    parser.add_argument("--blend-alpha", type=float, default=0.7)
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=[1.00, 1.05, 1.10, 1.15, 1.20, 1.30],
        help="スキャンする ev_threshold のリスト",
    )
    parser.add_argument(
        "--exclude-venues",
        nargs="*",
        default=[],
        help="lane1_value/lane1_kelly から除外する場（例: 24 で大村除外）",
    )
    args = parser.parse_args()

    features = _read_table(Path(args.features))
    payouts = _read_table(Path(args.payouts))
    odds = _read_table(Path(args.odds)) if args.odds else None

    if args.since and "race_date" in features.columns:
        features = features[pd.to_datetime(features["race_date"]) >= pd.Timestamp(args.since)]

    # 期間の前半/後半に2分割
    pred = predict_win_probability(features)
    pred["race_date"] = pd.to_datetime(pred["race_date"])
    sorted_dates = pred["race_date"].sort_values().unique()
    mid = sorted_dates[len(sorted_dates) // 2]
    train_pred = pred[pred["race_date"] < mid].copy()
    test_pred = pred[pred["race_date"] >= mid].copy()
    logger.info("split: train_dates < %s (%d races) / test_dates >= %s (%d races)",
                mid.date(), train_pred["race_id"].nunique(),
                mid.date(), test_pred["race_id"].nunique())

    # 前半でしきい値スキャン
    print("\n=== チューニング期間（前半） ===")
    print(f"{'ev_thr':>8} {'n_bets':>8} {'hit':>6} {'roi':>8}")
    train_results = {}
    for thr in args.thresholds:
        s = _eval_threshold(train_pred, payouts, odds,
                            strategy=args.strategy, ev_threshold=thr,
                            blend_alpha=args.blend_alpha,
                            excluded_venues=tuple(args.exclude_venues))
        train_results[thr] = s
        print(f"{thr:>8.2f} {s['n_bets']:>8d} {s['hit_rate']:>6.3f} {s['roi']:>+8.3f}")

    # 最良閾値（n_bets >= 10 の中で最大ROI）
    valid = [(thr, r) for thr, r in train_results.items() if r["n_bets"] >= 10]
    if not valid:
        valid = list(train_results.items())
    best_thr, best_train = max(valid, key=lambda x: x[1]["roi"])
    print(f"\n→ 前半の最良: ev_threshold={best_thr} (roi={best_train['roi']:+.3f})")

    # 後半で純粋検証
    print("\n=== 検証期間（後半・OOS） ===")
    s_oos = _eval_threshold(test_pred, payouts, odds,
                            strategy=args.strategy, ev_threshold=best_thr,
                            blend_alpha=args.blend_alpha,
                            excluded_venues=tuple(args.exclude_venues))
    print(f"  ev_threshold = {best_thr}")
    for k, v in s_oos.items():
        print(f"  {k}: {v}")

    # 参考: 後半で全閾値を見てチューニングのバイアスを判定
    print("\n=== 参考: 後半で全閾値スキャン ===")
    print(f"{'ev_thr':>8} {'n_bets':>8} {'hit':>6} {'roi':>8}")
    for thr in args.thresholds:
        s = _eval_threshold(test_pred, payouts, odds,
                            strategy=args.strategy, ev_threshold=thr,
                            blend_alpha=args.blend_alpha,
                            excluded_venues=tuple(args.exclude_venues))
        marker = " ← 採用" if abs(thr - best_thr) < 1e-9 else ""
        print(f"{thr:>8.2f} {s['n_bets']:>8d} {s['hit_rate']:>6.3f} {s['roi']:>+8.3f}{marker}")


if __name__ == "__main__":
    main()
