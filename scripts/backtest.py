"""学習済みモデルと過去データでバックテストを実行する。

使い方:
    python -m scripts.backtest \
        --features data/processed/features.parquet \
        --payouts  data/raw/races_payouts.parquet \
        --strategy kelly \
        --since 2024-06-01

戦略:
    flat / kelly / always_top1 / model_top1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.models.predict import predict_win_probability
from src.strategy.backtest import BacktestConfig, run_backtest
from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _read_table(p: Path) -> pd.DataFrame:
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="特徴量 parquet/csv")
    parser.add_argument("--payouts", required=True, help="払戻 parquet/csv")
    parser.add_argument("--odds", default=None, help="（任意）レース確定前の単勝オッズ")
    parser.add_argument(
        "--strategy",
        default="kelly",
        choices=["flat", "kelly", "always_top1", "model_top1", "lane1_value", "lane1_kelly"],
    )
    parser.add_argument("--since", default=None, help="この日付以降のレースのみ評価 (YYYY-MM-DD)")
    parser.add_argument("--until", default=None, help="この日付以前のレースのみ評価 (YYYY-MM-DD)")
    parser.add_argument("--initial-bankroll", type=float, default=100_000.0)
    parser.add_argument("--flat-stake", type=float, default=1_000.0)
    parser.add_argument("--kelly-fraction", type=float, default=0.25)
    parser.add_argument("--ev-threshold", type=float, default=1.05)
    parser.add_argument("--blend-alpha", type=float, default=0.7)
    parser.add_argument(
        "--exclude-venues",
        nargs="*",
        default=[],
        help="1コース勝率が低い場を lane1_value/lane1_kelly から除外（例: 24 で大村除外）",
    )
    parser.add_argument(
        "--max-odds",
        type=float,
        default=None,
        help="1号艇オッズの上限。これ超のレースは除外（高オッズ1号艇=構造的弱）",
    )
    parser.add_argument("--out", default=str(MODELS_DIR / "backtest"))
    args = parser.parse_args()

    features = _read_table(Path(args.features))
    payouts = _read_table(Path(args.payouts))
    odds = _read_table(Path(args.odds)) if args.odds else None

    if args.since and "race_date" in features.columns:
        features = features[pd.to_datetime(features["race_date"]) >= pd.Timestamp(args.since)]
        logger.info("since=%s で絞り込み: %d rows", args.since, len(features))
    if args.until and "race_date" in features.columns:
        features = features[pd.to_datetime(features["race_date"]) <= pd.Timestamp(args.until)]
        logger.info("until=%s で絞り込み: %d rows", args.until, len(features))

    pred = predict_win_probability(features)
    keep_cols = ["race_id", "lane", "race_date", "pred_win_prob"]
    pred = pred[[c for c in keep_cols if c in pred.columns]]

    cfg = BacktestConfig(
        strategy=args.strategy,
        initial_bankroll=args.initial_bankroll,
        flat_stake=args.flat_stake,
        kelly_fraction=args.kelly_fraction,
        ev_threshold=args.ev_threshold,
        blend_alpha=args.blend_alpha,
        excluded_venues=tuple(args.exclude_venues),
        max_odds=args.max_odds,
    )
    result = run_backtest(pred, payouts, odds_df=odds, config=cfg)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.bets.to_csv(out_dir / f"bets_{args.strategy}.csv", index=False)
    if not result.equity_curve.empty:
        result.equity_curve.to_csv(out_dir / f"equity_{args.strategy}.csv")
    (out_dir / f"summary_{args.strategy}.json").write_text(
        json.dumps(result.summary, indent=2, ensure_ascii=False)
    )
    logger.info("=== サマリ (%s) ===", args.strategy)
    for k, v in result.summary.items():
        logger.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
