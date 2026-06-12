"""保存したデータセットから特徴量を作成しモデルを学習する。

使い方:
    # 特徴量生成 + 学習（モデル新規作成 or 更新したい時）
    python -m scripts.train_model --input data/raw/races.parquet

    # 過去1年をOOS検証用に空けたい場合（=2025-05-10以前のみで学習）
    python -m scripts.train_model --input data/raw/races.parquet --train-until 2025-05-10

    # 特徴量だけ更新（毎日の運用ではこれ。モデルは固定で使う）
    python -m scripts.train_model --input data/raw/races.parquet --features-only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features.feature_engineering import build_features
from src.models.train import TrainConfig, train_model
from src.utils.config import PROCESSED_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="races.parquet または .csv")
    parser.add_argument("--target", default="is_win", choices=["is_win", "is_top2", "is_top3"])
    parser.add_argument(
        "--train-until",
        default=None,
        help="この日付以前のレースだけで学習。これ以降を OOS バックテスト期間にできる（YYYY-MM-DD）",
    )
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="特徴量だけ再生成しモデル学習はスキップ（毎日の運用用）",
    )
    args = parser.parse_args()

    p = Path(args.input)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    logger.info("読み込み: %s rows=%d", p, len(df))

    features = build_features(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
    logger.info("特徴量保存: %s rows=%d", PROCESSED_DIR / "features.parquet", len(features))

    if args.features_only:
        logger.info("--features-only 指定のため学習はスキップ")
        return

    train_features = features
    if args.train_until:
        cutoff = pd.Timestamp(args.train_until)
        train_features = features[pd.to_datetime(features["race_date"]) < cutoff]
        logger.info("学習対象を %s 以前に絞り込み: %d → %d rows",
                    cutoff.date(), len(features), len(train_features))

    cfg = TrainConfig(target=args.target)
    result = train_model(train_features, cfg)
    logger.info("学習完了: %s", result)


if __name__ == "__main__":
    main()
