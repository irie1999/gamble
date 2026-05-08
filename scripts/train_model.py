"""保存したデータセットから特徴量を作成しモデルを学習する。

使い方:
    python -m scripts.train_model --input data/raw/races.parquet
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
    args = parser.parse_args()

    p = Path(args.input)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    logger.info("読み込み: %s rows=%d", p, len(df))

    features = build_features(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(PROCESSED_DIR / "features.parquet", index=False)

    cfg = TrainConfig(target=args.target)
    result = train_model(features, cfg)
    logger.info("学習完了: %s", result)


if __name__ == "__main__":
    main()
