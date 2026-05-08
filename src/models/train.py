"""LightGBM による1着確率モデルの学習。

レース内6艇でソフトマックス正規化して各艇の1着確率を出す。
時系列リークを避けるため、学習は古い→新しい順に時系列分割で評価する。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.features.feature_engineering import FEATURE_COLUMNS
from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainConfig:
    target: str = "is_win"
    n_estimators: int = 800
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 30
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 5
    early_stopping_rounds: int = 50
    valid_ratio: float = 0.2  # 末尾を検証


def time_split(df: pd.DataFrame, valid_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("race_date").reset_index(drop=True)
    cut = int(len(df) * (1 - valid_ratio))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def train_model(
    features_df: pd.DataFrame,
    config: TrainConfig | None = None,
) -> dict:
    config = config or TrainConfig()
    df = features_df.dropna(subset=[config.target]).copy()
    df[config.target] = df[config.target].astype(int)

    train_df, valid_df = time_split(df, config.valid_ratio)
    logger.info("train=%d, valid=%d, features=%d", len(train_df), len(valid_df), len(FEATURE_COLUMNS))

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[config.target]
    X_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df[config.target]

    model = lgb.LGBMClassifier(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        min_child_samples=config.min_child_samples,
        feature_fraction=config.feature_fraction,
        bagging_fraction=config.bagging_fraction,
        bagging_freq=config.bagging_freq,
        objective="binary",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False)],
    )

    # レース単位のソフトマックス正規化後にlog_lossを評価
    valid_df = valid_df.assign(
        raw_score=model.predict_proba(X_valid)[:, 1]
    )
    valid_df["pred_win_prob"] = (
        valid_df.groupby("race_id")["raw_score"]
        .transform(lambda s: _softmax(s.to_numpy()))
    )
    # レース内で 1艇だけ y=1 になるはずなので、レース単位 multi-class log loss も算出
    race_logloss = _race_logloss(valid_df, target=config.target)

    metrics = {
        "valid_binary_logloss": float(log_loss(y_valid, valid_df["raw_score"].clip(1e-6, 1 - 1e-6))),
        "valid_race_softmax_logloss": float(race_logloss),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
    }
    logger.info("metrics=%s", metrics)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "lgb_win_model.joblib"
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, model_path)
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    return {"model_path": str(model_path), "metrics": metrics}


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _race_logloss(df: pd.DataFrame, *, target: str) -> float:
    losses = []
    for _, g in df.groupby("race_id"):
        if g[target].sum() != 1:
            continue
        p = g["pred_win_prob"].to_numpy().clip(1e-6, 1 - 1e-6)
        y = g[target].to_numpy()
        losses.append(-np.log(p[y == 1])[0])
    return float(np.mean(losses)) if losses else float("nan")
