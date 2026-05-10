"""LightGBM による1着確率モデルの学習。

レース内6艇でソフトマックス正規化して各艇の1着確率を出す。
時系列リークを避けるため、学習は古い→新しい順に時系列分割で評価する。

Isotonic 回帰によるキャリブレーション付き:
  生のLightGBM出力は穴艇の確率を過大評価しがち（バックテストで穴狙いが
  大爆死する原因）。学習後、検証セット前半で IsotonicRegression を学習し、
  予測時に必ず通すことで calibration curve を補正する。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss

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
    calib_ratio: float = 0.5  # valid のうち前半を calibration、後半をテスト用


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
    # valid を時系列で前半 (calibration) / 後半 (テスト) に分割
    valid_df = valid_df.sort_values("race_date").reset_index(drop=True)
    cut = int(len(valid_df) * config.calib_ratio)
    calib_df, test_df = valid_df.iloc[:cut].copy(), valid_df.iloc[cut:].copy()
    logger.info("train=%d, calib=%d, test=%d, features=%d",
                len(train_df), len(calib_df), len(test_df), len(FEATURE_COLUMNS))

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[config.target]
    X_calib = calib_df[FEATURE_COLUMNS]
    y_calib = calib_df[config.target]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[config.target]

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
        eval_set=[(X_calib, y_calib)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False)],
    )

    # ----- Isotonic キャリブレーション -----
    # calibration set 上で raw 予測 → 真の確率の写像を学習する
    calib_raw = model.predict_proba(X_calib)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(calib_raw, y_calib)

    # ----- テストセットで raw vs calibrated を比較 -----
    test_raw = model.predict_proba(X_test)[:, 1]
    test_cal = calibrator.transform(test_raw)
    test_df = test_df.assign(raw_score=test_raw, cal_score=test_cal)
    test_df["pred_win_prob_raw"] = (
        test_df.groupby("race_id")["raw_score"].transform(lambda s: _softmax(s.to_numpy()))
    )
    test_df["pred_win_prob"] = (
        test_df.groupby("race_id")["cal_score"].transform(lambda s: _softmax(s.to_numpy()))
    )

    metrics = {
        "test_binary_logloss_raw": float(log_loss(y_test, np.clip(test_raw, 1e-6, 1 - 1e-6))),
        "test_binary_logloss_calibrated": float(log_loss(y_test, np.clip(test_cal, 1e-6, 1 - 1e-6))),
        "test_brier_raw": float(brier_score_loss(y_test, test_raw)),
        "test_brier_calibrated": float(brier_score_loss(y_test, test_cal)),
        "test_race_softmax_logloss_raw": float(_race_logloss(test_df, target=config.target, prob_col="pred_win_prob_raw")),
        "test_race_softmax_logloss_calibrated": float(_race_logloss(test_df, target=config.target, prob_col="pred_win_prob")),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "n_train": int(len(train_df)),
        "n_calib": int(len(calib_df)),
        "n_test": int(len(test_df)),
    }
    logger.info("metrics=%s", metrics)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "lgb_win_model.joblib"
    joblib.dump({
        "model": model,
        "calibrator": calibrator,
        "feature_columns": FEATURE_COLUMNS,
    }, model_path)
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    return {"model_path": str(model_path), "metrics": metrics}


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _race_logloss(df: pd.DataFrame, *, target: str, prob_col: str = "pred_win_prob") -> float:
    losses = []
    for _, g in df.groupby("race_id"):
        if g[target].sum() != 1:
            continue
        p = g[prob_col].to_numpy().clip(1e-6, 1 - 1e-6)
        y = g[target].to_numpy()
        losses.append(-np.log(p[y == 1])[0])
    return float(np.mean(losses)) if losses else float("nan")
