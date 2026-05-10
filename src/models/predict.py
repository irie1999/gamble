"""学習済みモデルを使って各艇の1着確率を計算する。"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def load_model(path: Path | None = None) -> dict:
    path = path or (MODELS_DIR / "lgb_win_model.joblib")
    return joblib.load(path)


def predict_win_probability(features_df: pd.DataFrame, *, model_path: Path | None = None) -> pd.DataFrame:
    """1艇1行のDataFrameに pred_win_prob 列を追加して返す。

    bundle に calibrator が含まれていれば isotonic 補正を通したスコアを
    使う。レース単位ソフトマックス正規化はその後に適用。
    """
    bundle = load_model(model_path)
    model = bundle["model"]
    calibrator = bundle.get("calibrator")
    feat_cols = bundle["feature_columns"]

    X = features_df[feat_cols]
    raw = model.predict_proba(X)[:, 1]
    score = calibrator.transform(raw) if calibrator is not None else raw

    out = features_df.copy()
    out["raw_score"] = score
    out["pred_win_prob"] = (
        out.groupby("race_id")["raw_score"]
        .transform(lambda s: _softmax(s.to_numpy()))
    )
    return out


def expected_value_table(pred_df: pd.DataFrame, odds: dict[int, float]) -> pd.DataFrame:
    """単勝オッズに対する各艇の期待値（=確率×オッズ）を返すユーティリティ。

    odds: {lane: 単勝オッズ} を想定。
    """
    out = pred_df.copy()
    out["odds_win"] = out["lane"].map(odds).astype(float)
    out["ev_win"] = out["pred_win_prob"] * out["odds_win"]
    return out[["race_id", "lane", "pred_win_prob", "odds_win", "ev_win"]]
