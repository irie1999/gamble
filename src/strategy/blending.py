"""モデル確率と市場（公衆オッズ）情報をブレンドする。

参考:
  Benter, W. (1994). "Computer Based Horse Race Handicapping and Wagering Systems:
  A Report." In Efficiency of Racetrack Betting Markets (Hausch, Lo, Ziemba eds).
  - 結論: モデル単独より、対数オッズ空間で公衆オッズを混合した方がROIが大幅に改善する。
  - 公衆オッズはレース固有の情報（直前変更・調子等）を多く織り込んでいるため。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def odds_to_implied(odds: np.ndarray, takeout: float = 0.25) -> np.ndarray:
    """単勝オッズ→市場が示唆する確率（控除率を考慮）。

    競艇の控除率は券種により異なる。単勝はおおむね25%。
    p_market_i = (1 - takeout) / odds_i  を正規化して使うのが標準。
    """
    odds = np.asarray(odds, dtype=float)
    safe = np.where(odds > 0, odds, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(odds > 0, 1.0 / safe, 0.0)
    s = raw.sum()
    if s <= 0:
        return np.full_like(raw, 1.0 / len(raw))
    return raw / s


def benter_blend(
    p_model: np.ndarray,
    p_market: np.ndarray,
    *,
    alpha: float = 0.7,
) -> np.ndarray:
    """Benter式の対数線形ブレンド。

    p_blend ∝ p_model^alpha * p_market^(1 - alpha)

    alpha=1.0  で完全にモデル、alpha=0.0 で完全に市場。
    Benter は 0.7 前後を経験的に推奨（推定値はモデル品質依存）。
    """
    p_model = np.clip(np.asarray(p_model, dtype=float), 1e-9, 1.0)
    p_market = np.clip(np.asarray(p_market, dtype=float), 1e-9, 1.0)
    log_blend = alpha * np.log(p_model) + (1 - alpha) * np.log(p_market)
    blend = np.exp(log_blend - log_blend.max())
    return blend / blend.sum()


def add_blended_probability(
    pred_df: pd.DataFrame,
    *,
    odds_col: str = "odds_win",
    out_col: str = "blended_win_prob",
    alpha: float = 0.7,
    takeout: float = 0.25,
) -> pd.DataFrame:
    """`pred_win_prob` と `odds_win` からブレンド確率を付与。"""
    out = pred_df.copy()
    blended = []
    for _, g in out.groupby("race_id", sort=False):
        if odds_col in g.columns and g[odds_col].notna().all():
            p_market = odds_to_implied(g[odds_col].to_numpy(), takeout=takeout)
        else:
            p_market = np.full(len(g), 1.0 / len(g))
        p_model = g["pred_win_prob"].to_numpy()
        blended.append(pd.Series(
            benter_blend(p_model, p_market, alpha=alpha),
            index=g.index,
        ))
    out[out_col] = pd.concat(blended).sort_index()
    return out
