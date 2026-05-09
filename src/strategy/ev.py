"""期待値（EV）計算とベット選別。

参考:
  - Hausch, Ziemba & Rubinstein (1981). "Efficiency of the Market for Racetrack Betting."
    "favorite-longshot bias" を実証。本命に賭けるほどEVが高くなる傾向。
  - Asch, Malkiel & Quandt (1984/86) も同様の指摘。

実装方針:
  EV = p * odds  （p は真の勝率推定値、odds は払戻倍率）
  EV > threshold (例: 1.05) のときのみベット候補とする。
  さらに、推定誤差を踏まえて p の下限（例: 0.05）を要求する。
"""
from __future__ import annotations

import pandas as pd


def select_value_bets(
    pred_df: pd.DataFrame,
    *,
    prob_col: str = "blended_win_prob",
    odds_col: str = "odds_win",
    ev_threshold: float = 1.05,
    min_prob: float = 0.05,
    top_k_per_race: int = 1,
) -> pd.DataFrame:
    """EV > threshold かつ p > min_prob のベット候補を抽出。

    レース毎に EV 上位 top_k_per_race を残す。
    """
    df = pred_df.copy()
    df["ev"] = df[prob_col] * df[odds_col]

    cond = (df["ev"] > ev_threshold) & (df[prob_col] > min_prob) & (df[odds_col] > 1.0)
    bets = df[cond].copy()
    if bets.empty:
        return bets

    bets = (
        bets.sort_values(["race_id", "ev"], ascending=[True, False])
            .groupby("race_id", group_keys=False)
            .head(top_k_per_race)
    )
    return bets.reset_index(drop=True)
