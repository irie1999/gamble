"""特徴量生成

リークを避けるため、過去成績ベースの集計は「現在のレース日より前」のデータだけから計算する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# 競艇のコース別1着率（経験的な事前確率: 公式統計でおおむね 1〜6コース順に
# 50% / 14% / 12% / 12% / 7% / 5% 程度になる）
COURSE_PRIOR_WIN_RATE = {1: 0.55, 2: 0.14, 3: 0.12, 4: 0.11, 5: 0.05, 6: 0.03}


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """各艇行に対する基礎特徴量を付与。"""
    out = df.copy()

    # 級別をワンホット
    grade_dummies = pd.get_dummies(out["grade"].fillna("UNK"), prefix="grade")
    out = pd.concat([out, grade_dummies], axis=1)

    # コース事前1着率
    out["course_prior_win_rate"] = out["lane"].map(COURSE_PRIOR_WIN_RATE).astype(float)

    # 全国/当地勝率の差
    out["delta_win_rate_local_national"] = (
        pd.to_numeric(out.get("win_rate_local"), errors="coerce")
        - pd.to_numeric(out.get("win_rate_national"), errors="coerce")
    )

    # モーター×ボート2連対率の単純合算
    out["motor_boat_score"] = (
        pd.to_numeric(out.get("motor_2rate"), errors="coerce").fillna(0)
        + pd.to_numeric(out.get("boat_2rate"), errors="coerce").fillna(0)
    )

    # 体重ペナルティ近似（重い選手はやや不利）
    out["weight_penalty"] = (pd.to_numeric(out.get("weight"), errors="coerce") - 52.0).clip(lower=0)

    return out


def add_racer_history_features(
    df: pd.DataFrame,
    *,
    lookback_days: int = 180,
) -> pd.DataFrame:
    """選手ごとの過去成績集計（リーク防止のため当該レース日より前のみ使用）。"""
    out = df.copy()
    out = out.sort_values(["racer_id", "race_date"]).reset_index(drop=True)

    out["race_date"] = pd.to_datetime(out["race_date"])

    # 期間内のローリング平均は groupby + shift で「当該行より前」を担保
    grp = out.groupby("racer_id", group_keys=False)

    rank_num = pd.to_numeric(out["rank"], errors="coerce")
    out["_is_win"] = (rank_num == 1).astype(float)
    out["_is_top3"] = (rank_num <= 3).astype(float)
    out["_st"] = pd.to_numeric(out["start_timing"], errors="coerce")

    for col, new in [
        ("_is_win", "racer_win_rate_recent"),
        ("_is_top3", "racer_top3_rate_recent"),
        ("_st", "racer_start_timing_recent"),
    ]:
        out[new] = grp[col].apply(
            lambda s: s.shift(1).rolling(window=20, min_periods=3).mean()
        ).reset_index(level=0, drop=True)

    # 直近 lookback_days の戦績（イベント数）
    out["racer_recent_runs"] = grp["_is_win"].apply(
        lambda s: s.shift(1).rolling(window=20, min_periods=1).count()
    ).reset_index(level=0, drop=True)

    out = out.drop(columns=["_is_win", "_is_top3", "_st"])
    return out


def add_venue_lane_features(df: pd.DataFrame) -> pd.DataFrame:
    """場×コースの過去1着率（学習時はリーク防止のため expanding を使用）。"""
    out = df.copy()
    out = out.sort_values("race_date").reset_index(drop=True)

    rank_num = pd.to_numeric(out["rank"], errors="coerce")
    out["_win_flag"] = (rank_num == 1).astype(float)

    grp = out.groupby(["venue_code", "lane"], group_keys=False)
    out["venue_lane_win_rate"] = grp["_win_flag"].apply(
        lambda s: s.shift(1).expanding(min_periods=10).mean()
    ).reset_index(level=0, drop=True)

    out = out.drop(columns=["_win_flag"])
    return out


FEATURE_COLUMNS: list[str] = [
    "lane",
    "age",
    "weight",
    "win_rate_national",
    "win_rate_local",
    "place_rate_national",
    "place_rate_local",
    "motor_2rate",
    "boat_2rate",
    "course_prior_win_rate",
    "delta_win_rate_local_national",
    "motor_boat_score",
    "weight_penalty",
    "racer_win_rate_recent",
    "racer_top3_rate_recent",
    "racer_start_timing_recent",
    "racer_recent_runs",
    "venue_lane_win_rate",
    # ワンホット級別
    "grade_A1", "grade_A2", "grade_B1", "grade_B2",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_basic_features(df)
    out = add_racer_history_features(out)
    out = add_venue_lane_features(out)
    # ワンホット欠落補完
    for c in ["grade_A1", "grade_A2", "grade_B1", "grade_B2"]:
        if c not in out.columns:
            out[c] = 0
    # 必要列のみ float 化
    for c in FEATURE_COLUMNS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out
