"""
特徴量エンジニアリングモジュール
スクレイピングした生データを機械学習用の特徴量に変換する
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


# インコースアドバンテージ（1号艇が最も有利）
COURSE_WIN_RATE_BASELINE = {
    1: 0.553,
    2: 0.178,
    3: 0.118,
    4: 0.082,
    5: 0.044,
    6: 0.025,
}

# 級別エンコード
CLASS_MAP = {"A1": 4, "A2": 3, "B1": 2, "B2": 1, "": 0}


def load_raw(filename: str = "raw_data.json") -> pd.DataFrame:
    path = DATA_DIR / filename
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return pd.DataFrame(records)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    生データDFから特徴量DFを生成する
    各行は1艇分のデータ
    """
    df = df.copy()

    # --- 基本特徴量 ---
    df["class_enc"] = df["class"].map(CLASS_MAP).fillna(0).astype(int)
    df["course_baseline"] = df["boat_no"].map(COURSE_WIN_RATE_BASELINE)

    # --- 成績系（欠損は中央値で補完） ---
    rate_cols = [
        "national_win_rate", "national_2rate", "national_3rate",
        "local_win_rate", "local_2rate",
        "motor_2rate", "boat_2rate",
    ]
    for col in rate_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # --- 展示タイム（欠損は同レース平均で補完） ---
    df["exhibit_time"] = pd.to_numeric(df["exhibit_time"], errors="coerce")
    race_mean_exhibit = df.groupby(["date", "venue_code", "race_no"])["exhibit_time"].transform("mean")
    df["exhibit_time"] = df["exhibit_time"].fillna(race_mean_exhibit)

    # 展示タイムをレース内偏差に変換（速いほど正の値）
    race_std_exhibit = df.groupby(["date", "venue_code", "race_no"])["exhibit_time"].transform("std").replace(0, np.nan)
    race_mean_exhibit2 = df.groupby(["date", "venue_code", "race_no"])["exhibit_time"].transform("mean")
    df["exhibit_time_zscore"] = (race_mean_exhibit2 - df["exhibit_time"]) / race_std_exhibit.fillna(1)

    # --- モーター成績をレース内相対値に変換 ---
    for col in ["motor_2rate", "national_win_rate", "local_win_rate"]:
        race_mean = df.groupby(["date", "venue_code", "race_no"])[col].transform("mean")
        race_std = df.groupby(["date", "venue_code", "race_no"])[col].transform("std").replace(0, np.nan)
        df[f"{col}_rel"] = (df[col] - race_mean) / race_std.fillna(1)

    # --- インコースアドバンテージ特徴量 ---
    df["is_course1"] = (df["boat_no"] == 1).astype(int)
    df["is_course2"] = (df["boat_no"] == 2).astype(int)
    df["is_inner"] = (df["boat_no"] <= 3).astype(int)

    # --- 複合特徴量 ---
    # A1選手 × インコース（最強コンボ）
    df["a1_inner"] = (df["class_enc"] == 4) & (df["boat_no"] <= 2)
    df["a1_inner"] = df["a1_inner"].astype(int)

    # 展示タイムとモーター成績の合成スコア
    df["speed_score"] = df["exhibit_time_zscore"] * 0.6 + df["motor_2rate_rel"] * 0.4

    # 地元成績 vs 全国成績の差（地元巧者補正）
    df["local_advantage"] = df["local_win_rate"] - df["national_win_rate"]

    # --- ターゲット ---
    df["win"] = pd.to_numeric(df["win"], errors="coerce").fillna(0).astype(int)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

    return df


FEATURE_COLS = [
    "boat_no",
    "class_enc",
    "course_baseline",
    "national_win_rate",
    "national_2rate",
    "national_3rate",
    "local_win_rate",
    "local_2rate",
    "motor_2rate",
    "boat_2rate",
    "exhibit_time",
    "exhibit_time_zscore",
    "motor_2rate_rel",
    "national_win_rate_rel",
    "local_win_rate_rel",
    "is_course1",
    "is_course2",
    "is_inner",
    "a1_inner",
    "speed_score",
    "local_advantage",
]

TARGET_COL = "win"


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """特徴量行列Xとターゲットyを返す"""
    df_feat = build_features(df)
    available = [c for c in FEATURE_COLS if c in df_feat.columns]
    X = df_feat[available].copy()
    y = df_feat[TARGET_COL].copy()
    return X, y


def get_race_groups(df: pd.DataFrame) -> pd.Series:
    """時系列分割用のレースIDグループ"""
    df_feat = build_features(df)
    return df_feat["date"].astype(str) + "_" + df_feat["venue_code"] + "_" + df_feat["race_no"].astype(str)


if __name__ == "__main__":
    # サンプル動作確認（モックデータで）
    sample = [
        {
            "boat_no": i + 1,
            "player_id": f"400{i:04d}",
            "player_name": f"選手{i+1}",
            "class": ["A1", "A1", "A2", "B1", "B1", "B2"][i],
            "national_win_rate": [0.65, 0.52, 0.43, 0.38, 0.31, 0.22][i],
            "national_2rate": [0.72, 0.61, 0.55, 0.49, 0.40, 0.30][i],
            "national_3rate": [0.80, 0.70, 0.65, 0.60, 0.52, 0.42][i],
            "local_win_rate": [0.68, 0.50, 0.45, 0.35, 0.29, 0.20][i],
            "local_2rate": [0.75, 0.62, 0.56, 0.46, 0.38, 0.28][i],
            "motor_2rate": [0.45, 0.38, 0.42, 0.35, 0.30, 0.25][i],
            "boat_2rate": [0.42, 0.40, 0.38, 0.36, 0.32, 0.28][i],
            "exhibit_time": [6.72, 6.81, 6.78, 6.90, 6.88, 6.95][i],
            "venue_code": "01",
            "venue_name": "桐生",
            "date": "20260101",
            "race_no": 1,
            "rank": i + 1,
            "win": 1 if i == 0 else 0,
        }
        for i in range(6)
    ]
    df = pd.DataFrame(sample)
    X, y = prepare_dataset(df)
    print("特徴量一覧:")
    print(X.to_string())
    print(f"\nターゲット: {y.tolist()}")
    print(f"\n特徴量数: {X.shape[1]}")
