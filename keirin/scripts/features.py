"""
競輪特徴量エンジニアリングモジュール
競輪固有の「ライン戦略」「バンク特性」を含む特徴量を生成する
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# 級別エンコード（高いほど強い）
CLASS_MAP = {"S1": 6, "S2": 5, "A1": 4, "A2": 3, "A3": 2, "B1": 1, "": 0}

# バンク周長別インコース勝率補正
# 333m バンクは小回りでインが有利、500m は差し・捲りが届きやすい
BANK_INNER_BONUS = {333: 0.08, 400: 0.0, 500: -0.05}


def load_raw(filename: str = "raw_data.json") -> pd.DataFrame:
    path = DATA_DIR / filename
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return pd.DataFrame(records)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- 基本エンコード ---
    df["class_enc"] = df["class"].map(CLASS_MAP).fillna(0).astype(int)

    # --- 成績系（欠損補完） ---
    for col in ["win_rate", "second_rate", "third_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # 3着内率（勝率+2着率+3着率の代理）
    df["top3_rate"] = df["win_rate"] + df["second_rate"].fillna(0) + df["third_rate"].fillna(0)

    # --- ライン特徴量（競輪固有）---
    df["line_no"] = pd.to_numeric(df["line_no"], errors="coerce").fillna(0).astype(int)
    df["line_size"] = pd.to_numeric(df["line_size"], errors="coerce").fillna(1).astype(int)
    df["is_line_leader"] = pd.to_numeric(df["is_line_leader"], errors="coerce").fillna(0).astype(int)

    # ラインの平均勝率（ライン全体の強さ）
    race_key = ["date", "venue_code", "race_no"]
    line_key = race_key + ["line_no"]
    line_mean_wr = df[df["line_no"] > 0].groupby(line_key)["win_rate"].transform("mean")
    df["line_avg_win_rate"] = line_mean_wr.fillna(df["win_rate"])

    # ライン内での自分の順位（先頭=1が最も有利）
    df["line_rank_in_line"] = df.groupby(line_key)["car_no"].rank(method="min").astype(int)
    df.loc[df["line_no"] == 0, "line_rank_in_line"] = 1  # 単騎は1

    # 大きいライン（3人以上）に属しているか
    df["in_big_line"] = (df["line_size"] >= 3).astype(int)

    # ライン先頭 × ラインサイズの複合（先頭で大ラインが最強）
    df["leader_line_size"] = df["is_line_leader"] * df["line_size"]

    # --- バンク特性 ---
    df["bank_length"] = pd.to_numeric(df["bank_length"], errors="coerce").fillna(400).astype(int)
    df["bank_inner_bonus"] = df["bank_length"].map(BANK_INNER_BONUS).fillna(0.0)

    # 車番（内枠有利補正）
    df["car_no"] = pd.to_numeric(df["car_no"], errors="coerce").fillna(1).astype(int)
    df["is_inner_car"] = (df["car_no"] <= 3).astype(int)
    # バンク×内枠の複合
    df["bank_inner_effect"] = df["bank_inner_bonus"] * (4 - df["car_no"].clip(upper=4)) / 3

    # --- レース内相対値 ---
    for col in ["win_rate", "class_enc", "top3_rate"]:
        race_mean = df.groupby(race_key)[col].transform("mean")
        race_std = df.groupby(race_key)[col].transform("std").replace(0, np.nan)
        df[f"{col}_rel"] = (df[col] - race_mean) / race_std.fillna(1)

    # --- 単騎フラグ（ライン外の孤立選手）---
    df["is_solo"] = (df["line_no"] == 0).astype(int)

    # --- 強豪ライン先頭（S1がラインリーダー）---
    df["s1_leader"] = ((df["class_enc"] >= 5) & (df["is_line_leader"] == 1)).astype(int)

    # --- ターゲット ---
    df["win"] = pd.to_numeric(df["win"], errors="coerce").fillna(0).astype(int)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

    return df


FEATURE_COLS = [
    "car_no",
    "class_enc",
    "win_rate",
    "second_rate",
    "third_rate",
    "top3_rate",
    "line_no",
    "line_size",
    "is_line_leader",
    "line_avg_win_rate",
    "line_rank_in_line",
    "in_big_line",
    "leader_line_size",
    "bank_length",
    "bank_inner_bonus",
    "is_inner_car",
    "bank_inner_effect",
    "win_rate_rel",
    "class_enc_rel",
    "top3_rate_rel",
    "is_solo",
    "s1_leader",
]

TARGET_COL = "win"


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df_feat = build_features(df)
    available = [c for c in FEATURE_COLS if c in df_feat.columns]
    X = df_feat[available].copy()
    y = df_feat[TARGET_COL].copy()
    return X, y


if __name__ == "__main__":
    # モックデータで動作確認
    sample = []
    lines = [(1, [1, 2, 3]), (2, [4, 5]), (0, [6, 7])]  # ライン構成
    car_no = 1
    for line_no, members in lines:
        for i, _ in enumerate(members):
            is_leader = 1 if i == 0 and line_no > 0 else 0
            sample.append({
                "car_no": car_no,
                "line_no": line_no,
                "line_size": len(members) if line_no > 0 else 1,
                "is_line_leader": is_leader,
                "player_name": f"選手{car_no}",
                "class": ["S1", "A1", "A2", "S1", "A1", "A2", "B1"][car_no - 1],
                "win_rate": [0.32, 0.25, 0.18, 0.28, 0.20, 0.15, 0.10][car_no - 1],
                "second_rate": [0.25, 0.22, 0.20, 0.22, 0.18, 0.14, 0.09][car_no - 1],
                "third_rate": [0.20, 0.18, 0.17, 0.19, 0.16, 0.12, 0.08][car_no - 1],
                "bank_length": 333,
                "venue_code": "15",
                "venue_name": "前橋",
                "date": "20260101",
                "race_no": 1,
                "rank": car_no,
                "win": 1 if car_no == 1 else 0,
            })
            car_no += 1

    df = pd.DataFrame(sample)
    X, y = prepare_dataset(df)
    print("特徴量一覧:")
    print(X.to_string())
    print(f"\nターゲット: {y.tolist()}")
    print(f"\n特徴量数: {X.shape[1]}")
