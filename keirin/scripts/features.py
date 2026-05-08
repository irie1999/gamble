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


def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """選手ごとの時系列特徴量を追加（データリーク防止：過去レースのみ使用）"""
    df = df.copy()
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["win"] = pd.to_numeric(df["win"], errors="coerce").fillna(0).astype(int)
    df["date"] = df["date"].astype(str)

    # 日付順にソート（同日内はrace_noで安定化）
    df = df.sort_values(["date", "venue_code", "race_no", "car_no"]).reset_index(drop=True)

    # --- 選手単位の時系列特徴量 ---
    # shift(1)で当該レース自身のデータを除外し、過去データのみ参照
    grp = df.groupby("player_name", sort=False)

    # 直近3走・5走の勝率
    df["recent_win_3"] = grp["win"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df["recent_win_5"] = grp["win"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # 直近5走の3着内率
    top3_flag = (df["rank"] <= 3).astype(float)
    df["_top3_flag"] = top3_flag
    df["recent_top3_5"] = grp["_top3_flag"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df.drop(columns=["_top3_flag"], inplace=True)

    # 直近5走の平均着順（小さいほど良い）
    df["recent_avg_rank"] = grp["rank"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # 調子トレンド（直近勝率 - 通算勝率）：プラスなら上り調子
    df["form_trend"] = df["recent_win_5"] - df["win_rate"].fillna(0)

    # 直近5走の競走得点トレンド（kyosotenが上昇中かどうか）
    if "kyosoten" in df.columns:
        df["_kys"] = pd.to_numeric(df["kyosoten"], errors="coerce").fillna(0)
        df["recent_kyosoten_5"] = grp["_kys"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
        df["kyosoten_trend"] = df["recent_kyosoten_5"] - df["_kys"]
        df.drop(columns=["_kys"], inplace=True)
    else:
        df["recent_kyosoten_5"] = 0.0
        df["kyosoten_trend"] = 0.0

    # --- 会場別勝率（その会場での過去実績）---
    grp_venue = df.groupby(["player_name", "venue_code"], sort=False)
    df["venue_win_rate"] = grp_venue["win"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    # データ不足時は通算勝率で補完
    df["venue_win_rate"] = df["venue_win_rate"].fillna(df["win_rate"].fillna(0))

    # --- 前走からの休養日数 ---
    df["_date_dt"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["_last_date"] = grp["_date_dt"].transform(lambda x: x.shift(1))
    df["days_since_last"] = (df["_date_dt"] - df["_last_date"]).dt.days.fillna(14).clip(1, 60)
    df.drop(columns=["_date_dt", "_last_date"], inplace=True)

    # 欠損補完
    for col in ["recent_win_3", "recent_win_5", "recent_top3_5"]:
        df[col] = df[col].fillna(df["win_rate"].fillna(0))
    df["recent_avg_rank"] = df["recent_avg_rank"].fillna(4.0)
    df["form_trend"] = df["form_trend"].fillna(0.0)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- 基本エンコード ---
    df["class_enc"] = df["class"].map(CLASS_MAP).fillna(0).astype(int)

    # --- 成績系（欠損補完） ---
    for col in ["win_rate", "second_rate", "third_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    race_key = ["date", "venue_code", "race_no"]

    # --- 競走得点（kyosoten） ---
    if "kyosoten" in df.columns:
        df["kyosoten"] = pd.to_numeric(df["kyosoten"], errors="coerce")
        df["kyosoten"] = df["kyosoten"].fillna(df["kyosoten"].median())
        race_mean_ks = df.groupby(race_key)["kyosoten"].transform("mean")
        race_std_ks = df.groupby(race_key)["kyosoten"].transform("std").replace(0, np.nan)
        df["kyosoten_rel"] = (df["kyosoten"] - race_mean_ks) / race_std_ks.fillna(1)
    else:
        df["kyosoten"] = 0.0
        df["kyosoten_rel"] = 0.0

    # 3着内率（勝率+2着率+3着率の代理）
    df["top3_rate"] = df["win_rate"] + df["second_rate"].fillna(0) + df["third_rate"].fillna(0)

    # --- ライン特徴量（競輪固有）---
    df["line_no"] = pd.to_numeric(df["line_no"], errors="coerce").fillna(0).astype(int)
    df["line_size"] = pd.to_numeric(df["line_size"], errors="coerce").fillna(1).astype(int)
    df["is_line_leader"] = pd.to_numeric(df["is_line_leader"], errors="coerce").fillna(0).astype(int)

    # ラインの平均勝率（ライン全体の強さ）
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

    # --- 競争難易度：レース内の実力分散 ---
    # kyosoten の標準偏差が小さい = 拮抗レース（予測困難）
    # 標準偏差が大きい = 実力差明確（予測しやすい）
    if "kyosoten" in df.columns:
        race_kyosoten_std = df.groupby(race_key)["kyosoten"].transform("std").fillna(0)
        race_kyosoten_max = df.groupby(race_key)["kyosoten"].transform("max")
        df["race_competitiveness"] = race_kyosoten_std  # 大きいほど実力差あり
        # レース内での自分の競走得点優位性（最強との差）
        df["kyosoten_vs_top"] = df["kyosoten"] - race_kyosoten_max
    else:
        df["race_competitiveness"] = 0.0
        df["kyosoten_vs_top"] = 0.0

    # --- 時系列特徴量（直近成績・会場別・調子）---
    df = add_historical_features(df)

    # --- 時系列特徴量のレース内相対値 ---
    for col in ["recent_win_5", "venue_win_rate", "recent_avg_rank"]:
        if col in df.columns:
            race_mean = df.groupby(race_key)[col].transform("mean")
            race_std = df.groupby(race_key)[col].transform("std").replace(0, np.nan)
            df[f"{col}_rel"] = (df[col] - race_mean) / race_std.fillna(1)

    # --- 特徴量交互作用 ---
    # ラインリーダー × 直近調子（強いリーダーが好調なら最強）
    df["leader_recent_form"] = df["is_line_leader"] * df.get("recent_win_5", 0)
    # 競走得点相対優位 × レース難易度（強い選手が明確に有利なレース）
    df["kyosoten_edge"] = df["kyosoten_rel"] * df["race_competitiveness"].clip(lower=0)
    # 会場巧者 × ライン先頭（地力+本拠地効果）
    df["venue_leader"] = df["is_line_leader"] * df.get("venue_win_rate", 0)

    # --- ソフトラベル（学習用：着順の逆数を正規化）---
    soft = 1.0 / df["rank"].clip(lower=1)
    soft_sum = df.groupby(race_key)["rank"].transform(
        lambda x: (1.0 / x.clip(lower=1)).sum()
    )
    df["soft_label"] = (soft / soft_sum.replace(0, 1)).fillna(0)

    # --- LambdaRank用関連度ラベル（整数）---
    # 1位→3, 2位→2, 3位→1, 4位以下→0
    df["lambdarank_label"] = (4 - df["rank"]).clip(lower=0).fillna(0).astype(int)

    return df


FEATURE_COLS = [
    # car_no は除外（内枠バイアスが強すぎて常に1-2-3予測になるため）
    # is_inner_car / line_rank_in_line で位置優位性を残す
    "class_enc",
    "kyosoten",
    "kyosoten_rel",
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
    # --- 時系列特徴量 ---
    "recent_win_3",
    "recent_win_5",
    "recent_top3_5",
    "recent_avg_rank",
    "form_trend",
    "venue_win_rate",
    "days_since_last",
    "recent_win_5_rel",
    "venue_win_rate_rel",
    "recent_avg_rank_rel",
    # --- 競走得点トレンド・競争難易度 ---
    "recent_kyosoten_5",
    "kyosoten_trend",
    "race_competitiveness",
    "kyosoten_vs_top",
    # --- 特徴量交互作用 ---
    "leader_recent_form",
    "kyosoten_edge",
    "venue_leader",
]

TARGET_COL = "win"


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df_feat = build_features(df)
    available = [c for c in FEATURE_COLS if c in df_feat.columns]
    X = df_feat[available].copy()
    y = df_feat[TARGET_COL].copy()
    return X, y
