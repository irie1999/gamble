"""
競輪予測モデルモジュール
LightGBMで各選手の勝利確率を推定する
競艇と同様にTimeSeriesSplitで時系列評価を行う
"""

import json
import os
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, log_loss

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

from features import build_features, FEATURE_COLS, TARGET_COL

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 5,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_estimators": 500,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}


def load_data(filename: str = "raw_data.json") -> pd.DataFrame:
    path = DATA_DIR / filename
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return pd.DataFrame(records)


def train_evaluate(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """時系列CVで評価し、全データで最終モデルを学習"""
    df_feat = build_features(df)
    available = [c for c in FEATURE_COLS if c in df_feat.columns]
    X = df_feat[available].values
    y = df_feat[TARGET_COL].values
    dates = df_feat["date"].astype(str).values

    sort_idx = np.argsort(dates, kind="stable")
    X, y, dates = X[sort_idx], y[sort_idx], dates[sort_idx]

    unique_dates = np.unique(dates)
    n_dates = len(unique_dates)
    print(f"データ期間: {unique_dates[0]} → {unique_dates[-1]} ({n_dates}日間)")
    print(f"総レコード数: {len(X)}  (勝利数: {y.sum()})")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    date_indices = np.searchsorted(unique_dates, dates)

    metrics = {"auc": [], "logloss": []}
    models = []

    for fold, (train_di, val_di) in enumerate(tscv.split(unique_dates)):
        train_mask = np.isin(date_indices, train_di)
        val_mask = np.isin(date_indices, val_di)
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        if X_train.shape[0] < 100 or X_val.shape[0] < 10:
            continue

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, prob)
        ll = log_loss(y_val, prob)
        metrics["auc"].append(auc)
        metrics["logloss"].append(ll)
        models.append(model)
        print(f"  Fold {fold+1}: AUC={auc:.4f}  LogLoss={ll:.4f}")

    if not models:
        raise ValueError("学習データが不足しています")

    mean_auc = np.mean(metrics["auc"])
    mean_ll = np.mean(metrics["logloss"])

    random_logloss = -np.log(1 / 8)

    print(f"\n平均AUC: {mean_auc:.4f}  平均LogLoss: {mean_ll:.4f}")
    print(f"ランダム基準 AUC=0.500、LogLoss={random_logloss:.4f} (1/8人想定)")

    final_model = lgb.LGBMClassifier(**LGB_PARAMS)
    final_model.fit(X, y, callbacks=[lgb.log_evaluation(0)])

    feature_importance = dict(zip(available, final_model.feature_importances_))

    return {
        "model": final_model,
        "feature_cols": available,
        "metrics": {"cv_auc": mean_auc, "cv_logloss": mean_ll, "folds": metrics},
        "feature_importance": dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        ),
    }


def predict_race(
    model: lgb.LGBMClassifier,
    race_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """1レース分の勝利確率を推定"""
    df_feat = build_features(race_df)
    available = [c for c in feature_cols if c in df_feat.columns]
    X = df_feat[available].values

    probs = model.predict_proba(X)[:, 1]
    probs = probs / probs.sum()

    result = race_df[["car_no", "player_name", "line_no", "is_line_leader"]].copy().reset_index(drop=True)
    result["win_prob"] = probs
    result = result.sort_values("win_prob", ascending=False).reset_index(drop=True)
    return result


def save_model(result: dict) -> Path:
    model_path = MODEL_DIR / "lgb_model.txt"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # Write to system temp dir first to avoid OneDrive/cloud sync locks
    fd, tmp_name = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        result["model"].booster_.save_model(tmp_name)
        if model_path.exists():
            model_path.unlink()
        shutil.copy2(tmp_name, str(model_path))
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)

    meta = {
        "feature_cols": result["feature_cols"],
        "metrics": result["metrics"],
        "feature_importance": {k: int(v) for k, v in result["feature_importance"].items()},
    }
    meta_path = MODEL_DIR / "model_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nモデル保存: {model_path}")
    print(f"メタ情報: {meta_path}")
    return model_path


def load_model() -> tuple:
    model_path = MODEL_DIR / "lgb_model.txt"
    meta_path = MODEL_DIR / "model_meta.json"
    # Copy to system temp first to avoid OneDrive read locks
    fd, tmp_name = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        shutil.copy2(str(model_path), tmp_name)
        booster = lgb.Booster(model_file=tmp_name)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return booster, meta["feature_cols"], meta


def print_feature_importance(result: dict, top_n: int = 10) -> None:
    print(f"\n=== 特徴量重要度 Top{top_n} ===")
    max_imp = max(result["feature_importance"].values()) or 1
    for i, (feat, imp) in enumerate(list(result["feature_importance"].items())[:top_n]):
        bar = "█" * int(imp / max_imp * 20)
        print(f"  {i+1:2d}. {feat:<28s} {imp:5.0f}  {bar}")


def _generate_line_config(n_riders: int) -> list[tuple[int, int]]:
    """ランダムなライン構成を生成"""
    sizes = []
    remaining = n_riders
    line_no = 1
    while remaining > 0:
        size = min(np.random.choice([1, 2, 3], p=[0.2, 0.3, 0.5]), remaining)
        sizes.append((line_no, size))
        remaining -= size
        line_no += 1

    config = []
    car_no = 1
    for ln, size in sizes:
        for _ in range(size):
            config.append((ln if size > 1 else 0, car_no))
            car_no += 1
    return config


if __name__ == "__main__":
    np.random.seed(42)

    # 競輪モックデータ生成（ライン構成あり）
    records = []
    n_races = 300

    CLASS_MAP_RATE = {"S1": 0.28, "S2": 0.22, "A1": 0.17, "A2": 0.13, "A3": 0.10, "B1": 0.07}

    for race_id in range(n_races):
        day_offset = race_id // 10
        date = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=day_offset)).strftime("%Y%m%d")
        rno = race_id % 10 + 1
        bank = np.random.choice([333, 400, 500])

        # ライン構成を生成（3+2+2 or 3+3+1 など）
        n_riders = np.random.choice([7, 8, 9])
        line_configs = _generate_line_config(n_riders)
        car_no = 1
        # ライン先頭の強さを決める（先頭選手の勝率が高い）
        winner_line = np.random.choice(list(set(ln for ln, _ in line_configs)), p=None)
        winner_car = min(c for ln, c in line_configs if ln == winner_line)

        for line_no, cn in line_configs:
            is_leader = 1 if cn == min(c for l, c in line_configs if l == line_no) and line_no > 0 else 0
            line_size = sum(1 for l, _ in line_configs if l == line_no) if line_no > 0 else 1
            cls = np.random.choice(["S1", "S2", "A1", "A2", "A3", "B1"],
                                   p=[0.05, 0.10, 0.30, 0.30, 0.15, 0.10])
            wr = max(0.05, CLASS_MAP_RATE[cls] + np.random.normal(0, 0.04))
            records.append({
                "car_no": cn,
                "line_no": line_no,
                "line_size": line_size,
                "is_line_leader": is_leader,
                "player_name": f"選手{cn}",
                "class": cls,
                "win_rate": wr,
                "second_rate": max(0.05, wr * 0.85 + np.random.normal(0, 0.03)),
                "third_rate": max(0.05, wr * 0.75 + np.random.normal(0, 0.03)),
                "bank_length": bank,
                "venue_code": "15",
                "venue_name": "前橋",
                "date": date,
                "race_no": rno,
                "rank": cn,
                "win": 1 if cn == winner_car else 0,
            })

    df = pd.DataFrame(records)
    print("=== モデル学習・評価 ===")
    result = train_evaluate(df, n_splits=3)
    print_feature_importance(result)

    print("\n=== 1レース予測サンプル ===")
    sample_race = df[df["race_no"] == 1].head(8).copy()
    pred = predict_race(result["model"], sample_race, result["feature_cols"])
    print(pred.to_string(index=False))
