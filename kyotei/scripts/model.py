"""
予測モデルモジュール
LightGBMで各艇の勝利確率を推定する
時系列を崩さないようにTimeSeriesSplitで評価する
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV

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
    """
    時系列CVで評価し、最終モデルを返す
    各foldでは古いデータで学習→新しいデータで評価
    """
    df_feat = build_features(df)
    available = [c for c in FEATURE_COLS if c in df_feat.columns]
    X = df_feat[available].values
    y = df_feat[TARGET_COL].values
    dates = df_feat["date"].astype(str).values

    # 日付でソート
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
        raise ValueError("学習データが不足しています（最低2週間分以上必要）")

    mean_auc = np.mean(metrics["auc"])
    mean_ll = np.mean(metrics["logloss"])
    print(f"\n平均AUC: {mean_auc:.4f}  平均LogLoss: {mean_ll:.4f}")
    print(f"ランダム基準 AUC=0.500、LogLoss={-np.log(1/6):.4f} (1/6)")

    # 全データで最終モデルを学習
    final_model = lgb.LGBMClassifier(**LGB_PARAMS)
    final_model.fit(X, y, callbacks=[lgb.log_evaluation(0)])

    feature_importance = dict(zip(available, final_model.feature_importances_))

    return {
        "model": final_model,
        "feature_cols": available,
        "metrics": {
            "cv_auc": mean_auc,
            "cv_logloss": mean_ll,
            "folds": metrics,
        },
        "feature_importance": dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        ),
    }


def predict_race(model: lgb.LGBMClassifier, race_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    1レース分のDFを受け取り、各艇の勝利確率を付与して返す
    """
    df_feat = build_features(race_df)
    available = [c for c in feature_cols if c in df_feat.columns]
    X = df_feat[available].values

    probs = model.predict_proba(X)[:, 1]
    # レース内で確率を正規化（合計1になるように）
    probs = probs / probs.sum()

    result = race_df[["boat_no", "player_name"]].copy().reset_index(drop=True)
    result["win_prob"] = probs
    result = result.sort_values("win_prob", ascending=False).reset_index(drop=True)
    return result


def save_model(result: dict) -> Path:
    """モデルと設定を保存"""
    model_path = MODEL_DIR / "lgb_model.txt"
    result["model"].booster_.save_model(str(model_path))

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
    """保存済みモデルをロード"""
    model_path = MODEL_DIR / "lgb_model.txt"
    meta_path = MODEL_DIR / "model_meta.json"

    booster = lgb.Booster(model_file=str(model_path))

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    return booster, meta["feature_cols"], meta


def print_feature_importance(result: dict, top_n: int = 10) -> None:
    print(f"\n=== 特徴量重要度 Top{top_n} ===")
    for i, (feat, imp) in enumerate(list(result["feature_importance"].items())[:top_n]):
        bar = "█" * int(imp / max(result["feature_importance"].values()) * 20)
        print(f"  {i+1:2d}. {feat:<28s} {imp:5.0f}  {bar}")


if __name__ == "__main__":
    # モックデータでの動作確認
    np.random.seed(42)
    COURSE_WIN_RATE_BASELINE_MOCK = {1: 0.553, 2: 0.178, 3: 0.118, 4: 0.082, 5: 0.044, 6: 0.025}
    n_races = 200
    records = []
    for race_id in range(n_races):
        date_offset = race_id // 12
        date = f"2026{(date_offset // 30 + 1):02d}{(date_offset % 30 + 1):02d}"
        for boat_no in range(1, 7):
            # インコース有利を反映したモック勝率
            true_win_prob = COURSE_WIN_RATE_BASELINE_MOCK[boat_no]
            win = 1 if np.random.random() < true_win_prob * 6 / 6 else 0
            records.append({
                "boat_no": boat_no,
                "player_name": f"選手{boat_no}",
                "class": ["A1", "A1", "A2", "B1", "B1", "B2"][boat_no - 1],
                "national_win_rate": max(0.1, 0.55 - boat_no * 0.07 + np.random.normal(0, 0.05)),
                "national_2rate": max(0.2, 0.65 - boat_no * 0.07 + np.random.normal(0, 0.05)),
                "national_3rate": max(0.3, 0.75 - boat_no * 0.05 + np.random.normal(0, 0.05)),
                "local_win_rate": max(0.1, 0.50 - boat_no * 0.07 + np.random.normal(0, 0.05)),
                "local_2rate": max(0.2, 0.60 - boat_no * 0.07 + np.random.normal(0, 0.05)),
                "motor_2rate": max(0.1, 0.40 - boat_no * 0.03 + np.random.normal(0, 0.05)),
                "boat_2rate": max(0.1, 0.38 + np.random.normal(0, 0.04)),
                "exhibit_time": 6.75 + (boat_no - 1) * 0.05 + np.random.normal(0, 0.03),
                "venue_code": "01",
                "venue_name": "桐生",
                "date": date,
                "race_no": race_id % 12 + 1,
                "rank": boat_no,
                "win": win,
            })

    df = pd.DataFrame(records)
    print("=== モデル学習・評価 ===")
    result = train_evaluate(df, n_splits=3)
    print_feature_importance(result)

    print("\n=== 1レース予測サンプル ===")
    sample_race = df[df["race_no"] == 1].head(6).copy()
    pred = predict_race(result["model"], sample_race, result["feature_cols"])
    print(pred.to_string(index=False))
