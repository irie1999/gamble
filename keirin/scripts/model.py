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

# デフォルトパラメータ（Optunaで上書き可能）
LGB_PARAMS_DEFAULT = {
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

TUNED_PARAMS_PATH = MODEL_DIR / "lgb_params.json"


def _load_params() -> dict:
    """Optunaでチューニング済みパラメータがあれば読み込む"""
    if TUNED_PARAMS_PATH.exists():
        with open(TUNED_PARAMS_PATH, encoding="utf-8") as f:
            tuned = json.load(f)
        params = {**LGB_PARAMS_DEFAULT, **tuned}
        print(f"  チューニング済みパラメータを使用: {TUNED_PARAMS_PATH.name}")
        return params
    return LGB_PARAMS_DEFAULT.copy()


LGB_PARAMS = LGB_PARAMS_DEFAULT  # 後方互換


def load_data(filename: str = "raw_data.json") -> pd.DataFrame:
    path = DATA_DIR / filename
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return pd.DataFrame(records)


def _prepare_train_data(df: pd.DataFrame, soft_labels: bool = False):
    """特徴量行列・ターゲット・日付を返す共通処理"""
    df_feat = build_features(df)
    available = [c for c in FEATURE_COLS if c in df_feat.columns]
    X = df_feat[available].values
    target_col = "soft_label" if soft_labels and "soft_label" in df_feat.columns else TARGET_COL
    y = df_feat[target_col].values
    dates = df_feat["date"].astype(str).values
    sort_idx = np.argsort(dates, kind="stable")
    return X[sort_idx], y[sort_idx], dates[sort_idx], available


def tune_hyperparams(df: pd.DataFrame, n_trials: int = 50, n_splits: int = 3) -> dict:
    """Optunaでハイパーパラメータを最適化し、lgb_params.jsonに保存"""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("Optunaが未インストールです: pip install optuna")
        return {}

    X, y, dates, _ = _prepare_train_data(df)
    unique_dates = np.unique(dates)
    date_indices = np.searchsorted(unique_dates, dates)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        lgb_p = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "num_threads": -1,
            "seed": 42,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        num_round = 1000
        aucs = []
        for train_di, val_di in tscv.split(unique_dates):
            train_mask = np.isin(date_indices, train_di)
            val_mask = np.isin(date_indices, val_di)
            X_tr, y_tr = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            if X_tr.shape[0] < 100 or X_val.shape[0] < 10:
                continue
            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            b = lgb.train(lgb_p, dtrain, num_boost_round=num_round, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
            aucs.append(roc_auc_score(y_val, b.predict(X_val)))
        return np.mean(aucs) if aucs else 0.0

    print(f"Optuna最適化開始: {n_trials}試行 / {n_splits}fold CV")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["n_estimators"] = 1000
    print(f"\n最良AUC: {study.best_value:.4f}")
    print(f"最良パラメータ: {best}")

    TUNED_PARAMS_PATH.parent.mkdir(exist_ok=True)
    with open(TUNED_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    print(f"保存: {TUNED_PARAMS_PATH}")
    return best


def train_evaluate(df: pd.DataFrame, n_splits: int = 5, soft_labels: bool = True) -> dict:
    """時系列CVで評価し、全データで最終モデルを学習

    soft_labels=True: 1/rank正規化値を目標に学習（2位・3位にも訓練シグナル）
    """
    params = _load_params()

    # ソフトラベルか通常ラベルかでターゲットを選択
    df_feat = build_features(df)
    available = [c for c in FEATURE_COLS if c in df_feat.columns]
    X = df_feat[available].values
    target_col = "soft_label" if soft_labels and "soft_label" in df_feat.columns else TARGET_COL
    y_soft = df_feat[target_col].values           # 学習用（ソフトまたはバイナリ）
    y_hard = df_feat[TARGET_COL].values           # AUC評価用（常にバイナリ）
    dates = df_feat["date"].astype(str).values

    sort_idx = np.argsort(dates, kind="stable")
    X, y_soft, y_hard, dates = X[sort_idx], y_soft[sort_idx], y_hard[sort_idx], dates[sort_idx]

    if soft_labels and target_col == "soft_label":
        print(f"  ソフトラベル学習（1/rank正規化）: {target_col}")

    unique_dates = np.unique(dates)
    n_dates = len(unique_dates)
    print(f"データ期間: {unique_dates[0]} → {unique_dates[-1]} ({n_dates}日間)")
    print(f"総レコード数: {len(X)}  (勝利数: {y_hard.sum()})")

    n_splits = min(n_splits, n_dates - 1)
    if n_splits < 2:
        print(f"警告: データが少なすぎます（{n_dates}日）。最低7日分必要です。")
        return None
    tscv = TimeSeriesSplit(n_splits=n_splits)
    date_indices = np.searchsorted(unique_dates, dates)

    # lgb.train()用パラメータに変換（sklearn固有キーを除外）
    sklearn_only = {"n_estimators", "random_state", "verbose", "n_jobs"}
    lgb_params = {k: v for k, v in params.items() if k not in sklearn_only}
    lgb_params.setdefault("seed", params.get("random_state", 42))
    lgb_params.setdefault("num_threads", -1)
    lgb_params["verbosity"] = -1
    num_boost_round = params.get("n_estimators", 500)

    metrics = {"auc": [], "logloss": []}
    oof_probs = np.zeros(len(X))   # キャリブレーション用out-of-fold予測

    for fold, (train_di, val_di) in enumerate(tscv.split(unique_dates)):
        train_mask = np.isin(date_indices, train_di)
        val_mask = np.isin(date_indices, val_di)
        X_train, y_train = X[train_mask], y_soft[train_mask]
        y_soft_val = y_soft[val_mask]
        X_val = X[val_mask]
        y_val_hard = y_hard[val_mask]

        if X_train.shape[0] < 100 or X_val.shape[0] < 10:
            continue

        dtrain = lgb.Dataset(X_train, label=y_train)
        # Early stoppingはソフトラベルで評価（学習目標と一致させる）
        dval = lgb.Dataset(X_val, label=y_soft_val, reference=dtrain)
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        booster = lgb.train(lgb_params, dtrain, num_boost_round=num_boost_round,
                            valid_sets=[dval], callbacks=callbacks)

        prob = booster.predict(X_val)
        prob = np.clip(prob, 1e-7, 1 - 1e-7)
        oof_probs[val_mask] = prob
        auc = roc_auc_score(y_val_hard, prob)
        ll = log_loss(y_val_hard, prob)
        metrics["auc"].append(auc)
        metrics["logloss"].append(ll)
        print(f"  Fold {fold+1}: AUC={auc:.4f}  LogLoss={ll:.4f}")

    if not metrics["auc"]:
        raise ValueError("学習データが不足しています")

    mean_auc = np.mean(metrics["auc"])
    mean_ll = np.mean(metrics["logloss"])

    random_logloss = -np.log(1 / 8)
    print(f"\n平均AUC: {mean_auc:.4f}  平均LogLoss: {mean_ll:.4f}")
    print(f"ランダム基準 AUC=0.500、LogLoss={random_logloss:.4f} (1/8人想定)")

    # 全データで最終モデルを学習（ネイティブAPIでソフトラベル対応）
    dtrain_full = lgb.Dataset(X, label=y_soft)
    final_model = lgb.train(lgb_params, dtrain_full, num_boost_round=num_boost_round,
                            callbacks=[lgb.log_evaluation(0)])

    # 確率キャリブレーション（isotonic regression）
    calibrator = None
    oof_mask = oof_probs > 0
    if oof_mask.sum() > 100:
        from sklearn.isotonic import IsotonicRegression
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(oof_probs[oof_mask], y_hard[oof_mask])
        cal_probs = calibrator.predict(oof_probs[oof_mask])
        cal_auc = roc_auc_score(y_hard[oof_mask], cal_probs)
        print(f"キャリブレーション後AUC: {cal_auc:.4f}")

    feature_importance = dict(zip(available, final_model.feature_importance(importance_type="gain")))

    return {
        "model": final_model,
        "calibrator": calibrator,
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

    probs = model.predict(X)
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
        result["model"].save_model(tmp_name)
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

    # キャリブレーターを保存
    cal = result.get("calibrator")
    cal_path = MODEL_DIR / "calibrator.pkl"
    if cal is not None:
        import pickle
        with open(cal_path, "wb") as f:
            pickle.dump(cal, f)
        print(f"キャリブレーター保存: {cal_path}")
    elif cal_path.exists():
        cal_path.unlink()

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

    # キャリブレーターがあれば読み込む
    cal_path = MODEL_DIR / "calibrator.pkl"
    if cal_path.exists():
        import pickle
        with open(cal_path, "rb") as f:
            meta["calibrator"] = pickle.load(f)

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
