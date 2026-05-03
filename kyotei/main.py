"""
競艇予測システム メインスクリプト

使い方:
  python main.py collect   -- 過去データ収集（実際にスクレイピング）
  python main.py train     -- 収集済みデータでモデル学習
  python main.py backtest  -- バックテスト（モックデータで動作確認可能）
  python main.py demo      -- モックデータで全工程をデモ実行
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# スクリプトディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from scraper import collect_data, save_records, VENUE_CODES
from features import build_features, FEATURE_COLS, prepare_dataset
from model import train_evaluate, predict_race, save_model, load_model, print_feature_importance
from betting import (
    pick_best_win_bet, simulate_session, print_session_report, DEDUCTION_RATE
)

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def cmd_collect(args):
    """実際のサイトからデータを収集する"""
    today = datetime.now()
    end_date = today.strftime("%Y%m%d")
    start_date = (today - timedelta(days=args.days)).strftime("%Y%m%d")

    venue_codes = args.venues.split(",") if args.venues else None
    print(f"収集期間: {start_date} → {end_date}")
    print(f"対象場: {venue_codes or '全24場'}")

    records = collect_data(start_date, end_date, venue_codes=venue_codes, sleep_sec=1.5)
    if records:
        save_records(records, "raw_data.json")
        print(f"\n収集完了: {len(records)}件")
    else:
        print("データが取得できませんでした")


def cmd_train(args):
    """収集済みデータでモデルを学習する"""
    raw_path = DATA_DIR / "raw_data.json"
    if not raw_path.exists():
        print(f"データファイルが見つかりません: {raw_path}")
        print("まず `python main.py collect` を実行してください")
        return

    with open(raw_path, encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    print(f"学習データ: {len(df)}件")

    result = train_evaluate(df, n_splits=5)
    print_feature_importance(result)
    save_model(result)


def cmd_backtest(args):
    """
    保存済みデータ or モックデータでバックテストを実行
    モデルが保存済みなら実モデルを使用、なければモックで動作確認
    """
    raw_path = DATA_DIR / "raw_data.json"
    model_path = MODEL_DIR / "lgb_model.txt"

    if raw_path.exists() and model_path.exists():
        print("実モデルでバックテスト実行...")
        _backtest_real(args)
    else:
        print("モデル or データが未作成のためモックデータでバックテスト実行...")
        _backtest_mock(args)


def _backtest_real(args):
    import lightgbm as lgb
    booster, feature_cols, meta = load_model()

    with open(DATA_DIR / "raw_data.json", encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    df_feat = build_features(df)

    # 最後の20%をテスト期間とする
    dates = sorted(df_feat["date"].unique())
    split_idx = int(len(dates) * 0.8)
    test_dates = dates[split_idx:]
    df_test = df_feat[df_feat["date"].isin(test_dates)]

    races = []
    for (date, venue, rno), race_df in df_test.groupby(["date", "venue_code", "race_no"]):
        winner_row = race_df[race_df["win"] == 1]
        if winner_row.empty:
            continue
        winner = int(winner_row["boat_no"].values[0])

        X = race_df[[c for c in feature_cols if c in race_df.columns]].values
        probs = booster.predict(X)
        probs = probs / probs.sum()

        pred_df = pd.DataFrame({
            "boat_no": race_df["boat_no"].values,
            "player_name": race_df["player_name"].values,
            "win_prob": probs,
        })

        # テスト用ダミーオッズ（実際はスクレイピングで取得）
        true_p = np.array([0.553, 0.178, 0.118, 0.082, 0.044, 0.025])
        dummy_odds = {i + 1: round(1.0 / true_p[i] * (1 - DEDUCTION_RATE), 1) for i in range(6)}

        races.append({
            "race_id": f"{date}_{venue}_{rno}",
            "pred_df": pred_df,
            "odds": dummy_odds,
            "winner": winner,
        })

    session = simulate_session(races, initial_bankroll=args.bankroll, min_edge=args.min_edge)
    print_session_report(session)
    _save_results(session)


def _backtest_mock(args):
    """モックデータでエンドツーエンド動作確認"""
    np.random.seed(42)
    COURSE_PROBS = {1: 0.553, 2: 0.178, 3: 0.118, 4: 0.082, 5: 0.044, 6: 0.025}

    races = []
    for i in range(300):
        boat_nos = list(range(1, 7))
        true_p = np.array([COURSE_PROBS[b] for b in boat_nos])

        # モデルの予測確率（実際の確率に少しノイズを加えた値）
        pred_p = true_p + np.random.normal(0, 0.03, 6)
        pred_p = np.clip(pred_p, 0.005, 1)
        pred_p /= pred_p.sum()

        pred_df = pd.DataFrame({
            "boat_no": boat_nos,
            "player_name": [f"選手{b}" for b in boat_nos],
            "win_prob": pred_p,
        })

        # 市場オッズ（控除率25%込み）
        fair_odds = {b: round(1.0 / true_p[b - 1] * (1 - DEDUCTION_RATE), 1) for b in boat_nos}

        winner = np.random.choice(boat_nos, p=true_p)
        races.append({
            "race_id": f"mock_{i+1:03d}",
            "pred_df": pred_df,
            "odds": fair_odds,
            "winner": winner,
        })

    session = simulate_session(
        races,
        initial_bankroll=args.bankroll,
        min_edge=args.min_edge,
        kelly_frac=0.25,
    )
    print_session_report(session)
    _save_results(session, "mock_backtest.json")


def _save_results(session, filename: str = "backtest_result.json") -> None:
    result = {
        "initial_bankroll": session.initial_bankroll,
        "final_bankroll": session.final_bankroll,
        "profit": session.profit,
        "roi": session.roi,
        "total_bet": session.total_bet,
        "wins": session.wins,
        "losses": session.losses,
        "bets": [
            {
                "race_id": b.race_id,
                "bet_type": b.bet_type,
                "selections": list(b.selections),
                "predicted_prob": b.predicted_prob,
                "implied_prob": b.implied_prob,
                "edge": b.edge,
                "odds": b.odds,
                "bet_amount": b.bet_amount,
                "expected_value": b.expected_value,
            }
            for b in session.bets
        ],
    }
    path = RESULTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n結果を保存しました: {path}")


def cmd_demo(args):
    """モックデータで全工程をデモ実行（ネット不要）"""
    print("=" * 60)
    print(" 競艇予測システム デモ実行")
    print("=" * 60)

    # --- モックデータ生成 ---
    print("\n[1/4] モックデータ生成中...")
    np.random.seed(42)
    COURSE_PROBS = {1: 0.553, 2: 0.178, 3: 0.118, 4: 0.082, 5: 0.044, 6: 0.025}
    records = []
    n_races = 500
    for race_id in range(n_races):
        day_offset = race_id // 12
        date = (datetime(2025, 1, 1) + timedelta(days=day_offset)).strftime("%Y%m%d")
        rno = race_id % 12 + 1
        boat_nos = list(range(1, 7))
        true_p = np.array([COURSE_PROBS[b] for b in boat_nos])
        winner = np.random.choice(boat_nos, p=true_p)
        for boat_no in boat_nos:
            rank_approx = boat_no  # 簡略
            records.append({
                "boat_no": boat_no,
                "player_name": f"選手{boat_no}",
                "class": ["A1", "A1", "A2", "B1", "B1", "B2"][boat_no - 1],
                "national_win_rate": max(0.1, 0.55 - boat_no * 0.07 + np.random.normal(0, 0.04)),
                "national_2rate": max(0.2, 0.65 - boat_no * 0.06 + np.random.normal(0, 0.04)),
                "national_3rate": max(0.3, 0.75 - boat_no * 0.05 + np.random.normal(0, 0.04)),
                "local_win_rate": max(0.1, 0.50 - boat_no * 0.07 + np.random.normal(0, 0.04)),
                "local_2rate": max(0.2, 0.60 - boat_no * 0.06 + np.random.normal(0, 0.04)),
                "motor_2rate": max(0.1, 0.40 - boat_no * 0.03 + np.random.normal(0, 0.04)),
                "boat_2rate": max(0.1, 0.38 + np.random.normal(0, 0.04)),
                "exhibit_time": 6.75 + (boat_no - 1) * 0.05 + np.random.normal(0, 0.03),
                "venue_code": "01",
                "venue_name": "桐生",
                "date": date,
                "race_no": rno,
                "rank": rank_approx,
                "win": 1 if boat_no == winner else 0,
            })
    df = pd.DataFrame(records)
    print(f"   生成: {len(df)}件 / {n_races}レース")

    # --- 特徴量 ---
    print("\n[2/4] 特徴量エンジニアリング...")
    X, y = prepare_dataset(df)
    print(f"   特徴量数: {X.shape[1]}  サンプル数: {X.shape[0]}  勝率: {y.mean():.3f}")

    # --- モデル学習 ---
    print("\n[3/4] モデル学習・評価 (TimeSeriesCV)...")
    result = train_evaluate(df, n_splits=4)
    print_feature_importance(result, top_n=8)
    save_model(result)

    # --- バックテスト ---
    dates_all = sorted(df["date"].unique())
    split_date = dates_all[int(len(dates_all) * 0.8)]
    print(f"\n[4/4] バックテスト（{split_date}以降）...")
    test_df = df[df["date"] >= split_date].copy()
    races = []
    for (date, venue, rno), race_df in test_df.groupby(["date", "venue_code", "race_no"]):
        winner_row = race_df[race_df["win"] == 1]
        if winner_row.empty:
            continue
        winner = int(winner_row["boat_no"].values[0])

        pred_df_race = predict_race(result["model"], race_df, result["feature_cols"])
        fair_odds = {b: round(1.0 / COURSE_PROBS[b] * (1 - DEDUCTION_RATE), 1) for b in range(1, 7)}
        races.append({
            "race_id": f"{date}_{rno}",
            "pred_df": pred_df_race,
            "odds": fair_odds,
            "winner": winner,
        })

    session = simulate_session(races, initial_bankroll=50000, min_edge=0.05)
    print_session_report(session)
    _save_results(session, "demo_result.json")

    print("\n" + "=" * 60)
    print(" デモ完了！実データで使うには:")
    print("   python main.py collect --days 30")
    print("   python main.py train")
    print("   python main.py backtest")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="競艇予測システム")
    sub = parser.add_subparsers(dest="command")

    p_collect = sub.add_parser("collect", help="データ収集")
    p_collect.add_argument("--days", type=int, default=30, help="収集日数（デフォルト: 30）")
    p_collect.add_argument("--venues", type=str, default=None, help="場コード(カンマ区切り) 例: 01,02")

    sub.add_parser("train", help="モデル学習")

    p_bt = sub.add_parser("backtest", help="バックテスト")
    p_bt.add_argument("--bankroll", type=float, default=50000, help="初期資金（円）")
    p_bt.add_argument("--min-edge", type=float, default=0.05, help="最低エッジ（デフォルト: 0.05）")

    p_demo = sub.add_parser("demo", help="モックデータでデモ実行")

    args = parser.parse_args()

    if args.command == "collect":
        cmd_collect(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
