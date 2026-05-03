"""
競輪予測システム メインスクリプト

使い方:
  python main.py collect   -- 過去データ収集（実際にスクレイピング）
  python main.py train     -- 収集済みデータでモデル学習
  python main.py backtest  -- バックテスト
  python main.py demo      -- モックデータで全工程デモ実行
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from scraper import collect_data, save_records, VENUE_CODES
from features import build_features, FEATURE_COLS, prepare_dataset
from model import (
    train_evaluate, predict_race, save_model, load_model,
    print_feature_importance, _generate_line_config,
)
from betting import (
    simulate_session, print_session_report, DEDUCTION_RATE,
)

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CLASS_MAP_RATE = {"S1": 0.28, "S2": 0.22, "A1": 0.17, "A2": 0.13, "A3": 0.10, "B1": 0.07}


def cmd_collect(args):
    today = datetime.now()
    end_date = today.strftime("%Y%m%d")
    start_date = (today - timedelta(days=args.days)).strftime("%Y%m%d")
    venue_codes = args.venues.split(",") if args.venues else None
    print(f"収集期間: {start_date} → {end_date}")
    print(f"対象場: {venue_codes or '全場'}")
    records = collect_data(start_date, end_date, venue_codes=venue_codes, sleep_sec=1.5)
    if records:
        save_records(records)
    else:
        print("データが取得できませんでした")


def cmd_train(args):
    raw_path = DATA_DIR / "raw_data.json"
    if not raw_path.exists():
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

    dates = sorted(df_feat["date"].unique())
    split_idx = int(len(dates) * 0.8)
    test_dates = dates[split_idx:]
    df_test = df_feat[df_feat["date"].isin(test_dates)]

    races = []
    for (date, venue, rno), race_df in df_test.groupby(["date", "venue_code", "race_no"]):
        winner_row = race_df[race_df["win"] == 1]
        if winner_row.empty:
            continue
        winner = int(winner_row["car_no"].values[0])

        X = race_df[[c for c in feature_cols if c in race_df.columns]].values
        probs = booster.predict(X)
        probs = probs / probs.sum()

        pred_df = pd.DataFrame({
            "car_no": race_df["car_no"].values,
            "player_name": race_df["player_name"].values,
            "win_prob": probs,
            "is_line_leader": race_df.get("is_line_leader", pd.Series([0] * len(race_df))).values,
            "line_no": race_df.get("line_no", pd.Series([0] * len(race_df))).values,
        })

        n = len(race_df)
        dummy_odds = {j: round((1 - DEDUCTION_RATE) / (1 / n), 1) for j in race_df["car_no"].tolist()}

        races.append({
            "race_id": f"{date}_{venue}_{rno}",
            "pred_df": pred_df,
            "odds": dummy_odds,
            "winner": winner,
        })

    session = simulate_session(
        races,
        initial_bankroll=args.bankroll,
        min_edge=args.min_edge,
        line_leader_only=args.line_leader,
    )
    print_session_report(session)
    _save_results(session)


def _backtest_mock(args):
    np.random.seed(42)
    races = []
    for i in range(300):
        n = np.random.choice([7, 8, 9])
        true_probs = np.random.dirichlet(np.ones(n) * 2)
        winner = np.random.choice(range(1, n + 1), p=true_probs)

        market_probs = true_probs + np.random.normal(0, 0.03, n)
        market_probs = np.clip(market_probs, 0.01, 1)
        market_probs /= market_probs.sum()

        pred_probs = true_probs + np.random.normal(0, 0.015, n)
        pred_probs = np.clip(pred_probs, 0.005, 1)
        pred_probs /= pred_probs.sum()

        market_odds = {j: round((1 - DEDUCTION_RATE) / market_probs[j - 1], 1) for j in range(1, n + 1)}

        line_configs = _generate_line_config(n)
        pred_df = pd.DataFrame({
            "car_no": [c for _, c in line_configs],
            "player_name": [f"選手{c}" for _, c in line_configs],
            "win_prob": pred_probs[:len(line_configs)],
            "is_line_leader": [
                1 if c == min(cc for ll, cc in line_configs if ll == ln) and ln > 0 else 0
                for ln, c in line_configs
            ],
            "line_no": [ln for ln, _ in line_configs],
        })

        races.append({
            "race_id": f"mock_{i+1:03d}",
            "pred_df": pred_df,
            "odds": market_odds,
            "winner": winner,
        })

    session = simulate_session(
        races,
        initial_bankroll=args.bankroll,
        min_edge=args.min_edge,
        line_leader_only=args.line_leader,
    )
    print_session_report(session)
    _save_results(session, "mock_backtest.json")


def cmd_demo(args):
    print("=" * 60)
    print(" 競輪予測システム デモ実行")
    print("=" * 60)

    # --- モックデータ生成 ---
    print("\n[1/4] モックデータ生成中...")
    np.random.seed(42)
    records = []
    n_races = 600

    for race_id in range(n_races):
        day_offset = race_id // 12
        date = (datetime(2025, 1, 1) + timedelta(days=day_offset)).strftime("%Y%m%d")
        rno = race_id % 12 + 1
        bank = np.random.choice([333, 400, 500])
        n_riders = np.random.choice([7, 8, 9])
        line_configs = _generate_line_config(n_riders)

        winner_line = np.random.choice(list(set(ln for ln, _ in line_configs)))
        winner_car = min(c for ln, c in line_configs if ln == winner_line)

        for line_no, cn in line_configs:
            is_leader = (
                1 if cn == min(c for l, c in line_configs if l == line_no) and line_no > 0
                else 0
            )
            line_size = sum(1 for l, _ in line_configs if l == line_no) if line_no > 0 else 1
            cls = np.random.choice(
                ["S1", "S2", "A1", "A2", "A3", "B1"],
                p=[0.05, 0.10, 0.30, 0.30, 0.15, 0.10],
            )
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
    print(f"   生成: {len(df)}件 / {n_races}レース")

    # --- 特徴量 ---
    print("\n[2/4] 特徴量エンジニアリング...")
    X, y = prepare_dataset(df)
    print(f"   特徴量数: {X.shape[1]}  サンプル数: {X.shape[0]}  勝率: {y.mean():.3f}")

    # --- モデル ---
    print("\n[3/4] モデル学習・評価...")
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
        winner = int(winner_row["car_no"].values[0])

        pred_df_race = predict_race(result["model"], race_df, result["feature_cols"])

        n = len(race_df)
        # 市場オッズ：均等確率に少しノイズ（市場の非効率を模擬）
        market_p = np.full(n, 1.0 / n) + np.random.normal(0, 0.02, n)
        market_p = np.clip(market_p, 0.01, 1)
        market_p /= market_p.sum()
        car_nos = race_df["car_no"].tolist()
        market_odds = {car_nos[i]: round((1 - DEDUCTION_RATE) / market_p[i], 1) for i in range(n)}

        races.append({
            "race_id": f"{date}_{rno}",
            "pred_df": pred_df_race,
            "odds": market_odds,
            "winner": winner,
        })

    print("\n  全選手対象:")
    session = simulate_session(races, initial_bankroll=50000, min_edge=0.05)
    print_session_report(session)

    print("\n  ライン先頭のみ:")
    session2 = simulate_session(races, initial_bankroll=50000, min_edge=0.05, line_leader_only=True)
    print_session_report(session2)

    _save_results(session, "demo_result.json")

    print("\n" + "=" * 60)
    print(" デモ完了！実データで使うには:")
    print("   python main.py collect --days 30")
    print("   python main.py train")
    print("   python main.py backtest")
    print("=" * 60)


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


def main():
    parser = argparse.ArgumentParser(description="競輪予測システム")
    sub = parser.add_subparsers(dest="command")

    p_col = sub.add_parser("collect", help="データ収集")
    p_col.add_argument("--days", type=int, default=30)
    p_col.add_argument("--venues", type=str, default=None)

    sub.add_parser("train", help="モデル学習")

    p_bt = sub.add_parser("backtest", help="バックテスト")
    p_bt.add_argument("--bankroll", type=float, default=50000)
    p_bt.add_argument("--min-edge", dest="min_edge", type=float, default=0.05)
    p_bt.add_argument("--line-leader", dest="line_leader", action="store_true",
                      help="ライン先頭のみ対象")

    sub.add_parser("demo", help="デモ実行")

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
