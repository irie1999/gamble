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

from scraper import (
    collect_data, save_records, load_existing_records, VENUE_CODES,
    fetch_daily_schedule, fetch_entry_detail,
)
from features import build_features, FEATURE_COLS, prepare_dataset
from model import (
    train_evaluate, predict_race, save_model, load_model,
    print_feature_importance, _generate_line_config,
)
from betting import simulate_session, print_session_report, DEDUCTION_RATE, make_mock_odds, pick_bets, STRATEGIES

sys.path.insert(0, str(Path(__file__).parent.parent))
import report as html_report
from prob import ALL_BET_TYPES, BET_TYPE_NAMES

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CLASS_MAP_RATE = {"S1": 0.28, "S2": 0.22, "A1": 0.17, "A2": 0.13, "A3": 0.10, "B1": 0.07}


def cmd_compare(args):
    """全戦略をバックテストして比較レポートを生成"""
    model_path = MODEL_DIR / "lgb_model.txt"
    if not model_path.exists():
        print("モデルが未学習です。先に `python main.py train` を実行してください")
        return

    booster, feature_cols, meta = load_model()
    mean_auc = meta.get("metrics", {}).get("mean_auc", 0)

    with open(DATA_DIR / "raw_data.json", encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    df_feat = build_features(df)

    dates = sorted(df_feat["date"].unique())
    split_idx = int(len(dates) * 0.8)
    df_test = df_feat[df_feat["date"].isin(dates[split_idx:])]
    test_days = len(dates) - split_idx

    # 全賭け式でオッズを生成しておく（各戦略がサブセットを選ぶ）
    races_full = _build_races(booster, feature_cols, df_test, ALL_BET_TYPES)

    target_strategies = (
        args.strategies.split(",") if getattr(args, "strategies", None)
        else list(STRATEGIES.keys())
    )

    results = []
    print(f"\nモデルAUC: {mean_auc:.4f}  テスト期間: {test_days}日  レース数: {len(races_full)}\n")
    print(f"{'戦略':<14} {'ROI':>8} {'損益':>12} {'回数':>6} {'的中率':>7} {'賭け金':>12} {'説明'}")
    print("-" * 80)

    for name in target_strategies:
        strat = STRATEGIES.get(name)
        if strat is None:
            print(f"  [{name}] 不明な戦略名 → スキップ")
            continue

        session = simulate_session(
            races_full,
            initial_bankroll=args.bankroll,
            strategy=strat,
        )
        total = session.wins + session.losses
        win_rate = session.wins / total if total > 0 else 0
        results.append({
            "name": name,
            "description": strat["description"],
            "roi": session.roi,
            "profit": session.profit,
            "total_bets": total,
            "win_rate": win_rate,
            "total_bet": session.total_bet,
            "wins": session.wins,
            "losses": session.losses,
            "final_bankroll": session.final_bankroll,
        })
        roi_str = f"{session.roi:+.1%}"
        print(f"  {name:<12} {roi_str:>8} {session.profit:>+12,.0f}円 "
              f"{total:>6} {win_rate:>6.1%} {session.total_bet:>10,.0f}円  {strat['description']}")

    results.sort(key=lambda x: x["roi"], reverse=True)
    best = results[0]["name"] if results else "-"
    print(f"\n最良戦略: {best}\n")

    if args.html:
        _save_compare_html(results, mean_auc, test_days, args.bankroll)


def _save_compare_html(results: list[dict], auc: float, test_days: int, bankroll: float) -> None:
    rows = ""
    for i, r in enumerate(results):
        rank_badge = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
        roi_color = "#4ade80" if r["roi"] >= 0 else "#f87171"
        rows += f"""
        <tr>
          <td>{rank_badge}</td>
          <td><strong>{r['name']}</strong><br><small style="color:#64748b">{r['description']}</small></td>
          <td style="color:{roi_color};font-weight:700">{r['roi']:+.1%}</td>
          <td style="color:{roi_color}">{r['profit']:+,.0f}円</td>
          <td>{r['total_bets']}回</td>
          <td>{r['win_rate']:.1%}</td>
          <td>{r['total_bet']:,.0f}円</td>
          <td>{r['final_bankroll']:,.0f}円</td>
        </tr>"""

    bar_html = ""
    max_roi = max(abs(r["roi"]) for r in results) or 1
    for r in results:
        pct = r["roi"] / max_roi * 100
        color = "#4ade80" if r["roi"] >= 0 else "#f87171"
        bar_html += f"""
        <div style="margin-bottom:.8rem">
          <div style="display:flex;justify-content:space-between;font-size:.82rem;color:#94a3b8;margin-bottom:.25rem">
            <span>{r['name']}</span><span style="color:{color}">{r['roi']:+.1%}</span>
          </div>
          <div style="background:#0f172a;border-radius:4px;height:10px">
            <div style="width:{abs(pct):.0f}%;background:{color};height:100%;border-radius:4px"></div>
          </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="UTF-8"><title>戦略比較</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0}}
  .header{{background:#1e293b;padding:2rem;border-bottom:1px solid #334155}}
  .header h1{{font-size:1.6rem;font-weight:700}}
  .header .sub{{color:#94a3b8;margin-top:.3rem;font-size:.9rem}}
  .container{{max-width:1100px;margin:0 auto;padding:2rem}}
  .kpi{{display:flex;gap:1rem;margin-bottom:2rem;flex-wrap:wrap}}
  .kpi-card{{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:1rem 1.5rem}}
  .kpi-card .label{{font-size:.75rem;color:#64748b;text-transform:uppercase}}
  .kpi-card .value{{font-size:1.4rem;font-weight:700;color:#f1f5f9;margin-top:.2rem}}
  .section{{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:1.5rem;margin-bottom:1.5rem}}
  .section h2{{font-size:.95rem;font-weight:600;color:#94a3b8;margin-bottom:1rem;text-transform:uppercase}}
  table{{width:100%;border-collapse:collapse;font-size:.88rem}}
  th{{background:#0f172a;color:#64748b;padding:.6rem .8rem;text-align:left;font-size:.75rem;text-transform:uppercase}}
  td{{padding:.6rem .8rem;border-bottom:1px solid #0f172a;color:#cbd5e1}}
  tr:hover td{{background:#263347}}
  .warn{{background:#1e3a2f;border:1px solid #166534;border-radius:8px;padding:1rem;margin-top:1.5rem;color:#4ade80;font-size:.85rem}}
</style>
</head><body>
<div class="header">
  <h1>⚖️ 戦略比較レポート</h1>
  <div class="sub">生成: {datetime.now().strftime('%Y-%m-%d %H:%M')} ／ モデルAUC: {auc:.4f} ／ テスト: {test_days}日</div>
</div>
<div class="container">
  <div class="kpi">
    <div class="kpi-card"><div class="label">比較戦略数</div><div class="value">{len(results)}種</div></div>
    <div class="kpi-card"><div class="label">最良戦略</div><div class="value">{results[0]['name']}</div></div>
    <div class="kpi-card"><div class="label">最良ROI</div><div class="value" style="color:#4ade80">{results[0]['roi']:+.1%}</div></div>
    <div class="kpi-card"><div class="label">初期資金</div><div class="value">¥{bankroll:,.0f}</div></div>
  </div>

  <div class="section">
    <h2>ROI 比較</h2>
    {bar_html}
  </div>

  <div class="section">
    <h2>詳細比較（ROI順）</h2>
    <table>
      <thead><tr>
        <th>順位</th><th>戦略</th><th>ROI</th><th>損益</th>
        <th>賭け回数</th><th>的中率</th><th>総賭け金</th><th>最終資金</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
  <div class="warn">
    ⚠ オッズは過去勝率ベースの推定値です。実際の市場オッズとは異なります。ROIの絶対値より各戦略の相対比較にご活用ください。
  </div>
</div></body></html>"""

    path = RESULTS_DIR / "strategy_compare.html"
    path.write_text(html, encoding="utf-8")
    print(f"比較レポート保存: {path}")
    import webbrowser
    webbrowser.open(path.as_uri())


def cmd_predict(args):
    """今日の出走表からベッティングシグナルを生成"""
    model_path = MODEL_DIR / "lgb_model.txt"
    if not model_path.exists():
        print("モデルが未学習です。先に `python main.py train` を実行してください")
        return

    booster, feature_cols, meta = load_model()
    mean_auc = meta.get("metrics", {}).get("mean_auc", 0)

    date_str = args.date or datetime.now().strftime("%Y%m%d")
    bet_types = args.bet_types.split(",") if args.bet_types else ALL_BET_TYPES
    bankroll = args.bankroll

    print(f"\n{'='*60}")
    print(f"  競輪ベッティングシグナル  {date_str}")
    print(f"  モデルAUC: {mean_auc:.4f}  資金: {bankroll:,.0f}円")
    print(f"{'='*60}\n")

    races = fetch_daily_schedule(date_str)
    if not races:
        print(f"{date_str} のレース情報が取得できませんでした")
        print("※ネットワーク接続またはkdreams.jpへのアクセスを確認してください")
        return

    print(f"{len(races)}レース検出\n")

    any_signal = False
    signal_rows = []

    for race in sorted(races, key=lambda r: (r["venue_code"], r["race_no"])):
        records = fetch_entry_detail(race)
        if not records:
            continue

        df = pd.DataFrame(records).drop_duplicates(subset="car_no", keep="first")
        df_feat = build_features(df)

        feat_cols_available = [c for c in feature_cols if c in df_feat.columns]
        X = df_feat[feat_cols_available].fillna(0).values
        if len(X) == 0:
            continue

        probs = booster.predict(X)
        probs /= probs.sum()

        pred_df = pd.DataFrame({
            "car_no": df["car_no"].values,
            "player_name": df["player_name"].values,
            "win_prob": probs,
            "is_line_leader": df["is_line_leader"].fillna(0).astype(int).values,
            "line_no": df["line_no"].fillna(0).astype(int).values,
        })

        nos = df["car_no"].astype(int).tolist()
        market_p = df["win_rate"].fillna(1 / len(nos)).values.astype(float)
        market_p = np.clip(market_p, 0.01, 1)
        market_p /= market_p.sum()
        odds_dict = make_mock_odds(nos, market_p, bet_types, noise=0.0)

        race_bets = []
        for bt in bet_types:
            bets = pick_bets(pred_df, odds_dict.get(bt, {}), bankroll, bt)
            race_bets.extend(bets)

        if not race_bets:
            continue

        any_signal = True
        header = f"{race['venue_name']} R{race['race_no']}"
        print(f"【{header}】")

        pred_sorted = pred_df.sort_values("win_prob", ascending=False)
        parts = []
        for _, row in pred_sorted.iterrows():
            parts.append(f"{int(row['car_no'])}番({row['win_prob']:.0%})")
        print(f"  予測: {' > '.join(parts)}")

        for b in race_bets:
            bt_name = BET_TYPE_NAMES.get(b.bet_type, b.bet_type)
            sels = list(b.selections)
            print(f"  ▶ {bt_name} {sels}  オッズ{b.odds:.1f}倍  "
                  f"推奨額 {b.bet_amount:,}円  エッジ {b.edge:+.3f}  EV {b.expected_value:.2f}")
            signal_rows.append({
                "venue": race["venue_name"],
                "race_no": race["race_no"],
                "bet_type": bt_name,
                "selections": str(sels),
                "odds": b.odds,
                "bet_amount": b.bet_amount,
                "edge": b.edge,
                "ev": b.expected_value,
                "pred_prob": b.predicted_prob,
            })
        print()

    if not any_signal:
        print("本日のシグナルはありません（エッジ不足）")
        return

    total_bet = sum(r["bet_amount"] for r in signal_rows)
    print(f"合計推奨ベット額: {total_bet:,}円 / {len(signal_rows)}件")
    print("\n⚠ オッズは過去勝率ベースの推定値です。実際のオッズで金額を調整してください。\n")

    if args.html:
        _save_signal_html(signal_rows, date_str, mean_auc, bankroll)


def _save_signal_html(signal_rows: list[dict], date_str: str, auc: float, bankroll: float) -> None:
    rows_html = ""
    for r in signal_rows:
        rows_html += f"""
        <tr>
          <td>{r['venue']} R{r['race_no']}</td>
          <td>{r['bet_type']}</td>
          <td>{r['selections']}</td>
          <td>{r['pred_prob']:.1%}</td>
          <td>{r['odds']:.1f}倍</td>
          <td><span style="color:{'#4ade80' if r['edge']>=0 else '#f87171'}">{r['edge']:+.3f}</span></td>
          <td>{r['ev']:.2f}</td>
          <td><strong>{r['bet_amount']:,}円</strong></td>
        </tr>"""

    total = sum(r["bet_amount"] for r in signal_rows)
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head><meta charset="UTF-8"><title>競輪シグナル {date_str}</title>
<style>
  * {{box-sizing:border-box;margin:0;padding:0}}
  body {{font-family:'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}}
  .header {{background:#1e293b;padding:2rem;border-bottom:1px solid #334155}}
  .header h1 {{font-size:1.6rem;font-weight:700}}
  .header .sub {{color:#94a3b8;margin-top:.3rem;font-size:.9rem}}
  .container {{max-width:1200px;margin:0 auto;padding:2rem}}
  .kpi {{display:flex;gap:1rem;margin-bottom:2rem;flex-wrap:wrap}}
  .kpi-card {{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:1rem 1.5rem}}
  .kpi-card .label {{font-size:.75rem;color:#64748b;text-transform:uppercase}}
  .kpi-card .value {{font-size:1.4rem;font-weight:700;color:#f1f5f9;margin-top:.2rem}}
  .section {{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:1.5rem}}
  table {{width:100%;border-collapse:collapse;font-size:.88rem}}
  th {{background:#0f172a;color:#64748b;padding:.6rem .8rem;text-align:left;font-size:.75rem;text-transform:uppercase}}
  td {{padding:.55rem .8rem;border-bottom:1px solid #0f172a;color:#cbd5e1}}
  tr:hover td {{background:#263347}}
  .warn {{background:#1e3a2f;border:1px solid #166534;border-radius:8px;padding:1rem;margin-top:1.5rem;color:#4ade80;font-size:.88rem}}
</style>
</head>
<body>
<div class="header">
  <h1>📡 競輪ベッティングシグナル</h1>
  <div class="sub">対象日: {date_str} ／ 生成: {datetime.now().strftime('%Y-%m-%d %H:%M')} ／ モデルAUC: {auc:.4f}</div>
</div>
<div class="container">
  <div class="kpi">
    <div class="kpi-card"><div class="label">シグナル件数</div><div class="value">{len(signal_rows)}件</div></div>
    <div class="kpi-card"><div class="label">合計推奨ベット額</div><div class="value">¥{total:,}</div></div>
    <div class="kpi-card"><div class="label">資金</div><div class="value">¥{bankroll:,.0f}</div></div>
  </div>
  <div class="section">
    <table>
      <thead><tr>
        <th>レース</th><th>賭け式</th><th>選択</th>
        <th>予測P</th><th>オッズ(推定)</th><th>エッジ</th><th>EV</th><th>推奨額</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
  <div class="warn">
    ⚠ オッズは過去勝率ベースの推定値です。実際のオッズを確認してから賭け額を調整してください。
  </div>
</div>
</body></html>"""

    path = RESULTS_DIR / f"signal_{date_str}.html"
    path.write_text(html, encoding="utf-8")
    print(f"シグナルレポート保存: {path}")
    import webbrowser
    webbrowser.open(path.as_uri())


def cmd_collect(args):
    today = datetime.now()
    end_date = today.strftime("%Y%m%d")
    venue_codes = args.venues.split(",") if args.venues else None

    existing_records = []
    if not args.full:
        existing_records, latest_date = load_existing_records("raw_data.json")
        if latest_date:
            resume_date = (datetime.strptime(latest_date, "%Y%m%d") + timedelta(days=1))
            start_date = resume_date.strftime("%Y%m%d")
            print(f"既存データ: {len(existing_records)}件 (最終: {latest_date})")
            print(f"増分収集: {start_date} → {end_date}")
            if start_date > end_date:
                print("最新データがあります。収集不要です。")
                return
        else:
            start_date = (today - timedelta(days=args.days)).strftime("%Y%m%d")
            print(f"新規収集: {start_date} → {end_date}")
    else:
        start_date = (today - timedelta(days=args.days)).strftime("%Y%m%d")
        print(f"全期間収集（--full）: {start_date} → {end_date}")

    print(f"対象場: {venue_codes or '全場'}")
    collect_data(
        start_date, end_date,
        venue_codes=venue_codes,
        sleep_sec=args.sleep,
        existing_records=existing_records,
        checkpoint_days=7,
        filename="raw_data.json",
        workers=args.workers,
    )


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
    bet_types = args.bet_types.split(",") if args.bet_types else ALL_BET_TYPES
    if raw_path.exists() and model_path.exists():
        print("実モデルでバックテスト実行...")
        _backtest_real(args, bet_types)
    else:
        print("モデル or データが未作成のためモックデータでバックテスト実行...")
        _backtest_mock(args, bet_types)


def _build_races(booster, feature_cols, df_feat, bet_types):
    """バックテスト用レースリストを構築（モデル予測 + モックオッズ）"""
    races = []
    for (date, venue, rno), race_df in df_feat.groupby(["date", "venue_code", "race_no"]):
        race_df = race_df.drop_duplicates(subset="car_no", keep="first")
        winner_row = race_df[race_df["win"] == 1]
        if winner_row.empty:
            continue
        winner = int(winner_row["car_no"].values[0])
        finish = [winner] + [c for c in race_df["car_no"].tolist() if c != winner]

        X = race_df[[c for c in feature_cols if c in race_df.columns]].values
        probs = booster.predict(X)
        probs = probs / probs.sum()

        pred_df = pd.DataFrame({
            "car_no": race_df["car_no"].values,
            "player_name": race_df["player_name"].values,
            "win_prob": probs,
            "is_line_leader": race_df.get("is_line_leader", pd.Series([0]*len(race_df))).values,
            "line_no": race_df.get("line_no", pd.Series([0]*len(race_df))).values,
        })
        nos = race_df["car_no"].astype(int).tolist()
        market_p = race_df["win_rate"].fillna(1 / len(nos)).values.astype(float)
        market_p = np.clip(market_p, 0.01, 1)
        market_p /= market_p.sum()
        odds = make_mock_odds(nos, market_p, bet_types, noise=0.05)
        races.append({"race_id": f"{date}_{venue}_{rno}", "pred_df": pred_df,
                       "odds": odds, "finish_order": finish})
    return races


def _backtest_real(args, bet_types):
    booster, feature_cols, meta = load_model()

    with open(DATA_DIR / "raw_data.json", encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    df_feat = build_features(df)

    dates = sorted(df_feat["date"].unique())
    split_idx = int(len(dates) * 0.8)
    df_test = df_feat[df_feat["date"].isin(dates[split_idx:])]

    strategy = STRATEGIES.get(getattr(args, "strategy", None) or "")
    races = _build_races(booster, feature_cols, df_test, bet_types)
    session = simulate_session(races, initial_bankroll=args.bankroll,
                               bet_types=bet_types,
                               line_leader_only=getattr(args, "line_leader", False),
                               strategy=strategy)
    print_session_report(session)
    _save_results(session, html=getattr(args, "html", False))


def _backtest_mock(args, bet_types):
    np.random.seed(42)
    races = []
    for i in range(300):
        n = np.random.choice([7, 8, 9])
        nos = list(range(1, n + 1))
        true_p = np.random.dirichlet(np.ones(n) * 2)
        finish = list(np.random.choice(nos, size=min(n, 3), replace=False, p=true_p))
        while len(finish) < 3:
            finish.append(nos[len(finish)])

        market_p = true_p + np.random.normal(0, 0.03, n)
        market_p = np.clip(market_p, 0.005, 1)
        market_p /= market_p.sum()

        pred_p = true_p + np.random.normal(0, 0.015, n)
        pred_p = np.clip(pred_p, 0.005, 1)
        pred_p /= pred_p.sum()

        line_configs = _generate_line_config(n)
        pred_df = pd.DataFrame({
            "car_no": [c for _, c in line_configs],
            "player_name": [f"選手{c}" for _, c in line_configs],
            "win_prob": pred_p[:len(line_configs)],
            "is_line_leader": [
                1 if c == min(cc for ll, cc in line_configs if ll == ln) and ln > 0 else 0
                for ln, c in line_configs
            ],
            "line_no": [ln for ln, _ in line_configs],
        })
        odds = make_mock_odds(nos, market_p, bet_types, noise=0.0)
        races.append({"race_id": f"mock_{i+1:03d}", "pred_df": pred_df,
                       "odds": odds, "finish_order": finish})

    session = simulate_session(races, initial_bankroll=args.bankroll,
                               bet_types=bet_types, line_leader_only=args.line_leader)
    print_session_report(session)
    _save_results(session, "mock_backtest.json", html=getattr(args, "html", False))


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
        finish = [winner] + [c for c in race_df["car_no"].tolist() if c != winner]

        pred_df_race = predict_race(result["model"], race_df, result["feature_cols"])

        n = len(race_df)
        market_p = np.full(n, 1.0 / n) + np.random.normal(0, 0.03, n)
        market_p = np.clip(market_p, 0.01, 1)
        market_p /= market_p.sum()
        car_nos = race_df["car_no"].astype(int).tolist()
        odds = make_mock_odds(car_nos, market_p, ALL_BET_TYPES, noise=0.0)

        races.append({
            "race_id": f"{date}_{rno}",
            "pred_df": pred_df_race,
            "odds": odds,
            "finish_order": finish,
        })

    print("\n  全選手対象（全賭け式）:")
    session = simulate_session(races, initial_bankroll=50000, bet_types=ALL_BET_TYPES)
    print_session_report(session)

    print("\n  ライン先頭のみ（全賭け式）:")
    session2 = simulate_session(races, initial_bankroll=50000, bet_types=ALL_BET_TYPES,
                                line_leader_only=True)
    print_session_report(session2)

    _save_results(session, "demo_result.json", html=True)

    print("\n" + "=" * 60)
    print(" デモ完了！実データで使うには:")
    print("   python main.py collect --days 30")
    print("   python main.py train")
    print("   python main.py backtest --html")
    print("=" * 60)


def _save_results(session, filename: str = "backtest_result.json", html: bool = False) -> None:
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
                "win_flag": b.win_flag,
            }
            for b in session.bets
        ],
    }
    json_path = RESULTS_DIR / filename
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n結果を保存しました: {json_path}")

    if html:
        html_path = RESULTS_DIR / filename.replace(".json", ".html")
        html_report.save_and_open(result, html_path, title="競輪バックテスト結果")


def main():
    parser = argparse.ArgumentParser(description="競輪予測システム")
    sub = parser.add_subparsers(dest="command")

    p_col = sub.add_parser("collect", help="データ収集")
    p_col.add_argument("--days", type=int, default=365,
                       help="収集日数（新規 or --full 時。デフォルト: 365日）")
    p_col.add_argument("--venues", type=str, default=None,
                       help="場コードカンマ区切り（デフォルト: 全場）")
    p_col.add_argument("--full", action="store_true",
                       help="既存データを無視して全期間再収集")
    p_col.add_argument("--sleep", type=float, default=1.5,
                       help="リクエスト間隔（秒、デフォルト: 1.5）")
    p_col.add_argument("--workers", type=int, default=4,
                       help="並列処理数（デフォルト: 4）")

    sub.add_parser("train", help="モデル学習")

    p_bt = sub.add_parser("backtest", help="バックテスト")
    p_bt.add_argument("--bankroll", type=float, default=50000)
    p_bt.add_argument("--bet-types", dest="bet_types", type=str, default=None,
                      help=f"賭け式カンマ区切り (デフォルト:全式) 選択肢: {','.join(ALL_BET_TYPES)}")
    p_bt.add_argument("--line-leader", dest="line_leader", action="store_true",
                      help="ライン先頭のみ対象")
    p_bt.add_argument("--strategy", type=str, default=None,
                      help=f"戦略プリセット: {','.join(STRATEGIES.keys())}")
    p_bt.add_argument("--html", action="store_true", help="HTMLレポートを生成してブラウザで開く")

    p_cmp = sub.add_parser("compare", help="全戦略を一括バックテストして比較")
    p_cmp.add_argument("--bankroll", type=float, default=50000)
    p_cmp.add_argument("--strategies", type=str, default=None,
                       help=f"比較する戦略カンマ区切り（デフォルト:全戦略）: {','.join(STRATEGIES.keys())}")
    p_cmp.add_argument("--html", action="store_true", help="HTMLレポートを生成してブラウザで開く")

    sub.add_parser("demo", help="デモ実行")

    p_pred = sub.add_parser("predict", help="本日のシグナル生成")
    p_pred.add_argument("--date", type=str, default=None,
                        help="対象日 YYYYMMDD（デフォルト: 今日）")
    p_pred.add_argument("--bankroll", type=float, default=50000,
                        help="資金（Kelly計算用、デフォルト: 50000）")
    p_pred.add_argument("--bet-types", dest="bet_types", type=str, default=None,
                        help=f"賭け式カンマ区切り (デフォルト:全式) 選択肢: {','.join(ALL_BET_TYPES)}")
    p_pred.add_argument("--html", action="store_true", help="HTMLレポートを生成してブラウザで開く")

    args = parser.parse_args()
    if args.command == "collect":
        cmd_collect(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
