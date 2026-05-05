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
    fetch_daily_schedule, fetch_entry_detail, fetch_race_odds, load_odds_data,
)
from keirin_jp import (
    fetch_race_page, get_payout_odds, fetch_live_odds, fetch_today_races,
    get_race_encp, VENUE_CODE_TO_KCD,
)
from features import build_features, FEATURE_COLS, prepare_dataset
from model import (
    train_evaluate, predict_race, save_model, load_model,
    print_feature_importance, _generate_line_config,
)
from betting import simulate_session, print_session_report, make_mock_odds, pick_bets, STRATEGIES

sys.path.insert(0, str(Path(__file__).parent.parent))
import report as html_report
from prob import ALL_BET_TYPES, BET_TYPE_NAMES, KEIRIN_BET_TYPES

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
    mean_auc = meta.get("metrics", {}).get("cv_auc", meta.get("metrics", {}).get("mean_auc", 0))

    with open(DATA_DIR / "raw_data.json", encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    df_feat = build_features(df)

    dates = sorted(df_feat["date"].unique())
    test_ratio = getattr(args, "test_ratio", 0.3)
    split_idx = int(len(dates) * (1 - test_ratio))
    df_test = df_feat[df_feat["date"].isin(dates[split_idx:])]
    test_days = len(dates) - split_idx
    print(f"テスト分割: {1-test_ratio:.0%}学習 / {test_ratio:.0%}テスト  "
          f"({split_idx}日学習 / {test_days}日テスト)")

    # 全賭け式でオッズを生成しておく（各戦略がサブセットを選ぶ）
    races_full = _build_races(booster, feature_cols, df_test, KEIRIN_BET_TYPES)

    target_strategies = (
        args.strategies.split(",") if getattr(args, "strategies", None)
        else list(STRATEGIES.keys())
    )
    fixed_bet = getattr(args, "fixed_bet", None)
    mode_label = f"固定額{fixed_bet:,}円/bet" if fixed_bet else "Kelly基準"

    results = []
    print(f"\nモデルAUC: {mean_auc:.4f}  テスト期間: {test_days}日  "
          f"レース数: {len(races_full)}  [{mode_label}]\n")
    print(f"{'戦略':<14} {'ROI':>8} {'損益':>12} {'回数':>6} {'的中率':>7} {'賭け金':>12} {'説明'}")
    print("-" * 90)

    for name in target_strategies:
        strat = STRATEGIES.get(name)
        if strat is None:
            print(f"  [{name}] 不明な戦略名 → スキップ")
            continue

        run_strat = dict(strat)
        if fixed_bet:
            run_strat["fixed_bet"] = fixed_bet

        session = simulate_session(
            races_full,
            initial_bankroll=args.bankroll,
            strategy=run_strat,
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
        print(f"  {name:<14} {roi_str:>8} {session.profit:>+12,.0f}円 "
              f"{total:>6} {win_rate:>6.1%} {session.total_bet:>10,.0f}円  {strat['description']}")

    results.sort(key=lambda x: x["roi"], reverse=True)
    best = results[0]["name"] if results else "-"
    print(f"\n最良戦略: {best}  ※オッズは推定値のためROIの絶対値は参考程度\n")

    if args.html:
        _save_compare_html(results, mean_auc, test_days, args.bankroll, mode_label)


def _save_compare_html(results: list[dict], auc: float, test_days: int, bankroll: float, mode_label: str = "") -> None:
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
  <div class="sub">生成: {datetime.now().strftime('%Y-%m-%d %H:%M')} ／ モデルAUC: {auc:.4f} ／ テスト: {test_days}日 ／ {mode_label}</div>
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
    mean_auc = meta.get("metrics", {}).get("cv_auc", meta.get("metrics", {}).get("mean_auc", 0))

    if args.date:
        date_str = args.date
    elif getattr(args, "today", False):
        date_str = datetime.now().strftime("%Y%m%d")
    else:
        # デフォルトは明日（出走表は前日公開のため）
        date_str = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")
    bet_types = args.bet_types.split(",") if args.bet_types else KEIRIN_BET_TYPES
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

    top_n = getattr(args, "top_n", 20)
    min_ev = getattr(args, "min_ev", 1.5)
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

        # keirin.jp ライブオッズを優先取得
        odds_dict = {}
        kcd = VENUE_CODE_TO_KCD.get(str(race.get("venue_code", "")))
        if kcd:
            try:
                jdata = fetch_race_page(kcd, date_str, race["race_no"])
                if jdata:
                    encp = get_race_encp(jdata, race["race_no"])
                    if encp:
                        odds_dict = fetch_live_odds(encp, ["exacta", "quinella", "trifecta", "trio"])
                    if not odds_dict:
                        odds_dict = get_payout_odds(jdata)
            except Exception:
                pass

        # フォールバック: kdreams.jp スクレイピング
        if not odds_dict:
            odds_dict = fetch_race_odds(race, bet_types=["trifecta", "trio", "wide"])

        if not odds_dict:
            continue

        pred_sorted = pred_df.sort_values("win_prob", ascending=False)
        pred_str = " > ".join(
            f"{int(r['car_no'])}番({r['win_prob']:.0%})"
            for _, r in pred_sorted.iterrows()
        )

        for bt in bet_types:
            for b in pick_bets(pred_df, odds_dict.get(bt, {}), bankroll, bt):
                if b.expected_value < min_ev:
                    continue
                signal_rows.append({
                    "venue": race["venue_name"],
                    "race_no": race["race_no"],
                    "bet_type": BET_TYPE_NAMES.get(b.bet_type, b.bet_type),
                    "selections": str(list(b.selections)),
                    "odds": b.odds,
                    "bet_amount": b.bet_amount,
                    "edge": b.edge,
                    "ev": b.expected_value,
                    "pred_prob": b.predicted_prob,
                    "pred_str": pred_str,
                })

    if not signal_rows:
        print(f"シグナルがありません（EV≥{min_ev} の条件を満たすベットなし）")
        return

    # EV 降順でソートし上位 top_n を「おすすめ」とする
    signal_rows.sort(key=lambda x: x["ev"], reverse=True)
    top_picks = signal_rows[:top_n]

    print(f"\n━━━ TOP {top_n} おすすめベット（EV順） ━━━")
    print(f"{'#':<3} {'レース':<10} {'賭け式':<6} {'選択':<12} "
          f"{'予測P':>6} {'オッズ':>8} {'エッジ':>7} {'EV':>5} {'推奨額':>8}")
    print("-" * 72)
    for i, r in enumerate(top_picks, 1):
        print(f"{i:<3} {r['venue']} R{r['race_no']:<4} {r['bet_type']:<6} "
              f"{r['selections']:<12} {r['pred_prob']:>5.1%} {r['odds']:>6.1f}倍 "
              f"{r['edge']:>+6.3f} {r['ev']:>5.2f} {r['bet_amount']:>7,}円")

    total_top = sum(r["bet_amount"] for r in top_picks)
    print(f"\n合計推奨額（TOP{top_n}）: {total_top:,}円")
    print(f"全シグナル: {len(signal_rows)}件（EV≥{min_ev}）")
    print("\n⚠ オッズは推定値です。実際のオッズを確認してから賭け額を調整してください。\n")

    if args.html:
        _save_signal_html(signal_rows, top_picks, date_str, mean_auc, bankroll, top_n)


def _save_signal_html(
    signal_rows: list[dict],
    top_picks: list[dict],
    date_str: str,
    auc: float,
    bankroll: float,
    top_n: int = 20,
) -> None:
    def make_row(r: dict, highlight: bool = False) -> str:
        bg = ' style="background:#1a2e1a"' if highlight else ""
        return f"""
        <tr{bg}>
          <td>{r['venue']} R{r['race_no']}</td>
          <td>{r['bet_type']}</td>
          <td>{r['selections']}</td>
          <td>{r['pred_prob']:.1%}</td>
          <td>{r['odds']:.1f}倍</td>
          <td><span style="color:{'#4ade80' if r['edge']>=0 else '#f87171'}">{r['edge']:+.3f}</span></td>
          <td><strong style="color:#fbbf24">{r['ev']:.2f}</strong></td>
          <td><strong>{r['bet_amount']:,}円</strong></td>
        </tr>"""

    top_rows_html = "".join(make_row(r, highlight=True) for r in top_picks)
    all_rows_html = "".join(make_row(r) for r in signal_rows)

    total_top = sum(r["bet_amount"] for r in top_picks)
    total_all = sum(r["bet_amount"] for r in signal_rows)

    table_header = """
      <thead><tr>
        <th>レース</th><th>賭け式</th><th>選択</th>
        <th>予測P</th><th>オッズ(推定)</th><th>エッジ</th><th>EV</th><th>推奨額</th>
      </tr></thead>"""

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
  .section {{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:1.5rem;margin-bottom:1.5rem}}
  .section h2 {{font-size:.95rem;font-weight:600;color:#94a3b8;margin-bottom:1rem;text-transform:uppercase}}
  .top-badge {{display:inline-block;background:#d97706;color:#fff;font-size:.7rem;font-weight:700;
               border-radius:4px;padding:.1rem .4rem;margin-right:.4rem;vertical-align:middle}}
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
    <div class="kpi-card"><div class="label">全シグナル</div><div class="value">{len(signal_rows)}件</div></div>
    <div class="kpi-card"><div class="label">おすすめ TOP{top_n}</div><div class="value" style="color:#fbbf24">{len(top_picks)}件</div></div>
    <div class="kpi-card"><div class="label">TOP推奨額合計</div><div class="value" style="color:#fbbf24">¥{total_top:,}</div></div>
    <div class="kpi-card"><div class="label">資金</div><div class="value">¥{bankroll:,.0f}</div></div>
  </div>

  <div class="section">
    <h2><span class="top-badge">TOP {top_n}</span> おすすめベット（EV順）</h2>
    <table>
      {table_header}
      <tbody>{top_rows_html}</tbody>
    </table>
  </div>

  <div class="section">
    <h2>全シグナル一覧（{len(signal_rows)}件）</h2>
    <table>
      {table_header}
      <tbody>{all_rows_html}</tbody>
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


def cmd_pipeline(args):
    """払戻取得→着順修正→学習→戦略比較を一括実行。済みのステップはスキップ。"""
    import types
    from keirin.scripts.collect_payouts import build_race_list, fetch_and_store, load_raw, load_odds, save_odds
    from keirin.scripts.fix_ranks import fix_ranks as _fix_ranks, load_json, save_json

    raw_path  = DATA_DIR / "raw_data.json"
    odds_path = DATA_DIR / "odds_data.json"
    model_path = MODEL_DIR / "lgb_model.txt"

    def _step(label):
        print(f"\n{'─'*55}")
        print(f"  {label}")
        print(f"{'─'*55}")

    # ── Step 1: collect_payouts ──────────────────────────────
    _step("Step 1: 払戻データ取得")
    raw_data = load_raw()
    existing_odds = load_odds()

    need_payouts = args.force_payouts
    if not need_payouts:
        missing = sum(
            1 for r in raw_data
            if r.get("race_id") and (
                r["race_id"] not in existing_odds or
                "trifecta" not in existing_odds.get(r["race_id"], {})
            )
        )
        need_payouts = missing > 0
        if not need_payouts:
            print(f"  ✓ 全レースの払戻データ取得済み → スキップ")
        else:
            print(f"  → trifecta未取得: {missing}レース → 取得開始")

    if need_payouts:
        races = build_race_list(raw_data, None)
        updated = fetch_and_store(races, existing_odds, overwrite=args.force_payouts)
        save_odds(updated)
        print(f"  保存完了: {len(updated)}件")

    # ── Step 2: fix_ranks ───────────────────────────────────
    _step("Step 2: 着順修正")
    raw_data = load_json(raw_path)
    ranked = [r for r in raw_data if r.get("rank") is not None]
    if ranked:
        eq_rate = sum(1 for r in ranked if r["car_no"] == r["rank"]) / len(ranked)
    else:
        eq_rate = 1.0

    if eq_rate <= 0.25:
        print(f"  ✓ car_no=rank率: {eq_rate:.1%} → 修正済み スキップ")
    else:
        print(f"  → car_no=rank率: {eq_rate:.1%} → 修正開始")
        odds_raw = load_json(odds_path) if odds_path.exists() else {}
        fixed_data, stats = _fix_ranks(raw_data, odds_raw)
        ranked2 = [r for r in fixed_data if r.get("rank") is not None]
        eq2 = sum(1 for r in ranked2 if r["car_no"] == r["rank"]) / max(1, len(ranked2))
        save_json(raw_path, fixed_data)
        print(f"  修正完了: {stats['fixed']}レース  car_no=rank率: {eq2:.1%}")

    # ── Step 3: train ────────────────────────────────────────
    _step("Step 3: モデル学習")
    need_train = args.force_train or not model_path.exists()
    if not need_train:
        raw_mtime   = raw_path.stat().st_mtime
        model_mtime = model_path.stat().st_mtime
        need_train  = raw_mtime > model_mtime
        if not need_train:
            print(f"  ✓ モデル学習済み（raw_dataより新しい）→ スキップ")

    if need_train:
        with open(raw_path, encoding="utf-8") as f:
            records = json.load(f)
        df = pd.DataFrame(records)
        print(f"  学習データ: {len(df)}件")
        result = train_evaluate(df, n_splits=5)
        if result is None:
            print("  ✗ データ不足で学習できません（最低7日分必要）")
            return
        print_feature_importance(result)
        save_model(result)

    # ── Step 4: compare ──────────────────────────────────────
    _step("Step 4: 戦略比較")
    cmd_compare(args)


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
    bet_types = args.bet_types.split(",") if args.bet_types else KEIRIN_BET_TYPES
    if not raw_path.exists():
        print("データがありません。先に `python main.py collect` を実行してください")
        return
    if not model_path.exists():
        print("モデルがありません。先に `python main.py train` を実行してください")
        return
    print("実モデルでバックテスト実行...")
    _backtest_real(args, bet_types)


def _fetch_keirin_jp_odds(venue_code: str, date: str, race_no: int) -> dict:
    """keirin.jp から払戻金をオッズとして取得する。失敗時は空dict。"""
    kcd = VENUE_CODE_TO_KCD.get(str(venue_code))
    if kcd is None:
        return {}
    try:
        jdata = fetch_race_page(kcd, date, race_no)
        if jdata:
            return get_payout_odds(jdata)
    except Exception:
        pass
    return {}


def _build_races(booster, feature_cols, df_feat, bet_types, odds_data: dict | None = None):
    """バックテスト用レースリストを構築（モデル予測 + 払戻オッズ）"""
    # keirin.jp 払戻データを優先。取れない場合は scraper 収集の odds_data にフォールバック
    if odds_data is None:
        odds_data = load_odds_data()

    races = []
    skipped = 0
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

        # ローカルodds_data優先（collect_payouts.pyで収集済み）
        odds = {}
        if odds_data:
            for rid, od in odds_data.items():
                if rid[2:10] == date and rid[:2] == str(venue) and int(rid[14:16]) == rno:
                    # keirin.jp払戻データ（trifecta/trio/wide）のみ使用
                    odds = {bt: d for bt, d in od.items()
                            if bt in ("trifecta", "trio", "wide", "exacta", "quinella")}
                    break

        # フォールバック: keirin.jpから直接取得（ローカルになければ）
        if not odds:
            odds = _fetch_keirin_jp_odds(venue, date, rno)

        if not odds:
            skipped += 1
            continue

        races.append({"race_id": f"{date}_{venue}_{rno}", "pred_df": pred_df,
                       "odds": odds, "finish_order": finish})

    if skipped:
        print(f"  ※オッズデータなし: {skipped}レーススキップ")
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
                "win_flag": bool(b.win_flag),
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
    p_cmp.add_argument("--fixed-bet", dest="fixed_bet", type=int, default=None,
                       help="固定額ベット（例: 200）。Kellyの複利を排除し純粋な戦略比較ができる")
    p_cmp.add_argument("--html", action="store_true", help="HTMLレポートを生成してブラウザで開く")
    p_cmp.add_argument("--test-ratio", dest="test_ratio", type=float, default=0.3,
                       help="テストデータの割合（デフォルト: 0.3=30%%）")

    p_pipe = sub.add_parser("pipeline", help="払戻取得→着順修正→学習→比較を一括実行（済みはスキップ）")
    p_pipe.add_argument("--bankroll", type=float, default=50000)
    p_pipe.add_argument("--html", action="store_true", help="比較HTMLを生成")
    p_pipe.add_argument("--force-payouts", action="store_true", help="払戻データを強制再取得")
    p_pipe.add_argument("--force-train", action="store_true", help="モデルを強制再学習")
    p_pipe.add_argument("--test-ratio", dest="test_ratio", type=float, default=0.3,
                        help="テストデータの割合（デフォルト: 0.3=30%%）")

    sub.add_parser("demo", help="デモ実行")

    p_pred = sub.add_parser("predict", help="シグナル生成（デフォルト: 明日）")
    p_pred.add_argument("--date", type=str, default=None,
                        help="対象日 YYYYMMDD（デフォルト: 明日）")
    p_pred.add_argument("--tomorrow", action="store_true", help="明日を対象にする（デフォルト動作）")
    p_pred.add_argument("--today", action="store_true", help="今日を対象にする")
    p_pred.add_argument("--bankroll", type=float, default=50000,
                        help="資金（Kelly計算用、デフォルト: 50000）")
    p_pred.add_argument("--bet-types", dest="bet_types", type=str, default=None,
                        help=f"賭け式カンマ区切り (デフォルト:全式) 選択肢: {','.join(ALL_BET_TYPES)}")
    p_pred.add_argument("--top", dest="top_n", type=int, default=20,
                        help="おすすめ表示件数（EV上位、デフォルト: 20）")
    p_pred.add_argument("--min-ev", dest="min_ev", type=float, default=1.5,
                        help="最低EV閾値（デフォルト: 1.5）")
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
    elif args.command == "pipeline":
        cmd_pipeline(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
