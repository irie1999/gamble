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
    collect_data, save_records, load_existing_records, merge_records, VENUE_CODES,
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
from betting import simulate_session, print_session_report, pick_bets, STRATEGIES

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
    if test_days < 14:
        print(f"⚠ テスト期間が短すぎます（{test_days}日）。ROIは統計的に無意味です。最低14日、推奨30日以上のデータが必要です。")
    elif test_days < 30:
        print(f"⚠ テスト期間が少ない（{test_days}日）。ROIの信頼性は低め。30日以上推奨。")
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

        # keirin.jp ライブオッズを参考取得（選択には使わない、表示用）
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

        if not odds_dict:
            odds_dict = fetch_race_odds(race, bet_types=["trifecta", "trio", "wide"])

        pred_sorted = pred_df.sort_values("win_prob", ascending=False)
        pred_str = " > ".join(
            f"{int(r['car_no'])}番({r['win_prob']:.0%})"
            for _, r in pred_sorted.iterrows()
        )

        for bt in bet_types:
            bets = pick_bets(pred_df, bt, fixed_amount=100)
            for b in bets:
                # 参考用: ライブオッズがあれば表示に使う
                live_odds = odds_dict.get(bt, {}).get(b.selections, 0.0)
                signal_rows.append({
                    "venue": race["venue_name"],
                    "race_no": race["race_no"],
                    "bet_type": BET_TYPE_NAMES.get(b.bet_type, b.bet_type),
                    "selections": str(list(b.selections)),
                    "odds": live_odds,
                    "bet_amount": b.bet_amount,
                    "edge": 0.0,
                    "ev": live_odds * b.predicted_prob if live_odds > 0 else b.predicted_prob,
                    "pred_prob": b.predicted_prob,
                    "pred_str": pred_str,
                })

    if not signal_rows:
        print("シグナルがありません")
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
    print(f"全シグナル: {len(signal_rows)}件")
    print("\n⚠ オッズはライブ参考値です（取得できない場合は0）。実際のオッズを確認してから購入してください。\n")

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

    existing_records, _ = load_existing_records("raw_data.json")
    start_date = (today - timedelta(days=args.days)).strftime("%Y%m%d")

    # 指定期間のうち「まだ取得していない日付」だけ収集してマージ
    if existing_records:
        existing_dates = sorted({r.get("date") for r in existing_records if r.get("date")})
        print(f"既存データ: {len(existing_records)}件  期間: {existing_dates[0]}〜{existing_dates[-1]}")
    else:
        print("既存データなし → 新規収集")

    print(f"収集対象期間: {start_date} → {end_date}（{args.days}日分）")
    print(f"対象場: {venue_codes or '全場'}")
    print()

    new_records = collect_data(
        start_date, end_date,
        venue_codes=venue_codes,
        sleep_sec=args.sleep,
        existing_records=existing_records,
        checkpoint_days=7,
        filename="raw_data.json",
        workers=args.workers,
        skip_existing_dates=True,  # 取得済みの日付はスキップ
    )

    # 既存データと新規データをマージして保存
    if existing_records:
        merged = merge_records(existing_records, new_records)
        save_records(merged, "raw_data.json")
        new_count = len(merged) - len(existing_records)
        print(f"マージ完了: 既存{len(existing_records)}件 + 新規{new_count}件 → {len(merged)}件")


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

    if getattr(args, "skip_payouts", False):
        print("  ✓ --skip-payouts 指定 → スキップ")
    else:
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


def _str_keys_to_tuples(d: dict) -> dict:
    """JSON文字列キー('4_1_6')をタプル(4,1,6)に変換"""
    result = {}
    for k, v in d.items():
        if isinstance(k, str):
            sep = "_" if "_" in k else ","
            result[tuple(int(x) for x in k.split(sep))] = v
        else:
            result[k] = v
    return result


def _build_races(booster, feature_cols, df_feat, bet_types, odds_data: dict | None = None):
    """バックテスト用レースリストを構築

    ベット選択: モデル予測確率の上位組み合わせ（市場オッズ不要）
    払戻計算:  odds_data の実際の払戻額（当選時のみ存在）
    着順判定:  rank列（fix_ranks.py適用済み）
    """
    if odds_data is None:
        odds_data = load_odds_data()
    odds_by_race_id = odds_data if odds_data else {}

    races = []
    for (date, venue, rno), race_df in df_feat.groupby(["date", "venue_code", "race_no"]):
        race_df = race_df.drop_duplicates(subset="car_no", keep="first")

        winner_row = race_df[race_df["win"] == 1]
        if winner_row.empty:
            continue

        # rank列から正しい着順を構築（fix_ranks.py適用済みを前提）
        if "rank" in race_df.columns and race_df["rank"].notna().any():
            sorted_df = race_df.sort_values("rank", na_position="last")
            finish = [int(c) for c in sorted_df["car_no"].tolist()]
        else:
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

        race_id = race_df["race_id"].iloc[0] if "race_id" in race_df.columns else None

        # 実際の払戻額（当選時のみ存在）
        actual_payouts: dict = {}
        od = None
        if race_id and race_id in odds_by_race_id:
            od = odds_by_race_id[race_id]
        elif odds_by_race_id:
            for rid, entry in odds_by_race_id.items():
                if rid[2:10] == str(date) and int(rid[12:16]) == rno:
                    od = entry
                    break
        if od:
            for bt, d in od.items():
                if bt in KEIRIN_BET_TYPES:
                    actual_payouts[bt] = _str_keys_to_tuples(d)

        races.append({
            "race_id": race_id or f"{date}_{venue}_{rno}",
            "pred_df": pred_df,
            "payouts": actual_payouts,
            "finish_order": finish,
        })

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

        races.append({
            "race_id": f"{date}_{rno}",
            "pred_df": pred_df_race,
            "payouts": {},
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


def cmd_detail(args):
    """指定戦略の全ベット明細をHTMLで出力"""
    model_path = MODEL_DIR / "lgb_model.txt"
    if not model_path.exists():
        print("モデルがありません。先に `python main.py train` を実行してください")
        return

    strategy_name = args.strategy
    strat = STRATEGIES.get(strategy_name)
    if strat is None:
        print(f"不明な戦略: {strategy_name}  選択肢: {', '.join(STRATEGIES.keys())}")
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

    races = _build_races(booster, feature_cols, df_test, KEIRIN_BET_TYPES)
    session = simulate_session(races, initial_bankroll=args.bankroll, strategy=strat)

    print(f"戦略: {strategy_name}  期間: {test_days}日  "
          f"ROI: {session.roi:+.1%}  ベット数: {len(session.bets)}")

    _save_detail_html(session, strategy_name, strat["description"], mean_auc, test_days, args.bankroll)


def _save_detail_html(session, strategy_name: str, description: str, auc: float, test_days: int, bankroll: float) -> None:
    """全ベット明細 + 累積損益グラフをHTMLで保存"""
    # 累積損益データ（ベット順）
    cumulative = []
    running = 0.0
    for b in session.bets:
        if b.win_flag:
            running += b.bet_amount * b.return_odds - b.bet_amount
        else:
            running -= b.bet_amount
        cumulative.append(running)

    cumulative_json = json.dumps(cumulative)

    # ベット明細テーブル行
    rows_html = ""
    for i, b in enumerate(session.bets):
        sel_str = "-".join(str(s) for s in b.selections)
        bt_name = BET_TYPE_NAMES.get(b.bet_type, b.bet_type)
        win_cls = "win" if b.win_flag else "lose"
        win_str = f"◎ {b.return_odds:.1f}倍" if b.win_flag else "✗"
        payout = b.bet_amount * b.return_odds if b.win_flag else 0
        net = payout - b.bet_amount
        net_str = f"{net:+,.0f}"
        net_cls = "pos" if net > 0 else "neg"
        rows_html += (
            f'<tr class="{win_cls}">'
            f'<td>{i+1}</td>'
            f'<td class="mono">{b.race_id}</td>'
            f'<td>{bt_name}</td>'
            f'<td class="sel">{sel_str}</td>'
            f'<td>{b.predicted_prob:.1%}</td>'
            f'<td>{b.bet_amount:,}円</td>'
            f'<td>{win_str}</td>'
            f'<td>{payout:,.0f}円</td>'
            f'<td class="{net_cls}">{net_str}円</td>'
            f'<td class="{net_cls}">{cumulative[i]:+,.0f}円</td>'
            f'</tr>\n'
        )

    total_bets = len(session.bets)
    win_rate = session.wins / total_bets if total_bets > 0 else 0
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>詳細バックテスト: {strategy_name}</title>
<style>
  body {{ font-family: 'Meiryo', sans-serif; background:#0f172a; color:#e2e8f0; margin:0; padding:20px; }}
  h1 {{ color:#f8fafc; font-size:1.4rem; margin-bottom:4px; }}
  .meta {{ color:#94a3b8; font-size:.85rem; margin-bottom:20px; }}
  .cards {{ display:flex; gap:14px; flex-wrap:wrap; margin-bottom:24px; }}
  .card {{ background:#1e293b; border-radius:10px; padding:14px 20px; min-width:130px; }}
  .card .label {{ font-size:.75rem; color:#94a3b8; margin-bottom:4px; }}
  .card .value {{ font-size:1.5rem; font-weight:700; }}
  .pos {{ color:#4ade80; }} .neg {{ color:#f87171; }}
  canvas {{ background:#1e293b; border-radius:10px; margin-bottom:24px; width:100%; max-height:260px; }}
  table {{ width:100%; border-collapse:collapse; font-size:.82rem; }}
  th {{ background:#1e293b; color:#94a3b8; padding:8px 10px; text-align:left; position:sticky; top:0; }}
  td {{ padding:6px 10px; border-bottom:1px solid #1e293b; }}
  tr.win {{ background:#14532d22; }}
  tr.lose {{ background:#1e293b44; }}
  tr:hover {{ background:#334155; }}
  .mono {{ font-family:monospace; font-size:.78rem; color:#94a3b8; }}
  .sel {{ font-weight:700; letter-spacing:1px; }}
  .search {{ margin-bottom:12px; }}
  .search input {{ background:#1e293b; border:1px solid #334155; color:#e2e8f0; padding:6px 12px; border-radius:6px; width:260px; }}
  #pagination {{ margin-top:10px; color:#94a3b8; font-size:.82rem; }}
  .page-btn {{ background:#1e293b; border:1px solid #334155; color:#e2e8f0; padding:4px 10px; border-radius:4px; cursor:pointer; margin:0 2px; }}
  .page-btn.active {{ background:#3b82f6; border-color:#3b82f6; }}
</style>
</head>
<body>
<h1>詳細バックテスト: {strategy_name}</h1>
<div class="meta">{description}　／　生成: {now_str}　／　モデルAUC: {auc:.4f}　／　テスト: {test_days}日</div>

<div class="cards">
  <div class="card"><div class="label">ROI</div><div class="value {'pos' if session.roi >= 0 else 'neg'}">{session.roi:+.1%}</div></div>
  <div class="card"><div class="label">損益</div><div class="value {'pos' if session.profit >= 0 else 'neg'}">{session.profit:+,.0f}円</div></div>
  <div class="card"><div class="label">最終資金</div><div class="value">{session.final_bankroll:,.0f}円</div></div>
  <div class="card"><div class="label">ベット数</div><div class="value">{total_bets:,}回</div></div>
  <div class="card"><div class="label">的中数</div><div class="value pos">{session.wins}回</div></div>
  <div class="card"><div class="label">的中率</div><div class="value">{win_rate:.1%}</div></div>
  <div class="card"><div class="label">総賭け金</div><div class="value">{session.total_bet:,.0f}円</div></div>
  <div class="card"><div class="label">初期資金</div><div class="value">{bankroll:,.0f}円</div></div>
</div>

<canvas id="chart"></canvas>

<div class="search">
  <input id="filter" type="text" placeholder="レースIDや選択車番で絞り込み..." oninput="applyFilter()">
  <span id="count" style="margin-left:10px;color:#94a3b8;font-size:.82rem;"></span>
</div>

<table>
<thead>
  <tr>
    <th>#</th><th>レースID</th><th>賭け式</th><th>選択</th>
    <th>予測確率</th><th>賭け金</th><th>結果</th><th>払戻</th><th>収支</th><th>累積</th>
  </tr>
</thead>
<tbody id="tbody">
{rows_html}
</tbody>
</table>
<div id="pagination"></div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<script>
const allRows = Array.from(document.querySelectorAll('#tbody tr'));
const cumData = {cumulative_json};
let filtered = allRows;
const PAGE = 200;
let page = 0;

function applyFilter() {{
  const q = document.getElementById('filter').value.toLowerCase();
  filtered = q ? allRows.filter(r => r.innerText.toLowerCase().includes(q)) : allRows;
  page = 0;
  render();
}}

function render() {{
  const start = page * PAGE, end = start + PAGE;
  allRows.forEach(r => r.style.display = 'none');
  filtered.slice(start, end).forEach(r => r.style.display = '');
  document.getElementById('count').textContent = filtered.length + '件';
  const pages = Math.ceil(filtered.length / PAGE);
  let p = '<span style="margin-right:6px">ページ:</span>';
  for (let i = 0; i < pages; i++) {{
    p += `<button class="page-btn ${{i===page?'active':''}}" onclick="goPage(${{i}})">${{i+1}}</button>`;
  }}
  document.getElementById('pagination').innerHTML = p;
}}

function goPage(n) {{ page = n; render(); }}

// Chart
const ctx = document.getElementById('chart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels: cumData.map((_, i) => i + 1),
    datasets: [{{
      label: '累積損益（円）',
      data: cumData,
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59,130,246,0.08)',
      borderWidth: 1.5,
      pointRadius: 0,
      fill: true,
      tension: 0.1,
    }}, {{
      label: '±0ライン',
      data: cumData.map(() => 0),
      borderColor: '#475569',
      borderWidth: 1,
      pointRadius: 0,
      borderDash: [4, 4],
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }},
    scales: {{
      x: {{ display: false }},
      y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }}
    }}
  }}
}});

render();
</script>
</body>
</html>"""

    out_path = RESULTS_DIR / f"detail_{strategy_name}.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nHTMLを保存しました: {out_path}")
    import webbrowser
    webbrowser.open(out_path.as_uri())


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
    p_pipe.add_argument("--skip-payouts", dest="skip_payouts", action="store_true",
                        help="Step1払戻取得をスキップ（学習・比較のみ実行したい場合）")
    p_pipe.add_argument("--force-train", action="store_true", help="モデルを強制再学習")
    p_pipe.add_argument("--test-ratio", dest="test_ratio", type=float, default=0.3,
                        help="テストデータの割合（デフォルト: 0.3=30%%）")

    sub.add_parser("demo", help="デモ実行")

    p_det = sub.add_parser("detail", help="指定戦略の全ベット明細をHTMLで出力")
    p_det.add_argument("--strategy", type=str, default="trifecta_mid",
                       help=f"戦略名（デフォルト: trifecta_mid）: {', '.join(STRATEGIES.keys())}")
    p_det.add_argument("--bankroll", type=float, default=50000)
    p_det.add_argument("--test-ratio", dest="test_ratio", type=float, default=0.3,
                       help="テストデータの割合（デフォルト: 0.3）")

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
    elif args.command == "detail":
        cmd_detail(args)
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
