"""v2 戦略のパラメータを最新データで再校正するスクリプト。

predict_win_probability を1回だけ実行し、その結果に対して多数の
BacktestConfig を試して最適なパラメータ組合せを探索する。

ステップ:
  1. 単軸感度分析: v2 を起点に各パラメータを単独で振って ROI/PnL 変化を見る
  2. 場別分析: 場ごとの ROI/件数 を出し、構造的に弱い場を洗い出す
  3. 上位組合せ探索: 単軸で良かった値を組み合わせて再評価
  4. Walk-forward 検証: 期間を train/test に分けて過学習チェック

使い方:
    python -m scripts.tune_strategy --since 2024-05-01
    python -m scripts.tune_strategy --since 2024-05-01 --until 2026-04-30 --test-days 30
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.predict import predict_win_probability
from src.strategy.backtest import BacktestConfig, run_backtest
from src.utils.config import MODELS_DIR, PROCESSED_DIR, RAW_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 現状の v2 設定（baseline）
V2_BASELINE = dict(
    strategy="lane1_kelly",
    initial_bankroll=100_000.0,
    kelly_fraction=0.5,
    ev_threshold=1.05,
    min_odds=3.0,
    max_odds=10.0,
    odds_shrinkage_max=0.3,
    excluded_venues=("04", "03", "02", "14", "01", "24", "10"),
)


def _read_table(p: Path) -> pd.DataFrame:
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _run(pred: pd.DataFrame, payouts: pd.DataFrame, odds: pd.DataFrame,
         **overrides) -> dict:
    """1組の設定で backtest を走らせて summary を返す。"""
    cfg_kw = {**V2_BASELINE, **overrides}
    cfg = BacktestConfig(**cfg_kw)
    res = run_backtest(pred, payouts, odds_df=odds, config=cfg)
    return res.summary


def _fmt_row(label: str, s: dict) -> str:
    n = s.get("n_bets", 0)
    pnl = s.get("pnl", 0)
    roi = s.get("roi", 0)
    hit = s.get("hit_rate", 0)
    dd = abs(s.get("max_drawdown", 0))
    sign = "+" if pnl > 0 else ""
    return (f"{label:<28} n={n:>5,} ROI={sign}{roi*100:>6.1f}% "
            f"PnL={sign}¥{pnl:>10,.0f} 勝率={hit*100:>5.1f}% "
            f"MaxDD={dd*100:>5.1f}%")


def _sweep(pred, payouts, odds, param: str, values: list, *,
           label_fmt: str = "{}={}") -> list[tuple[Any, dict]]:
    """単一パラメータを振って結果のリストを返す。"""
    results = []
    print(f"\n--- 単軸: {param} ---")
    for v in values:
        summary = _run(pred, payouts, odds, **{param: v})
        results.append((v, summary))
        label = label_fmt.format(param, v)
        print(_fmt_row(label, summary))
    return results


def _per_venue_analysis(pred, payouts, odds):
    """v2 設定で全場対象に走らせた後、場ごとの ROI/件数を集計。"""
    print("\n=== 場別 ROI (v2 baseline, 全場対象) ===")
    cfg_kw = {**V2_BASELINE, "excluded_venues": ()}
    cfg = BacktestConfig(**cfg_kw)
    res = run_backtest(pred, payouts, odds_df=odds, config=cfg)
    if res.bets.empty:
        print("  ベット 0 件")
        return
    bets = res.bets[res.bets["race_finished"]].copy()
    bets["venue"] = bets["race_id"].astype(str).str.split("-").str[1]
    grp = bets.groupby("venue").agg(
        n=("stake", "size"),
        stake=("stake", "sum"),
        pnl=("pnl", "sum"),
        hit=("hit", "sum"),
    )
    grp["roi"] = grp["pnl"] / grp["stake"]
    grp["hit_rate"] = grp["hit"] / grp["n"]
    grp = grp.sort_values("roi")

    print(f"{'場':<6} {'件数':>6} {'勝率':>7} {'ステーク':>11} {'PnL':>12} {'ROI':>9}")
    print("-" * 60)
    for venue, r in grp.iterrows():
        sign = "+" if r["pnl"] > 0 else ""
        print(f"{venue:<6} {int(r['n']):>6,} {r['hit_rate']*100:>6.1f}% "
              f"¥{int(r['stake']):>10,} {sign}¥{int(r['pnl']):>10,} "
              f"{sign}{r['roi']*100:>7.1f}%")

    # 現状除外リスト ＋ ROI<0 で N>=30 の場を新たに「除外候補」として提案
    current_excl = set(V2_BASELINE["excluded_venues"])
    weak = grp[(grp["roi"] < 0) & (grp["n"] >= 30)].index.tolist()
    new_excl = [v for v in weak if v not in current_excl]
    if new_excl:
        print(f"\n  ⚠ 新たな除外候補 (ROI<0 かつ N≥30): {sorted(new_excl)}")
    else:
        print(f"\n  ✓ 現状除外リスト以外で弱い場なし")


def _top_combinations(pred, payouts, odds, top_evs, top_min_odds,
                      top_max_odds, top_kelly):
    """上位値の組合せをグリッド探索。"""
    print(f"\n=== 上位値の組合せ探索 ===")
    print(f"  EV候補:        {top_evs}")
    print(f"  min_odds候補:  {top_min_odds}")
    print(f"  max_odds候補:  {top_max_odds}")
    print(f"  Kelly候補:     {top_kelly}")
    n = len(top_evs) * len(top_min_odds) * len(top_max_odds) * len(top_kelly)
    print(f"  探索数: {n}")

    results = []
    for ev in top_evs:
        for mn in top_min_odds:
            for mx in top_max_odds:
                for kf in top_kelly:
                    if mn >= mx:
                        continue
                    summary = _run(pred, payouts, odds,
                                   ev_threshold=ev, min_odds=mn,
                                   max_odds=mx, kelly_fraction=kf)
                    results.append({
                        "ev": ev, "min_odds": mn, "max_odds": mx,
                        "kelly": kf, **summary,
                    })

    # ROI 上位を表示（最低件数 200 件以上に絞る）
    df = pd.DataFrame(results)
    df = df[df["n_bets"] >= 200]
    df_top = df.sort_values("pnl", ascending=False).head(10)
    print(f"\n--- PnL 上位10 (n_bets ≥ 200) ---")
    print(f"{'ev':>5} {'min':>5} {'max':>5} {'KF':>5} {'件数':>6} "
          f"{'勝率':>7} {'PnL':>12} {'ROI':>9} {'MaxDD':>8}")
    print("-" * 75)
    for _, r in df_top.iterrows():
        sign = "+" if r["pnl"] > 0 else ""
        print(f"{r['ev']:>5.2f} {r['min_odds']:>5.1f} {r['max_odds']:>5.1f} "
              f"{r['kelly']:>5.2f} {int(r['n_bets']):>6,} "
              f"{r['hit_rate']*100:>6.1f}% {sign}¥{int(r['pnl']):>10,} "
              f"{sign}{r['roi']*100:>7.1f}% {abs(r['max_drawdown'])*100:>6.1f}%")
    return df_top


def _walk_forward(pred_full, payouts, odds, best_cfg: dict, test_days: int):
    """ベスト設定で時系列分割 → 過学習チェック。"""
    if "race_date" not in pred_full.columns:
        print("\n  ⚠ race_date 列がないので walk-forward スキップ")
        return
    dates = pd.to_datetime(pred_full["race_date"])
    cutoff = dates.max() - pd.Timedelta(days=test_days)
    train_pred = pred_full[dates <= cutoff]
    test_pred = pred_full[dates > cutoff]

    print(f"\n=== Walk-forward 検証 (test 直近 {test_days}日) ===")
    print(f"  Train期間: {train_pred['race_date'].min()} 〜 {cutoff.date()}")
    print(f"  Test期間:  {(cutoff + pd.Timedelta(days=1)).date()} 〜 {test_pred['race_date'].max()}")

    train_sum = _run(train_pred, payouts, odds, **best_cfg)
    test_sum = _run(test_pred, payouts, odds, **best_cfg)

    print(_fmt_row("  Train:", train_sum))
    print(_fmt_row("  Test: ", test_sum))

    train_roi = train_sum.get("roi", 0)
    test_roi = test_sum.get("roi", 0)
    if test_sum.get("n_bets", 0) < 20:
        print(f"  ⚠ Test件数が少ない({test_sum['n_bets']}件) → 過学習判断は弱い")
    elif test_roi < train_roi * 0.5:
        print(f"  ⚠ Test ROI が Train の半分未満 → 過学習の疑い")
    elif test_roi > 0:
        print(f"  ✓ Test も黒字 → 汎化している可能性高")
    else:
        print(f"  △ Test 赤字 → 慎重に")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default=str(PROCESSED_DIR / "features.parquet"))
    p.add_argument("--payouts", default=str(RAW_DIR / "races_payouts.parquet"))
    p.add_argument("--odds", default=str(RAW_DIR / "odds_win.csv"))
    p.add_argument("--since", default=None)
    p.add_argument("--until", default=None)
    p.add_argument("--test-days", type=int, default=60,
                   help="Walk-forward 検証で test に切り出す直近日数")
    p.add_argument("--skip-venue", action="store_true",
                   help="場別分析をスキップ（時間短縮）")
    p.add_argument("--skip-combo", action="store_true",
                   help="組合せ探索をスキップ")
    p.add_argument("--out", default=str(MODELS_DIR / "tune_strategy"))
    args = p.parse_args()

    print(f"=== データ読込 ===")
    features = _read_table(Path(args.features))
    payouts = _read_table(Path(args.payouts))
    odds = _read_table(Path(args.odds))
    if args.since and "race_date" in features.columns:
        features = features[pd.to_datetime(features["race_date"]) >= pd.Timestamp(args.since)]
    if args.until and "race_date" in features.columns:
        features = features[pd.to_datetime(features["race_date"]) <= pd.Timestamp(args.until)]
    print(f"  features: {len(features):,} rows")
    print(f"  payouts:  {len(payouts):,} rows")
    print(f"  odds:     {len(odds):,} rows")

    print(f"\n=== モデル推論 (1回のみ) ===")
    pred = predict_win_probability(features)
    keep = ["race_id", "lane", "race_date", "pred_win_prob"]
    pred = pred[[c for c in keep if c in pred.columns]]
    print(f"  pred: {len(pred):,} rows")

    print(f"\n=== Baseline (v2 現状設定) ===")
    base = _run(pred, payouts, odds)
    print(_fmt_row("v2 baseline", base))

    print(f"\n=== 1. 単軸感度分析 ===")
    ev_results = _sweep(pred, payouts, odds, "ev_threshold",
                        [1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.50])
    min_results = _sweep(pred, payouts, odds, "min_odds",
                         [2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
    max_results = _sweep(pred, payouts, odds, "max_odds",
                         [6.0, 8.0, 10.0, 12.0, 15.0])
    kelly_results = _sweep(pred, payouts, odds, "kelly_fraction",
                           [0.10, 0.25, 0.5, 0.75, 1.0])
    shrink_results = _sweep(pred, payouts, odds, "odds_shrinkage_max",
                            [0.0, 0.15, 0.3, 0.5])

    if not args.skip_venue:
        print(f"\n=== 2. 場別 ROI ===")
        _per_venue_analysis(pred, payouts, odds)

    if not args.skip_combo:
        # 単軸 PnL 上位3つを組合せ候補に
        def _top3(results, default):
            sub = sorted(results, key=lambda x: x[1].get("pnl", 0), reverse=True)[:3]
            return [v for v, _ in sub] or [default]

        top_df = _top_combinations(
            pred, payouts, odds,
            top_evs=_top3(ev_results, 1.05),
            top_min_odds=_top3(min_results, 3.0),
            top_max_odds=_top3(max_results, 10.0),
            top_kelly=_top3(kelly_results, 0.5),
        )

        if len(top_df):
            best = top_df.iloc[0]
            best_cfg = {
                "ev_threshold": float(best["ev"]),
                "min_odds": float(best["min_odds"]),
                "max_odds": float(best["max_odds"]),
                "kelly_fraction": float(best["kelly"]),
            }
            print(f"\n=== ベスト設定 ===")
            print(f"  {best_cfg}")
            print(f"  → run_signal_v2 に反映する場合:")
            print(f"    --ev-threshold {best_cfg['ev_threshold']}")
            print(f"    --min-odds {best_cfg['min_odds']}")
            print(f"    --max-odds {best_cfg['max_odds']}")
            print(f"    --kelly-fraction {best_cfg['kelly_fraction']}")

            _walk_forward(pred, payouts, odds, best_cfg, args.test_days)

    # 結果保存
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "tune_summary.json"
    summary_path.write_text(json.dumps({
        "baseline": base,
        "sweep_ev": {str(v): s for v, s in ev_results},
        "sweep_min_odds": {str(v): s for v, s in min_results},
        "sweep_max_odds": {str(v): s for v, s in max_results},
        "sweep_kelly": {str(v): s for v, s in kelly_results},
        "sweep_shrinkage": {str(v): s for v, s in shrink_results},
    }, indent=2, ensure_ascii=False, default=str))
    print(f"\n結果保存: {summary_path}")


if __name__ == "__main__":
    main()
