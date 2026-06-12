"""過去ベットで「シグナルが締切まで持続したか」を確定オッズで判定し成績比較。

データの制約:
  - 的中ベット: payout/100 = settled_odds → settled_EV を計算可能 → 持続/消失分類
  - 外れベット: 1号艇敗北 → 1号艇の確定オッズが payouts に無い → 持続/消失不明

そのため「持続のみ買う戦略」のシミュレーションは仮定込みになる:
  - 仮定A (独立): 持続率は的中/外れで同じ → 外れも的中の持続率で按分
  - 仮定B (悲観): 外れは全て持続扱い (= 持続戦略でも回避できない)
  - 仮定C (楽観): 外れは全て消失扱い (= 持続戦略で全て回避できる)

仮定Cがありえない一方、AとBの間に真の値があると考えられる。

使い方:
    python -m scripts.analyze_signal_persistence
    python -m scripts.analyze_signal_persistence --bets data/models/backtest/history/bets_lane1_kelly.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import MODELS_DIR, PROCESSED_DIR


def _load_bets(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"race_id", "stake", "pnl", "hit", "odds_win",
              "blended_win_prob"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"必須列が不足: {missing}")
    if "settled_odds" not in df.columns:
        raise SystemExit("settled_odds 列がありません。最新のbacktest結果を使ってください")
    # bool化
    df["hit"] = df["hit"].astype(bool)
    return df


def _classify_persistence(bets: pd.DataFrame, ev_threshold: float) -> pd.DataFrame:
    """的中ベットに settled_EV を計算し persistence ラベルを付与。"""
    df = bets.copy()
    df["settled_ev"] = df["blended_win_prob"] * df["settled_odds"]
    # 的中ベットだけ確定オッズ判定可能。外れは "unknown"
    def _label(r):
        if not r["hit"]:
            return "unknown_loss"  # 外れ: 持続不明
        if pd.isna(r["settled_ev"]):
            return "unknown_loss"
        return "persisted" if r["settled_ev"] > ev_threshold else "vanished"
    df["persistence"] = df.apply(_label, axis=1)
    df["odds_ratio"] = df["settled_odds"] / df["odds_win"]
    return df


def _print_basic_stats(df: pd.DataFrame, ev_threshold: float) -> None:
    print(f"\n=== 基本統計 ===")
    n = len(df)
    n_hit = int(df["hit"].sum())
    n_miss = n - n_hit
    print(f"  ベット数: {n:,}")
    print(f"  的中:    {n_hit:,} ({n_hit/n*100:.1f}%)")
    print(f"  外れ:    {n_miss:,} ({n_miss/n*100:.1f}%)")

    hits = df[df["hit"]]
    if hits.empty:
        return
    n_persist = int((hits["persistence"] == "persisted").sum())
    n_vanish = int((hits["persistence"] == "vanished").sum())
    print(f"\n  的中の内訳 (settled_EV > {ev_threshold} で持続判定):")
    print(f"    持続 (settled_EV > {ev_threshold}): {n_persist:,} ({n_persist/n_hit*100:.1f}%)")
    print(f"    消失 (settled_EV ≤ {ev_threshold}): {n_vanish:,} ({n_vanish/n_hit*100:.1f}%)")

    # オッズ収縮率の分布
    rh = hits["odds_ratio"].dropna()
    print(f"\n  的中時オッズ収縮率の分布:")
    print(f"    settled/scraped 平均: {rh.mean():.3f} (1.0=変化なし)")
    print(f"    50%以上収縮した的中: {int((rh < 0.5).sum())}件 ({(rh < 0.5).mean()*100:.1f}%)")
    print(f"    30%以上収縮した的中: {int((rh < 0.7).sum())}件 ({(rh < 0.7).mean()*100:.1f}%)")


def _print_pnl_by_group(df: pd.DataFrame) -> None:
    print(f"\n=== 持続/消失グループ別の PnL ===")
    hits = df[df["hit"]]
    if hits.empty:
        return
    print(f"\n{'群':<25}{'件数':>6}{'ステーク':>11}{'PnL':>13}{'平均PnL':>10}")
    print("-" * 70)
    for label, sub in [
        ("持続的中 (持続signal)", hits[hits["persistence"] == "persisted"]),
        ("消失的中 (vanished)", hits[hits["persistence"] == "vanished"]),
    ]:
        if sub.empty:
            print(f"{label:<25}{0:>6}")
            continue
        st = int(sub["stake"].sum())
        pn = int(sub["pnl"].sum())
        avg = int(sub["pnl"].mean())
        print(f"{label:<25}{len(sub):>6,}¥{st:>10,}+¥{pn:>11,}+¥{avg:>8,}")

    misses = df[~df["hit"]]
    if not misses.empty:
        st = int(misses["stake"].sum())
        pn = int(misses["pnl"].sum())
        avg = int(misses["pnl"].mean())
        print(f"{'外れ (持続不明)':<25}{len(misses):>6,}¥{st:>10,}¥{pn:>11,}¥{avg:>8,}")


def _simulate_persisted_only_strategy(df: pd.DataFrame) -> None:
    """『持続のみ買う戦略』を3つの仮定で評価。"""
    print(f"\n=== 「持続のみ買う」戦略のシミュレーション ===")

    total_stake = float(df["stake"].sum())
    total_pnl = float(df["pnl"].sum())
    total_roi = total_pnl / total_stake if total_stake else 0
    print(f"\n  全シグナル (現状=即発注): n={len(df):,}, "
          f"PnL=+¥{int(total_pnl):,}, ROI={total_roi*100:.1f}%")

    hits = df[df["hit"]]
    misses = df[~df["hit"]]
    n_hit = len(hits)
    n_miss = len(misses)
    persist_hits = hits[hits["persistence"] == "persisted"]
    vanish_hits = hits[hits["persistence"] == "vanished"]
    persist_hit_rate_among_hits = len(persist_hits) / n_hit if n_hit else 0

    print(f"\n  仮定A (独立: 持続率は的中/外れで同じ {persist_hit_rate_among_hits*100:.1f}%):")
    # 外れの (1-持続率) 分は消失なので回避できる
    est_persist_misses = int(n_miss * persist_hit_rate_among_hits)
    est_vanish_misses = n_miss - est_persist_misses
    a_stake = float(persist_hits["stake"].sum()) + float(misses["stake"].sum()) * persist_hit_rate_among_hits
    a_pnl = float(persist_hits["pnl"].sum()) + float(misses["pnl"].sum()) * persist_hit_rate_among_hits
    a_roi = a_pnl / a_stake if a_stake else 0
    print(f"    保持: {len(persist_hits) + est_persist_misses:,}件 "
          f"(持続的中 {len(persist_hits)} + 推定持続外れ {est_persist_misses})")
    print(f"    PnL: +¥{int(a_pnl):,}  ROI: {a_roi*100:.1f}%  "
          f"(全シグナル比: {(a_roi-total_roi)*100:+.1f}pt)")

    print(f"\n  仮定B (悲観: 外れは全て持続=回避不能):")
    b_stake = float(persist_hits["stake"].sum()) + float(misses["stake"].sum())
    b_pnl = float(persist_hits["pnl"].sum()) + float(misses["pnl"].sum())
    b_roi = b_pnl / b_stake if b_stake else 0
    print(f"    保持: {len(persist_hits) + n_miss:,}件 (持続的中 + 外れ全部)")
    print(f"    PnL: +¥{int(b_pnl):,}  ROI: {b_roi*100:.1f}%  "
          f"(全シグナル比: {(b_roi-total_roi)*100:+.1f}pt)")

    print(f"\n  仮定C (楽観: 外れは全て消失=全て回避できる):")
    c_stake = float(persist_hits["stake"].sum())
    c_pnl = float(persist_hits["pnl"].sum())
    c_roi = c_pnl / c_stake if c_stake else 0
    print(f"    保持: {len(persist_hits):,}件 (持続的中のみ)")
    print(f"    PnL: +¥{int(c_pnl):,}  ROI: {c_roi*100:.1f}%  "
          f"(全シグナル比: {(c_roi-total_roi)*100:+.1f}pt)")

    print(f"\n  解釈:")
    print(f"    真の「持続のみ」戦略の成績は A 〜 B の間 (Cはあり得ない上限)")
    if a_roi > total_roi:
        print(f"    → A (独立仮定) で改善 → 持続フィルタは効く可能性")
    else:
        print(f"    → A (独立仮定) で改善せず → 持続フィルタの利益限定的")


def _odds_shrinkage_vs_hit_rate(df: pd.DataFrame) -> None:
    """的中ベットだけだが、収縮率と的中の関係を見る。"""
    hits = df[df["hit"]]
    if hits.empty:
        return
    print(f"\n=== 的中時の収縮率分布（参考） ===")
    bins = [(0, 0.5, "≥50%収縮"),
            (0.5, 0.7, "30-50%収縮"),
            (0.7, 0.9, "10-30%収縮"),
            (0.9, 1.1, "ほぼ変化なし"),
            (1.1, 2.0, "上昇")]
    print(f"\n{'収縮帯':<15}{'件数':>7}{'平均PnL':>12}")
    for lo, hi, label in bins:
        sub = hits[(hits["odds_ratio"] >= lo) & (hits["odds_ratio"] < hi)]
        if sub.empty:
            continue
        avg = int(sub["pnl"].mean())
        print(f"{label:<15}{len(sub):>7}+¥{avg:>10,}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bets", default=None,
                   help="bets_lane1_kelly.csv のパス。デフォルトは backtest_history を試す")
    p.add_argument("--ev-threshold", type=float, default=1.05,
                   help="持続判定のEV閾値（バックテストと同じ値推奨）")
    args = p.parse_args()

    candidates = [
        Path(args.bets) if args.bets else None,
        PROCESSED_DIR / "backtest_history" / "bets_lane1_kelly.csv",
        MODELS_DIR / "backtest" / "bets_lane1_kelly.csv",
        MODELS_DIR / "tune_strategy" / "bets_lane1_kelly.csv",
    ]
    bets_path = next((p for p in candidates if p and p.exists()), None)
    if bets_path is None:
        print("bets_lane1_kelly.csv が見つかりません。")
        print("先に2年バックテストを実行してください:")
        print("  python -m scripts.run_signal_v2 --autofill --backtest-days 730")
        return

    print(f"=== 入力: {bets_path} ===")
    bets = _load_bets(bets_path)
    print(f"  ベット数: {len(bets):,}")

    bets_clf = _classify_persistence(bets, args.ev_threshold)
    _print_basic_stats(bets_clf, args.ev_threshold)
    _print_pnl_by_group(bets_clf)
    _simulate_persisted_only_strategy(bets_clf)
    _odds_shrinkage_vs_hit_rate(bets_clf)


if __name__ == "__main__":
    main()
