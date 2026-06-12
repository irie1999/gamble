"""1号艇シグナルの不振原因を診断: 直近の艇別1着率を歴史と比較。

直近30日で 2号艇1着率が異常に高い → 構造変化の可能性
特定の場/オッズ帯に偏ってる → 該当だけ除外で対処可

使い方:
    python -m scripts.diagnose_lane_shift
    python -m scripts.diagnose_lane_shift --recent-days 30 --baseline-days 730
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.utils.config import RAW_DIR, VENUE_CODES


def _load_winners(payouts_path) -> pd.DataFrame:
    """payouts から各レースの 1着艇 (winner_lane) を抽出。"""
    df = pd.read_parquet(payouts_path)
    win_df = df[df["bet_type"] == "win"].copy()
    # combo は「1」「2」など単一数字（trifecta は "1-2-3" だが win では単艇）
    win_df["winner_lane"] = pd.to_numeric(win_df["combo"], errors="coerce")
    win_df = win_df.dropna(subset=["winner_lane"])
    win_df["winner_lane"] = win_df["winner_lane"].astype(int)
    win_df["race_date"] = pd.to_datetime(
        win_df["race_id"].astype(str).str[:8], format="%Y%m%d", errors="coerce"
    )
    win_df["venue"] = win_df["race_id"].astype(str).str.split("-").str[1]
    return win_df[["race_id", "winner_lane", "race_date", "venue"]]


def _load_lane1_odds(odds_path) -> pd.DataFrame:
    df = pd.read_csv(odds_path)
    return df[df["lane"] == 1][["race_id", "odds_win"]].rename(
        columns={"odds_win": "lane1_odds"}
    )


def _lane_distribution(df: pd.DataFrame, label: str) -> dict:
    """艇別1着率の分布を計算。"""
    n = len(df)
    dist = df["winner_lane"].value_counts().sort_index()
    res = {"label": label, "n": n}
    for lane in range(1, 7):
        count = int(dist.get(lane, 0))
        rate = count / n if n else 0
        res[f"lane{lane}"] = (count, rate)
    return res


def _print_lane_table(rows: list[dict]) -> None:
    print(f"\n{'期間':<22}{'N':>7}  " + "  ".join(f"L{i}" for i in range(1, 7)))
    print("-" * 70)
    for r in rows:
        cells = "  ".join(f"{r[f'lane{i}'][1]*100:>5.1f}%" for i in range(1, 7))
        print(f"{r['label']:<22}{r['n']:>7,}  {cells}")


def _binomial_pvalue(observed: int, n: int, p0: float) -> float:
    """片側 二項検定の p値 近似（正規近似）。observed が p0 より大きい側の検定。"""
    if n == 0:
        return 1.0
    mu = n * p0
    sigma = (n * p0 * (1 - p0)) ** 0.5
    if sigma == 0:
        return 1.0
    z = (observed - mu) / sigma
    # 簡易: 正規分布右側
    from math import erfc, sqrt
    return float(0.5 * erfc(z / sqrt(2)))


def _per_venue_recent_vs_baseline(recent: pd.DataFrame, baseline: pd.DataFrame,
                                  min_recent: int = 20) -> None:
    """場ごとに直近の2号艇1着率を比較。"""
    print(f"\n=== 場別 2号艇1着率 (直近 vs ベースライン) ===")
    print(f"対象: 直近データで N≥{min_recent}件 の場のみ")

    rows = []
    for venue in sorted(recent["venue"].unique()):
        r = recent[recent["venue"] == venue]
        if len(r) < min_recent:
            continue
        b = baseline[baseline["venue"] == venue]
        if b.empty:
            continue
        r_rate = (r["winner_lane"] == 2).mean()
        b_rate = (b["winner_lane"] == 2).mean()
        diff = r_rate - b_rate
        p = _binomial_pvalue(int((r["winner_lane"] == 2).sum()), len(r), b_rate)
        rows.append({
            "venue": venue,
            "name": VENUE_CODES.get(venue, "?"),
            "n_recent": len(r),
            "r_rate": r_rate,
            "b_rate": b_rate,
            "diff": diff,
            "p": p,
        })

    # diff 降順
    rows.sort(key=lambda x: -x["diff"])
    print(f"\n{'場':<10} {'N直近':>6} {'直近':>7} {'ベース':>7} {'差分':>8} {'p値':>8}  判定")
    print("-" * 65)
    for r in rows:
        flag = "⚠ 異常" if r["p"] < 0.05 and r["diff"] > 0 else ""
        if r["p"] < 0.01 and r["diff"] > 0:
            flag = "🚨 強い異常"
        print(f"{r['name']:<8}({r['venue']}) {r['n_recent']:>6,} "
              f"{r['r_rate']*100:>6.1f}% {r['b_rate']*100:>6.1f}% "
              f"{r['diff']*100:>+6.1f}pt {r['p']:>7.3f}  {flag}")


def _monthly_trend(df: pd.DataFrame) -> None:
    """月別の 2号艇1着率の推移。"""
    print(f"\n=== 月別 2号艇1着率の推移 ===")
    df = df.copy()
    df["month"] = df["race_date"].dt.to_period("M")
    grp = df.groupby("month").agg(
        n=("winner_lane", "size"),
        l2=("winner_lane", lambda s: (s == 2).sum()),
    )
    grp["rate"] = grp["l2"] / grp["n"]
    grp = grp.sort_index()
    print(f"\n{'月':<10}{'N':>7}{'2号艇1着率':>11}")
    print("-" * 32)
    for m, r in grp.iterrows():
        bar_len = int(r["rate"] * 100)
        bar = "█" * bar_len
        print(f"{str(m):<10}{int(r['n']):>7,}{r['rate']*100:>10.1f}% {bar}")


def _bet_zone_focus(df: pd.DataFrame, label: str,
                    lane1_odds_min: float = 3.0,
                    lane1_odds_max: float = 10.0) -> None:
    """v2 のベット対象帯 (1号艇オッズ 3-10倍) に絞った艇別1着率。"""
    sub = df[(df["lane1_odds"] >= lane1_odds_min)
             & (df["lane1_odds"] <= lane1_odds_max)]
    if sub.empty:
        return
    print(f"\n--- {label}: 1号艇オッズ [{lane1_odds_min}, {lane1_odds_max}] N={len(sub):,} ---")
    rates = sub["winner_lane"].value_counts(normalize=True).sort_index()
    for lane in range(1, 7):
        rate = rates.get(lane, 0) * 100
        bar = "█" * int(rate / 2)
        print(f"  {lane}号艇: {rate:>5.1f}%  {bar}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--payouts", default=str(RAW_DIR / "races_payouts.parquet"))
    p.add_argument("--odds", default=str(RAW_DIR / "odds_win.csv"))
    p.add_argument("--recent-days", type=int, default=30,
                   help="「直近」とする日数")
    p.add_argument("--baseline-days", type=int, default=730,
                   help="ベースラインの日数（最大期間）")
    args = p.parse_args()

    print("=== データ読込 ===")
    winners = _load_winners(args.payouts)
    lane1 = _load_lane1_odds(args.odds)
    df = winners.merge(lane1, on="race_id", how="left")
    print(f"  全期間: {len(df):,} レース ({df['race_date'].min().date()} 〜 {df['race_date'].max().date()})")

    end = df["race_date"].max()
    recent_cutoff = end - pd.Timedelta(days=args.recent_days)
    baseline_cutoff = end - pd.Timedelta(days=args.baseline_days)

    recent = df[df["race_date"] > recent_cutoff]
    baseline = df[(df["race_date"] > baseline_cutoff) & (df["race_date"] <= recent_cutoff)]
    all_data = df[df["race_date"] > baseline_cutoff]

    print(f"  直近 ({args.recent_days}日): {len(recent):,} レース")
    print(f"  ベース ({args.baseline_days-args.recent_days}日): {len(baseline):,} レース")

    print(f"\n=== 1. 艇別1着率（全レース） ===")
    rows = [
        _lane_distribution(baseline, f"ベースライン ({args.baseline_days-args.recent_days}日)"),
        _lane_distribution(recent, f"直近 ({args.recent_days}日)"),
    ]
    _print_lane_table(rows)

    # 2号艇1着率の有意性
    base_l2 = (baseline["winner_lane"] == 2).mean()
    rec_l2_count = int((recent["winner_lane"] == 2).sum())
    rec_l2_rate = rec_l2_count / len(recent) if len(recent) else 0
    diff = rec_l2_rate - base_l2
    p_val = _binomial_pvalue(rec_l2_count, len(recent), base_l2)
    print(f"\n  2号艇1着率の変化: {base_l2*100:.1f}% → {rec_l2_rate*100:.1f}% "
          f"({diff*100:+.1f}pt)")
    print(f"  片側二項検定 p値: {p_val:.4f}", end=" ")
    if p_val < 0.001:
        print("→ 🚨 極めて有意（構造変化の強い証拠）")
    elif p_val < 0.01:
        print("→ ⚠ 有意（構造変化の可能性）")
    elif p_val < 0.05:
        print("→ ⚠ やや有意")
    else:
        print("→ 偶然の可能性が高い")

    print(f"\n=== 2. v2 ベット対象帯 (1号艇オッズ 3-10倍) での艇別1着率 ===")
    _bet_zone_focus(baseline, f"ベースライン")
    _bet_zone_focus(recent, f"直近 {args.recent_days}日")

    _per_venue_recent_vs_baseline(recent, baseline)
    _monthly_trend(all_data)

    print(f"\n=== 結論 ===")
    if p_val < 0.01 and diff > 0:
        print(f"  🚨 直近の 2号艇1着率上昇は統計的に有意 (p={p_val:.4f})")
        print(f"     → 構造変化の可能性。モデル再学習 or 一時停止を推奨")
    elif p_val < 0.05:
        print(f"  ⚠ やや有意 (p={p_val:.4f})。追加データで再評価推奨")
    else:
        print(f"  ✓ 統計的に偶然の範囲内 (p={p_val:.4f})。続行可能")


if __name__ == "__main__":
    main()
