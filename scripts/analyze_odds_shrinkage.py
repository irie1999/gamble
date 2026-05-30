"""1号艇 単勝オッズの「スクレイプ値 → 確定値」の収縮率を実測。

odds_win.csv (スクレイプ時点) と races_payouts.parquet (実際の確定オッズ)
を突き合わせて、1号艇のオッズが平均何%下がるかを集計する。

これで `--odds-shrinkage-max` パラメータを実データから校正できる。

使い方:
    python -m scripts.analyze_odds_shrinkage
    python -m scripts.analyze_odds_shrinkage --since 2024-01-01
    python -m scripts.analyze_odds_shrinkage --min-odds 3.0 --max-odds 10.0
"""
from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd

from src.utils.config import RAW_DIR


def _load_data(odds_path, payouts_path):
    odds = pd.read_csv(odds_path)
    odds = odds[odds["lane"] == 1].copy()
    odds = odds.rename(columns={"odds_win": "scraped_odds"})
    odds = odds[["race_id", "scraped_odds"]]

    pay = pd.read_parquet(payouts_path)
    pay = pay[(pay["bet_type"] == "win") & (pay["combo"] == "1")].copy()
    pay["settled_odds"] = pay["payout_yen"].astype(float) / 100.0
    pay = pay[["race_id", "settled_odds"]]

    df = odds.merge(pay, on="race_id", how="inner")
    df["race_date"] = pd.to_datetime(
        df["race_id"].astype(str).str[:8], format="%Y%m%d", errors="coerce"
    )
    df = df[(df["scraped_odds"] >= 1.0) & (df["settled_odds"] >= 1.0)]
    df["shrinkage"] = (df["settled_odds"] - df["scraped_odds"]) / df["scraped_odds"]
    df["ratio"] = df["settled_odds"] / df["scraped_odds"]
    return df


def _print_overall(df: pd.DataFrame):
    print(f"\n=== 全体 (N={len(df):,}) ===")
    print(f"  スクレイプオッズ 平均: {df['scraped_odds'].mean():.2f} 倍 / 中央値: {df['scraped_odds'].median():.2f}")
    print(f"  確定オッズ       平均: {df['settled_odds'].mean():.2f} 倍 / 中央値: {df['settled_odds'].median():.2f}")
    print(f"  収縮率           平均: {df['shrinkage'].mean()*100:+.1f}% / 中央値: {df['shrinkage'].median()*100:+.1f}%")
    print(f"  10%/25%/50%/75%/90%tile (収縮率): "
          f"{df['shrinkage'].quantile(0.10)*100:+.0f}% / "
          f"{df['shrinkage'].quantile(0.25)*100:+.0f}% / "
          f"{df['shrinkage'].quantile(0.50)*100:+.0f}% / "
          f"{df['shrinkage'].quantile(0.75)*100:+.0f}% / "
          f"{df['shrinkage'].quantile(0.90)*100:+.0f}%")
    drop_pct = (df["shrinkage"] < 0).mean() * 100
    big_drop = (df["shrinkage"] < -0.5).mean() * 100
    print(f"  下落した割合: {drop_pct:.1f}% / 50%以上下落した割合: {big_drop:.1f}%")


def _print_by_band(df: pd.DataFrame):
    bands = [
        ("〜2.0倍",  0,   2.0),
        ("2-3倍",    2.0, 3.0),
        ("3-5倍",    3.0, 5.0),
        ("5-7倍",    5.0, 7.0),
        ("7-10倍",   7.0, 10.0),
        ("10倍〜",  10.0, 1e9),
    ]
    print(f"\n=== オッズ帯別（スクレイプ値で分類） ===")
    fmt = "{name:>8} {n:>7} {scr:>7} {set:>7} {mean:>8} {med:>8} {q90:>8} {big:>9}"
    print(fmt.format(name="帯", n="件数", scr="スク平均", set="確定平均",
                     mean="平均収縮", med="中央収縮", q90="90%tile",
                     big="≥50%下落"))
    print("-" * 75)
    for name, lo, hi in bands:
        sub = df[(df["scraped_odds"] >= lo) & (df["scraped_odds"] < hi)]
        if sub.empty:
            continue
        big = (sub["shrinkage"] < -0.5).mean() * 100
        print(fmt.format(
            name=name, n=f"{len(sub):,}",
            scr=f"{sub['scraped_odds'].mean():.2f}",
            set=f"{sub['settled_odds'].mean():.2f}",
            mean=f"{sub['shrinkage'].mean()*100:+.1f}%",
            med=f"{sub['shrinkage'].median()*100:+.1f}%",
            q90=f"{sub['shrinkage'].quantile(0.90)*100:+.0f}%",
            big=f"{big:.1f}%",
        ))


def _print_recommendation(df: pd.DataFrame):
    """v2 戦略のオッズ帯 (3.0〜10.0倍) で集計し、推奨 max_shrinkage を示す。

    収縮率は負値（下落）なので、保守的 = より大きな下落を見込む =
    分布の下側（左裾）quantile を使う。
    """
    sub = df[(df["scraped_odds"] >= 3.0) & (df["scraped_odds"] <= 10.0)]
    if sub.empty:
        return
    print(f"\n=== 推奨 max_shrinkage (v2 帯: 3.0〜10.0倍 / N={len(sub):,}) ===")
    print(f"  分布 (収縮率, マイナス=下落):")
    print(f"    平均   : {sub['shrinkage'].mean()*100:+.1f}%")
    print(f"    中央値 : {sub['shrinkage'].median()*100:+.1f}%")
    print(f"    25%tile: {sub['shrinkage'].quantile(0.25)*100:+.0f}% (下位25% = これ以上下がる確率25%)")
    print(f"    10%tile: {sub['shrinkage'].quantile(0.10)*100:+.0f}% (下位10% = これ以上下がる確率10%)")

    avg = abs(sub["shrinkage"].mean())
    med = abs(sub["shrinkage"].median())
    q25 = abs(sub["shrinkage"].quantile(0.25))  # 下落側 25% tile
    q10 = abs(sub["shrinkage"].quantile(0.10))  # 下落側 10% tile
    print()
    print(f"  解釈:")
    print(f"    現状の --odds-shrinkage-max 0.30 = 30% 下落を仮定")
    if 0.30 >= q25:
        print(f"    → 現状は 75% のケースをカバー (十分保守的)")
    elif 0.30 >= med:
        print(f"    → 現状は中央値はカバー、上位25%の下落 ({q25*100:.0f}%) には届かない")
    elif 0.30 >= avg:
        print(f"    → 現状は平均値はカバー、半数のケースで読み負け")
    else:
        print(f"    → 現状は平均すら下回る ({avg*100:.0f}%) → 上げるべき")
    print()
    print(f"  推奨候補（保守度別）:")
    print(f"    アグレッシブ: --odds-shrinkage-max {avg:.2f}  (平均値カバー → 半数で読み負ける)")
    print(f"    バランス:    --odds-shrinkage-max {med:.2f}  (中央値カバー → 半数カバー)")
    print(f"    保守的:      --odds-shrinkage-max {q25:.2f}  (75%カバー → 4回に3回は耐える)")
    print(f"    超保守:      --odds-shrinkage-max {q10:.2f}  (90%カバー → 10回に9回は耐える)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--odds", default=str(RAW_DIR / "odds_win.csv"))
    p.add_argument("--payouts", default=str(RAW_DIR / "races_payouts.parquet"))
    p.add_argument("--since", default=None, help="YYYY-MM-DD 以降")
    p.add_argument("--until", default=None, help="YYYY-MM-DD 以前")
    args = p.parse_args()

    df = _load_data(args.odds, args.payouts)
    if args.since:
        df = df[df["race_date"] >= pd.Timestamp(args.since)]
    if args.until:
        df = df[df["race_date"] <= pd.Timestamp(args.until)]

    if df.empty:
        print("データ無し")
        return

    period_min = df["race_date"].min().date()
    period_max = df["race_date"].max().date()
    print(f"対象期間: {period_min} 〜 {period_max}  ({len(df):,} レース)")

    _print_overall(df)
    _print_by_band(df)
    _print_recommendation(df)


if __name__ == "__main__":
    main()
