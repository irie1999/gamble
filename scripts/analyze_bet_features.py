"""過去ベット履歴から「より当たりやすい特徴」を抽出。

bets_lane1_kelly.csv (944件等) を読み込み、特徴量ごとの ROI / 勝率を集計。
新しいフィルタ条件を提案する。

使い方:
    python -m scripts.analyze_bet_features

    # 特定のベットファイルを指定
    python -m scripts.analyze_bet_features --bets path/to/bets.csv

    # 最低件数を変える（少サンプルでもブレイクダウンを見たい時）
    python -m scripts.analyze_bet_features --min-n 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import PROCESSED_DIR, VENUE_CODES


def _load_bets(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["hit"] = df["hit"].astype(bool)
    # race_id から venue / race_no / date を抽出
    parts = df["race_id"].astype(str).str.split("-", expand=True)
    df["venue"] = parts[1]
    df["race_no"] = pd.to_numeric(parts[2], errors="coerce")
    df["race_date"] = pd.to_datetime(parts[0], format="%Y%m%d", errors="coerce")
    df["dow"] = df["race_date"].dt.dayofweek  # 0=月
    df["month"] = df["race_date"].dt.month
    return df


def _segment_stats(df: pd.DataFrame, group_col: str, *,
                   min_n: int = 30) -> pd.DataFrame:
    """グループごとに件数/勝率/ROI を計算し、最低件数フィルタ適用。"""
    g = df.groupby(group_col).agg(
        n=("stake", "size"),
        hits=("hit", "sum"),
        stake=("stake", "sum"),
        pnl=("pnl", "sum"),
    )
    g["hit_rate"] = g["hits"] / g["n"]
    g["roi"] = g["pnl"] / g["stake"]
    g = g[g["n"] >= min_n]
    return g.sort_values("roi", ascending=False)


def _print_segment(label: str, stats: pd.DataFrame,
                   value_format: str = "{}", baseline_roi: float = 0,
                   top_n: int = 10) -> None:
    print(f"\n=== {label} ===")
    if stats.empty:
        print("  (件数不足)")
        return
    print(f"{'値':<14} {'件数':>6} {'勝率':>7} {'ROI':>8}{'差':>8}")
    print("-" * 50)
    for v, r in stats.iterrows():
        v_str = value_format.format(v)
        diff = r["roi"] - baseline_roi
        flag = " 🌟" if diff > 0.20 and r["n"] >= 30 else (" ⚠" if diff < -0.30 else "")
        print(f"{v_str:<14} {int(r['n']):>6,} {r['hit_rate']*100:>6.1f}% "
              f"{r['roi']*100:>+7.1f}%{diff*100:>+7.1f}pt{flag}")


def _ev_bands(ev: float) -> str:
    if ev < 1.10: return "1.05-1.10"
    if ev < 1.15: return "1.10-1.15"
    if ev < 1.20: return "1.15-1.20"
    if ev < 1.30: return "1.20-1.30"
    if ev < 1.50: return "1.30-1.50"
    return "1.50+"


def _odds_bands(odds: float) -> str:
    if odds < 4.0: return "3.0-4.0"
    if odds < 5.0: return "4.0-5.0"
    if odds < 6.0: return "5.0-6.0"
    if odds < 7.0: return "6.0-7.0"
    if odds < 8.0: return "7.0-8.0"
    return "8.0-10.0"


def _race_no_bands(rno: int) -> str:
    if rno <= 3: return "1-3R (朝)"
    if rno <= 6: return "4-6R (昼)"
    if rno <= 9: return "7-9R (午後)"
    return "10-12R (夜)"


def _dow_name(d: int) -> str:
    return ["月", "火", "水", "木", "金", "土", "日"][int(d)]


def _venue_name(code: str) -> str:
    return f"{VENUE_CODES.get(str(code), '?')}({code})"


def _combined_analysis(df: pd.DataFrame, baseline_roi: float, min_n: int = 30):
    """2軸の組合せで良い/悪いセグメントを探す。"""
    print(f"\n=== 2軸組合せ: EV帯 × オッズ帯 ===")
    cross = df.groupby(["ev_band", "odds_band"]).agg(
        n=("stake", "size"),
        hits=("hit", "sum"),
        stake=("stake", "sum"),
        pnl=("pnl", "sum"),
    )
    cross["hit_rate"] = cross["hits"] / cross["n"]
    cross["roi"] = cross["pnl"] / cross["stake"]
    cross = cross[cross["n"] >= min_n].sort_values("roi", ascending=False)
    if cross.empty:
        print("  (件数不足)")
        return
    print(f"\n{'EV帯':<12} {'オッズ帯':<10} {'件数':>6} {'勝率':>7} {'ROI':>8}{'差':>8}")
    print("-" * 60)
    for (ev, od), r in cross.head(15).iterrows():
        diff = r["roi"] - baseline_roi
        flag = " 🌟" if diff > 0.20 else (" ⚠" if diff < -0.30 else "")
        print(f"{ev:<12} {od:<10} {int(r['n']):>6,} {r['hit_rate']*100:>6.1f}% "
              f"{r['roi']*100:>+7.1f}%{diff*100:>+7.1f}pt{flag}")


def _suggest_filters(df: pd.DataFrame, baseline_roi: float):
    """単純に「ベースラインより明らかに ROI 低いセグメントを除外」する案を提示。"""
    print(f"\n=== フィルタ提案 (ベースライン ROI {baseline_roi*100:.1f}% より明らかに悪いセグメント除外) ===")
    candidates = []

    # 場
    for venue, sub in df.groupby("venue"):
        if len(sub) < 30:
            continue
        roi = sub["pnl"].sum() / sub["stake"].sum()
        if roi < baseline_roi - 0.30 and roi < 0:
            candidates.append(("venue", venue, _venue_name(venue),
                               len(sub), roi))

    # オッズ帯
    for band, sub in df.groupby("odds_band"):
        if len(sub) < 30:
            continue
        roi = sub["pnl"].sum() / sub["stake"].sum()
        if roi < baseline_roi - 0.30 and roi < 0:
            candidates.append(("odds_band", band, band, len(sub), roi))

    # EV帯
    for band, sub in df.groupby("ev_band"):
        if len(sub) < 30:
            continue
        roi = sub["pnl"].sum() / sub["stake"].sum()
        if roi < baseline_roi - 0.30 and roi < 0:
            candidates.append(("ev_band", band, band, len(sub), roi))

    if not candidates:
        print("  → 明らかに悪いセグメント無し（全体的に分散）")
        return

    candidates.sort(key=lambda x: x[4])
    print(f"\n{'カテゴリ':<10} {'値':<20} {'件数':>6} {'ROI':>8}")
    print("-" * 50)
    for cat, val, label, n, roi in candidates:
        print(f"{cat:<10} {label:<20} {n:>6} {roi*100:>+7.1f}%")

    # 「これらを全部除外したら？」の試算
    print(f"\n=== 「上記を全部除外したら？」シミュレーション ===")
    mask = pd.Series([True] * len(df), index=df.index)
    for cat, val, _, _, _ in candidates:
        if cat == "venue":
            mask &= df["venue"] != val
        elif cat == "odds_band":
            mask &= df["odds_band"] != val
        elif cat == "ev_band":
            mask &= df["ev_band"] != val
    filtered = df[mask]
    if filtered.empty:
        print("  全部除外 → ベット候補ゼロに")
        return
    n_new = len(filtered)
    n_drop = len(df) - n_new
    stake = filtered["stake"].sum()
    pnl = filtered["pnl"].sum()
    new_roi = pnl / stake if stake else 0
    new_hit = filtered["hit"].sum() / n_new
    print(f"  残: {n_new:,}件 (除外 {n_drop:,}件)")
    print(f"  勝率: {new_hit*100:.1f}%")
    print(f"  PnL: {'+' if pnl > 0 else ''}¥{int(pnl):,}")
    print(f"  ROI: {new_roi*100:+.1f}% (元 {baseline_roi*100:+.1f}% → {(new_roi-baseline_roi)*100:+.1f}pt)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bets", default=None,
                   help="bets_lane1_kelly.csv のパス")
    p.add_argument("--min-n", type=int, default=30,
                   help="セグメント表示の最低件数（デフォルト30）")
    args = p.parse_args()

    candidates = [
        Path(args.bets) if args.bets else None,
        PROCESSED_DIR / "backtest_history" / "bets_lane1_kelly.csv",
    ]
    bets_path = next((p for p in candidates if p and p.exists()), None)
    if bets_path is None:
        print("bets_lane1_kelly.csv が見つかりません。"
              "先に `run_signal_v2 --autofill --backtest-days 730` を実行してください。")
        return

    print(f"=== 入力: {bets_path} ===")
    df = _load_bets(bets_path)
    df = df[df["race_finished"] if "race_finished" in df.columns else df.index.notnull()]

    # ベースライン
    n = len(df)
    stake_total = df["stake"].sum()
    pnl_total = df["pnl"].sum()
    baseline_roi = pnl_total / stake_total if stake_total else 0
    hit_rate = df["hit"].sum() / n
    print(f"  全ベット: {n:,}件 / 勝率 {hit_rate*100:.1f}% / "
          f"ROI {baseline_roi*100:+.1f}% / PnL +¥{int(pnl_total):,}")

    # 特徴量を計算
    df["ev_band"] = df["ev"].apply(_ev_bands)
    df["odds_band"] = df["odds_win"].apply(_odds_bands)
    df["rno_band"] = df["race_no"].apply(_race_no_bands)
    df["dow_name"] = df["dow"].apply(_dow_name)

    # 1. 単軸ブレイクダウン
    _print_segment("EV帯別",
                   _segment_stats(df, "ev_band", min_n=args.min_n),
                   baseline_roi=baseline_roi)
    _print_segment("オッズ帯別",
                   _segment_stats(df, "odds_band", min_n=args.min_n),
                   baseline_roi=baseline_roi)
    _print_segment("場別",
                   _segment_stats(df, "venue", min_n=args.min_n),
                   value_format="{}", baseline_roi=baseline_roi)
    # venue は名前付きで再表示
    venue_stats = _segment_stats(df, "venue", min_n=args.min_n)
    if not venue_stats.empty:
        print(f"\n=== 場別 (場名付き) ===")
        print(f"{'場':<14} {'件数':>6} {'勝率':>7} {'ROI':>8}")
        for v, r in venue_stats.iterrows():
            diff = r["roi"] - baseline_roi
            flag = " 🌟" if diff > 0.20 else (" ⚠" if diff < -0.30 else "")
            print(f"{_venue_name(v):<14} {int(r['n']):>6,} "
                  f"{r['hit_rate']*100:>6.1f}% "
                  f"{r['roi']*100:>+7.1f}%{flag}")

    _print_segment("レース番号帯別",
                   _segment_stats(df, "rno_band", min_n=args.min_n),
                   baseline_roi=baseline_roi)
    _print_segment("曜日別",
                   _segment_stats(df, "dow_name", min_n=args.min_n),
                   baseline_roi=baseline_roi)
    _print_segment("月別",
                   _segment_stats(df, "month", min_n=args.min_n),
                   value_format="{}月", baseline_roi=baseline_roi)

    # 2. 2軸組合せ
    _combined_analysis(df, baseline_roi, min_n=20)

    # 3. フィルタ提案
    _suggest_filters(df, baseline_roi)


if __name__ == "__main__":
    main()
