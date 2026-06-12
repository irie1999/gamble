"""bets_lane1_kelly.csv を時系列に分けて、Train vs Test の成績を比較。

各 N (30/60/90/120/150/180日) について:
  Train期間: 全期間 - 直近N日
  Test期間:  直近N日
  → 両者の勝率・ROI・PnL を比較

戦略が時間的に安定か（直近で崩れていないか）を確認するスクリプト。
過学習や構造変化があれば、Test が Train より大幅に悪化する。

使い方:
    python -m scripts.analyze_temporal_stability

    # 期間カスタム
    python -m scripts.analyze_temporal_stability --periods 7 14 30 90

    # 別のベットファイル
    python -m scripts.analyze_temporal_stability --bets path/to/bets.csv
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd

from src.utils.config import PROCESSED_DIR


def _load_bets(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["hit"] = df["hit"].astype(bool)
    df["race_date"] = pd.to_datetime(
        df["race_id"].astype(str).str[:8], format="%Y%m%d", errors="coerce"
    )
    if "race_finished" in df.columns:
        df = df[df["race_finished"].astype(bool)]
    return df.dropna(subset=["race_date"])


def _stats(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"n": 0, "hits": 0, "stake": 0, "pnl": 0,
                "hit_rate": 0, "roi": 0,
                "first": None, "last": None}
    fin = df  # already finished filter applied
    return {
        "n": n,
        "hits": int(fin["hit"].sum()),
        "stake": int(fin["stake"].sum()),
        "pnl": int(fin["pnl"].sum()),
        "hit_rate": fin["hit"].sum() / n if n else 0,
        "roi": fin["pnl"].sum() / fin["stake"].sum() if fin["stake"].sum() else 0,
        "first": fin["race_date"].min(),
        "last": fin["race_date"].max(),
    }


def _format_period(label: str, s: dict) -> str:
    if s["n"] == 0:
        return f"  {label}: データ無し"
    sign = "+" if s["pnl"] > 0 else ""
    period = f"{s['first'].date()}〜{s['last'].date()}" if s["first"] else "-"
    return (
        f"  {label}:\n"
        f"    期間: {period}\n"
        f"    件数: {s['n']:,} / 勝率 {s['hit_rate']*100:>5.1f}% "
        f"({s['hits']}/{s['n']})\n"
        f"    ステーク: ¥{s['stake']:,}\n"
        f"    PnL: {sign}¥{s['pnl']:,}\n"
        f"    ROI: {sign}{s['roi']*100:>5.1f}%"
    )


def _judge(train_roi: float, test_roi: float, test_n: int) -> str:
    """Train vs Test の関係性をざっくり判定。"""
    diff = test_roi - train_roi
    if test_n < 30:
        return "△ Test件数が少ない (<30件) → 統計的に判断弱い"
    if diff > 0.30:
        return "🌟 Test が Train を大幅に上回る（直近好調）"
    if diff > 0:
        return "✓ Test が Train と同等以上"
    if diff > -0.20:
        return "△ Test が Train より僅かに悪い（誤差範囲）"
    if diff > -0.50:
        return "⚠ Test が Train より明らかに悪い（要注意）"
    return "🚨 Test が Train より大幅悪化（構造変化の疑い）"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bets", default=None,
                   help="bets_lane1_kelly.csv のパス")
    p.add_argument("--periods", type=int, nargs="*",
                   default=[30, 60, 90, 120, 150, 180],
                   help="Test期間（日数）のリスト。デフォルト 30/60/90/120/150/180")
    args = p.parse_args()

    candidates = [
        Path(args.bets) if args.bets else None,
        PROCESSED_DIR / "backtest_history" / "bets_lane1_kelly.csv",
    ]
    bets_path = next((p for p in candidates if p and p.exists()), None)
    if bets_path is None:
        print("bets_lane1_kelly.csv が見つかりません。")
        print("先に `python -m scripts.run_signal_v2 --autofill --backtest-days 730` を実行")
        return

    print(f"=== 入力: {bets_path} ===")
    df = _load_bets(bets_path)
    if df.empty:
        print("確定ベット無し")
        return

    cutoff = df["race_date"].max()
    earliest = df["race_date"].min()
    print(f"  全期間: {earliest.date()} 〜 {cutoff.date()}  "
          f"({(cutoff - earliest).days + 1}日, {len(df):,}件)")

    overall = _stats(df)
    print(f"\n  全期間統計: 勝率 {overall['hit_rate']*100:.1f}% / "
          f"ROI {overall['roi']*100:+.1f}% / PnL +¥{overall['pnl']:,}")

    print(f"\n=== 時系列分割: Train vs Test ===")

    for n_days in sorted(args.periods):
        cutoff_date = cutoff - timedelta(days=n_days - 1)
        test = df[df["race_date"] >= cutoff_date]
        train = df[df["race_date"] < cutoff_date]

        train_s = _stats(train)
        test_s = _stats(test)

        print(f"\n{'='*60}")
        print(f"【直近 {n_days}日 を Test に切り出し】")
        print(_format_period(f"Train ({(cutoff_date - timedelta(days=1) - earliest).days + 1}日)", train_s))
        print(_format_period(f"Test  ({n_days}日)", test_s))

        if test_s["n"] > 0 and train_s["n"] > 0:
            roi_diff = test_s["roi"] - train_s["roi"]
            hit_diff = test_s["hit_rate"] - train_s["hit_rate"]
            sign = "+" if roi_diff >= 0 else ""
            hit_sign = "+" if hit_diff >= 0 else ""
            print(f"\n  差分 (Test - Train):")
            print(f"    勝率: {hit_sign}{hit_diff*100:.1f}pt")
            print(f"    ROI:  {sign}{roi_diff*100:.1f}pt")
            print(f"  判定: {_judge(train_s['roi'], test_s['roi'], test_s['n'])}")

    # サマリ表
    print(f"\n\n{'='*70}")
    print(f"=== サマリ表 ===")
    print(f"{'除外期間':<10} {'Test件数':>8} {'Test勝率':>10} "
          f"{'Test ROI':>10} {'Train ROI':>11} {'差分':>10}")
    print("-" * 65)
    for n_days in sorted(args.periods):
        cutoff_date = cutoff - timedelta(days=n_days - 1)
        test = df[df["race_date"] >= cutoff_date]
        train = df[df["race_date"] < cutoff_date]
        test_s = _stats(test)
        train_s = _stats(train)
        if test_s["n"] == 0 or train_s["n"] == 0:
            continue
        diff = test_s["roi"] - train_s["roi"]
        sign = "+" if diff >= 0 else ""
        print(f"直近{n_days:>3}日   {test_s['n']:>7,} "
              f"{test_s['hit_rate']*100:>8.1f}% "
              f"{test_s['roi']*100:>+8.1f}% "
              f"{train_s['roi']*100:>+9.1f}% "
              f"{sign}{diff*100:>+7.1f}pt")


if __name__ == "__main__":
    main()
