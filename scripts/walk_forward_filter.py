"""真の Walk-Forward テスト: Train期間でフィルタを「学習」、Test期間で検証。

各 N (30/60/.../180日) について:
  1. Train期間 = 全期間 - 直近N日
  2. Train期間で「明らかに弱いセグメント」(場・EV帯・オッズ帯) を発見
  3. その除外フィルタを生成（= アルゴリズムを「学習」）
  4. Train・Test の両期間にフィルタを適用
  5. フィルタ有無で ROI を比較
     - Train で改善する → フィルタは Train データに合っている
     - Test でも改善する → 汎化している（過学習でない）
     - Test で悪化する → 過学習

使い方:
    python -m scripts.walk_forward_filter

    # フィルタ閾値を変える（より厳しく除外: 弱いセグメントが多くなる）
    python -m scripts.walk_forward_filter --exclude-threshold 0.20

    # 期間カスタム
    python -m scripts.walk_forward_filter --periods 7 14 30 90
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd

from src.utils.config import PROCESSED_DIR, VENUE_CODES


def _load_bets(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["hit"] = df["hit"].astype(bool)
    df["race_date"] = pd.to_datetime(
        df["race_id"].astype(str).str[:8], format="%Y%m%d", errors="coerce"
    )
    parts = df["race_id"].astype(str).str.split("-", expand=True)
    df["venue"] = parts[1]
    df["race_no"] = pd.to_numeric(parts[2], errors="coerce")
    if "race_finished" in df.columns:
        df = df[df["race_finished"].astype(bool)]
    df = df.dropna(subset=["race_date"])

    # 特徴量帯
    def _ev(e):
        if e < 1.10: return "1.05-1.10"
        if e < 1.15: return "1.10-1.15"
        if e < 1.20: return "1.15-1.20"
        if e < 1.30: return "1.20-1.30"
        if e < 1.50: return "1.30-1.50"
        return "1.50+"

    def _odds(o):
        if o < 4.0: return "3.0-4.0"
        if o < 5.0: return "4.0-5.0"
        if o < 6.0: return "5.0-6.0"
        if o < 7.0: return "6.0-7.0"
        if o < 8.0: return "7.0-8.0"
        return "8.0-10.0"

    df["ev_band"] = df["ev"].apply(_ev)
    df["odds_band"] = df["odds_win"].apply(_odds)
    return df


def _calc_roi(df: pd.DataFrame) -> tuple[float, int, int, int]:
    if df.empty:
        return 0.0, 0, 0, 0
    n = len(df)
    hits = int(df["hit"].sum())
    stake = int(df["stake"].sum())
    pnl = int(df["pnl"].sum())
    roi = pnl / stake if stake else 0
    return roi, n, hits, pnl


def _learn_filter(train: pd.DataFrame, threshold: float = 0.30,
                  min_n: int = 30) -> dict:
    """Train データから「除外するセグメント」を学習。

    各セグメント別 ROI が Train全体の ROI から threshold pt 以上低く、
    かつ件数が min_n 以上のものを除外候補とする。
    """
    baseline_roi, _, _, _ = _calc_roi(train)
    excluded: dict[str, set] = {
        "venue": set(),
        "ev_band": set(),
        "odds_band": set(),
    }
    for axis in ("venue", "ev_band", "odds_band"):
        for value, sub in train.groupby(axis):
            if len(sub) < min_n:
                continue
            seg_roi = sub["pnl"].sum() / sub["stake"].sum() if sub["stake"].sum() else 0
            if seg_roi < baseline_roi - threshold:
                excluded[axis].add(value)
    return excluded


def _apply_filter(df: pd.DataFrame, excluded: dict) -> pd.DataFrame:
    mask = pd.Series([True] * len(df), index=df.index)
    for axis, values in excluded.items():
        if values:
            mask &= ~df[axis].isin(values)
    return df[mask]


def _format_filter(excluded: dict) -> str:
    parts = []
    if excluded["venue"]:
        names = [f"{VENUE_CODES.get(v, '?')}({v})" for v in sorted(excluded["venue"])]
        parts.append(f"場除外: {', '.join(names)}")
    if excluded["ev_band"]:
        parts.append(f"EV帯除外: {', '.join(sorted(excluded['ev_band']))}")
    if excluded["odds_band"]:
        parts.append(f"オッズ帯除外: {', '.join(sorted(excluded['odds_band']))}")
    return "\n      ".join(parts) if parts else "(除外なし)"


def _judge_generalization(train_improvement: float, test_improvement: float,
                          test_n: int) -> str:
    if test_n < 30:
        return "△ Testサンプル少 → 統計的に判定弱い"
    # Train でも Test でも改善 → 汎化
    if train_improvement > 0 and test_improvement > 0.05:
        return "🌟 Train でも Test でも改善 → フィルタは汎化"
    # Train 改善・Test も同方向
    if train_improvement > 0 and test_improvement > -0.05:
        return "✓ Train でも Test でも改善 (Test 弱め)"
    # Train 改善・Test 微減
    if train_improvement > 0 and test_improvement > -0.20:
        return "△ Test では効果薄 (フィルタの効果限定的)"
    # Train 改善・Test 悪化
    if train_improvement > 0 and test_improvement < -0.20:
        return "🚨 過学習の可能性 (Test で悪化)"
    return "‐ フィルタ無し or 効果なし"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bets", default=None,
                   help="bets_lane1_kelly.csv のパス")
    p.add_argument("--periods", type=int, nargs="*",
                   default=[30, 60, 90, 120, 150, 180],
                   help="Test期間（日数）のリスト。デフォルト 30/60/90/120/150/180")
    p.add_argument("--exclude-threshold", type=float, default=0.30,
                   help="セグメント除外の閾値（ベースライン-Xpt以下なら除外、デフォルト 0.30）")
    p.add_argument("--min-n", type=int, default=30,
                   help="セグメントの最低件数（デフォルト 30）")
    args = p.parse_args()

    candidates = [
        Path(args.bets) if args.bets else None,
        PROCESSED_DIR / "backtest_history" / "bets_lane1_kelly.csv",
    ]
    bets_path = next((p for p in candidates if p and p.exists()), None)
    if bets_path is None:
        print("bets_lane1_kelly.csv が見つかりません。"
              "`run_signal_v2 --autofill --backtest-days 730` を実行してください")
        return

    print(f"=== 入力: {bets_path} ===")
    df = _load_bets(bets_path)
    if df.empty:
        print("確定ベット無し")
        return

    cutoff = df["race_date"].max()
    earliest = df["race_date"].min()
    print(f"  全期間: {earliest.date()} 〜 {cutoff.date()} ({len(df):,}件)")
    print(f"  パラメータ: 除外閾値 -{args.exclude_threshold*100:.0f}pt / "
          f"最低件数 {args.min_n}件")

    summary_rows: list[dict] = []

    for n_days in sorted(args.periods):
        cutoff_date = cutoff - timedelta(days=n_days - 1)
        train = df[df["race_date"] < cutoff_date]
        test = df[df["race_date"] >= cutoff_date]

        if len(train) < 100 or len(test) < 10:
            continue

        # フィルタを Train から学習
        excluded = _learn_filter(train,
                                 threshold=args.exclude_threshold,
                                 min_n=args.min_n)

        # 4組の ROI 計算
        train_raw_roi, train_n, _, train_pnl = _calc_roi(train)
        test_raw_roi, test_n, _, test_pnl = _calc_roi(test)
        train_f = _apply_filter(train, excluded)
        test_f = _apply_filter(test, excluded)
        train_f_roi, train_f_n, _, train_f_pnl = _calc_roi(train_f)
        test_f_roi, test_f_n, _, test_f_pnl = _calc_roi(test_f)

        train_improvement = train_f_roi - train_raw_roi
        test_improvement = test_f_roi - test_raw_roi

        print(f"\n{'='*70}")
        print(f"【直近 {n_days}日 を Test に切り出し】")
        print(f"  Train: {train_n:,}件 ({train['race_date'].min().date()}"
              f"〜{train['race_date'].max().date()})")
        print(f"  Test : {test_n:,}件 ({test['race_date'].min().date()}"
              f"〜{test['race_date'].max().date()})")
        print(f"\n  学習したフィルタ:")
        print(f"      {_format_filter(excluded)}")

        print(f"\n  Train成績:")
        print(f"    フィルタ無し: {train_n:>5}件 ROI {train_raw_roi*100:>+6.1f}% "
              f"PnL +¥{train_pnl:,}")
        print(f"    フィルタ有り: {train_f_n:>5}件 ROI {train_f_roi*100:>+6.1f}% "
              f"PnL +¥{train_f_pnl:,}"
              f" ({train_improvement*100:+.1f}pt)")

        print(f"\n  Test成績:")
        print(f"    フィルタ無し: {test_n:>5}件 ROI {test_raw_roi*100:>+6.1f}% "
              f"PnL {'+'if test_pnl>0 else ''}¥{test_pnl:,}")
        print(f"    フィルタ有り: {test_f_n:>5}件 ROI {test_f_roi*100:>+6.1f}% "
              f"PnL {'+'if test_f_pnl>0 else ''}¥{test_f_pnl:,}"
              f" ({test_improvement*100:+.1f}pt)")

        print(f"\n  判定: {_judge_generalization(train_improvement, test_improvement, test_n)}")

        summary_rows.append({
            "N": n_days,
            "test_n": test_n,
            "train_raw": train_raw_roi,
            "train_f": train_f_roi,
            "test_raw": test_raw_roi,
            "test_f": test_f_roi,
            "train_imp": train_improvement,
            "test_imp": test_improvement,
        })

    # サマリ表
    print(f"\n\n{'='*70}")
    print(f"=== サマリ表 (Train/Test を生 ROI と フィルタ後 ROI で比較) ===")
    print(f"{'Test期間':<10}{'Test件数':>8}{'Train生':>10}{'Train+F':>10}"
          f"{'Test生':>10}{'Test+F':>10}{'Test改善':>10}")
    print("-" * 70)
    for r in summary_rows:
        sign = "+" if r["test_imp"] >= 0 else ""
        print(f"直近{r['N']:>3}日   {r['test_n']:>7,}"
              f" {r['train_raw']*100:>+7.1f}%"
              f" {r['train_f']*100:>+7.1f}%"
              f" {r['test_raw']*100:>+7.1f}%"
              f" {r['test_f']*100:>+7.1f}%"
              f"   {sign}{r['test_imp']*100:>+6.1f}pt")

    print(f"\n判定の見方:")
    print(f"  Test改善 + → フィルタは Test でも有効（汎化）")
    print(f"  Test改善 - → 過学習 or 直近の市場変化（フィルタ効かない）")


if __name__ == "__main__":
    main()
