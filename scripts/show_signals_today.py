"""本日 1度でも候補化したレースを、結果付きで一覧表示。

signal_snapshots を集計し、payouts と照合して的中/外れ/PnLを表示。
HTML を開かずにターミナルだけで確認したい時用の軽量ツール。

使い方:
    python -m scripts.show_signals_today
    python -m scripts.show_signals_today --date 2026-05-28
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from src.utils.config import PROCESSED_DIR, RAW_DIR, VENUE_CODES


def _load_snapshots(target: date) -> pd.DataFrame:
    log_path = PROCESSED_DIR / "signal_log" / f"signal_snapshots_{target.strftime('%Y%m%d')}.csv"
    if not log_path.exists():
        return pd.DataFrame()
    return pd.read_csv(log_path)


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df["snapshot_at"] = pd.to_datetime(df["snapshot_at"], errors="coerce")
    g = df.groupby("race_id")
    agg = pd.DataFrame({
        "first_seen": g["snapshot_at"].min(),
        "last_seen": g["snapshot_at"].max(),
        "n": g["snapshot_at"].count(),
        "max_ev": g["ev"].max(),
        "latest_ev": g["ev"].last(),
        "latest_odds": g["odds_win"].last(),
        "latest_stake": g["stake"].last(),
    }).reset_index()
    return agg


def _load_winners() -> dict[str, dict]:
    payouts_path = RAW_DIR / "races_payouts.parquet"
    if not payouts_path.exists():
        return {}
    payouts = pd.read_parquet(payouts_path,
                              columns=["race_id", "bet_type", "combo", "payout_yen"])
    win = payouts[payouts["bet_type"] == "win"].copy()
    win["winner_lane"] = pd.to_numeric(win["combo"], errors="coerce")
    win["payout_yen"] = pd.to_numeric(win["payout_yen"], errors="coerce")
    win = win.dropna(subset=["winner_lane"])
    return win.set_index("race_id")[["winner_lane", "payout_yen"]].to_dict("index")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="対象日 YYYY-MM-DD。省略時は今日")
    args = p.parse_args()
    target = date.fromisoformat(args.date) if args.date else date.today()

    df = _load_snapshots(target)
    if df.empty:
        print(f"signal_log なし: {target}")
        return

    agg = _aggregate(df)
    winners = _load_winners()

    # 最高EV 降順で並び替え
    agg = agg.sort_values("max_ev", ascending=False).reset_index(drop=True)

    print(f"=== 本日 ({target}) のシグナル履歴 — {len(agg)}件 ===")
    print()
    fmt_hdr = "{venue:>10} {rno:>4} {seen:>11} {ev_max:>7} {ev_now:>7} {odds:>6} {stake:>9}  {status}"
    print(fmt_hdr.format(
        venue="場", rno="R", seen="観測時間",
        ev_max="最高EV", ev_now="最新EV", odds="オッズ",
        stake="ステーク", status="結果",
    ))
    print("-" * 95)

    total_stake = 0
    total_pnl = 0
    n_finished = 0
    n_hit = 0
    for _, r in agg.iterrows():
        rid = str(r["race_id"])
        parts = rid.split("-")
        venue = VENUE_CODES.get(parts[1], "?") if len(parts) > 1 else "?"
        rno = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
        seen = f"{r['first_seen'].strftime('%H:%M')}-{r['last_seen'].strftime('%H:%M')}"
        stake = int(r["latest_stake"]) if pd.notna(r["latest_stake"]) else 0

        result = winners.get(rid)
        if result and pd.notna(result["winner_lane"]):
            winner = int(result["winner_lane"])
            hit = (winner == 1)
            if hit:
                payout = float(result["payout_yen"])
                pnl = int(stake * payout / 100.0 - stake)
                status = f"🟢 1着 +¥{pnl:,}"
                n_hit += 1
            else:
                pnl = -stake
                status = f"✕ {winner}号艇1着 -¥{stake:,}"
            total_stake += stake
            total_pnl += pnl
            n_finished += 1
        else:
            status = "⏳ 未確定"

        print(fmt_hdr.format(
            venue=f"{venue}({parts[1]})", rno=f"{rno}R", seen=seen,
            ev_max=f"{r['max_ev']:.2f}", ev_now=f"{r['latest_ev']:.2f}",
            odds=f"{r['latest_odds']:.2f}",
            stake=f"¥{stake:,}", status=status,
        ))

    print("-" * 95)
    print()
    if n_finished:
        rate = n_hit / n_finished * 100.0
        roi = (total_pnl / total_stake * 100.0) if total_stake else 0.0
        sign = "+" if total_pnl > 0 else ""
        print(f"確定: {n_finished}件 (的中 {n_hit}件 / 勝率 {rate:.1f}%)")
        print(f"合計ステーク: ¥{total_stake:,}")
        print(f"合計PnL:      {sign}¥{total_pnl:,}")
        print(f"ROI:          {sign}{roi:.1f}%")
    else:
        print("確定済みベットなし")


if __name__ == "__main__":
    main()
