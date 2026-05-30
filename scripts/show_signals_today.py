"""シグナル履歴を結果付きで一覧表示。今日 / 任意日 / 過去N日まで対応。

signal_snapshots を集計し、payouts と照合して的中/外れ/PnLを表示。
HTML を開かずにターミナルだけで確認したい時用の軽量ツール。

使い方:
    python -m scripts.show_signals_today                  # 今日のみ
    python -m scripts.show_signals_today --date 2026-05-28
    python -m scripts.show_signals_today --days 7         # 過去7日（今日含む）の日別サマリ
    python -m scripts.show_signals_today --days 7 -v      # サマリ + 個別レース全表示
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.config import PROCESSED_DIR, RAW_DIR, VENUE_CODES


def _load_snapshots(target: date) -> pd.DataFrame:
    log_path = (PROCESSED_DIR / "signal_log"
                / f"signal_snapshots_{target.strftime('%Y%m%d')}.csv")
    if not log_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(log_path)
    except Exception:
        return pd.DataFrame()
    return df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["snapshot_at"] = pd.to_datetime(df["snapshot_at"], errors="coerce")
    g = df.groupby("race_id")
    return pd.DataFrame({
        "first_seen": g["snapshot_at"].min(),
        "last_seen": g["snapshot_at"].max(),
        "n": g["snapshot_at"].count(),
        "max_ev": g["ev"].max(),
        "latest_ev": g["ev"].last(),
        "latest_odds": g["odds_win"].last(),
        "latest_stake": g["stake"].last(),
    }).reset_index()


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


def _enrich_row(rid: str, latest_stake: int, winners: dict) -> dict:
    """payouts と突合して finished/hit/pnl を返す。"""
    result = winners.get(rid)
    if not result or pd.isna(result["winner_lane"]):
        return {"finished": False, "hit": False, "pnl": 0, "winner_lane": None}
    winner = int(result["winner_lane"])
    hit = (winner == 1)
    if hit:
        payout = float(result["payout_yen"])
        pnl = int(latest_stake * payout / 100.0 - latest_stake)
    else:
        pnl = -latest_stake
    return {"finished": True, "hit": hit, "pnl": pnl, "winner_lane": winner}


def _collect(targets: Iterable[date], winners: dict) -> pd.DataFrame:
    """対象日それぞれの snapshot を集計し1つの DataFrame に。
    各行に date / finished / hit / pnl / winner_lane を付与。
    """
    parts: list[pd.DataFrame] = []
    for t in targets:
        df = _load_snapshots(t)
        if df.empty:
            continue
        agg = _aggregate(df)
        agg["date"] = t.isoformat()
        parts.append(agg)
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    enriched = combined.apply(
        lambda r: _enrich_row(str(r["race_id"]),
                              int(r["latest_stake"]) if pd.notna(r["latest_stake"]) else 0,
                              winners),
        axis=1, result_type="expand",
    )
    return pd.concat([combined, enriched], axis=1)


def _print_daily_summary(df: pd.DataFrame) -> None:
    print("【日別サマリ】")
    fmt = "{date:<11} {n:>5} {fin:>5} {hit:>5} {rate:>7} {stake:>10} {pnl:>10} {roi:>8}"
    print(fmt.format(date="日付", n="件数", fin="確定", hit="的中",
                     rate="勝率", stake="ステーク", pnl="PnL", roi="ROI"))
    print("-" * 75)
    for d, g in df.groupby("date"):
        n = len(g)
        fin = int(g["finished"].sum())
        hit = int(g["hit"].sum())
        rate = f"{hit / fin * 100:.1f}%" if fin else "-"
        stake = int(g[g["finished"]]["latest_stake"].sum())
        pnl = int(g["pnl"].sum())
        roi = f"{pnl / stake * 100:+.1f}%" if stake else "-"
        sign = "+" if pnl > 0 else ""
        print(fmt.format(date=d, n=f"{n}", fin=f"{fin}", hit=f"{hit}",
                         rate=rate, stake=f"¥{stake:,}",
                         pnl=f"{sign}¥{pnl:,}", roi=roi))


def _print_details(df: pd.DataFrame) -> None:
    print("【個別レース】")
    fmt = "{date:<11} {venue:>10} {rno:>4} {seen:>11} {ev_max:>7} {odds:>6} {stake:>9}  {status}"
    print(fmt.format(date="日付", venue="場", rno="R", seen="観測時間",
                     ev_max="最高EV", odds="オッズ", stake="ステーク", status="結果"))
    print("-" * 95)
    df_sorted = df.sort_values(["date", "max_ev"], ascending=[True, False])
    for _, r in df_sorted.iterrows():
        rid = str(r["race_id"])
        parts = rid.split("-")
        venue = VENUE_CODES.get(parts[1], "?") if len(parts) > 1 else "?"
        rno = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
        seen = f"{r['first_seen'].strftime('%H:%M')}-{r['last_seen'].strftime('%H:%M')}"
        stake = int(r["latest_stake"]) if pd.notna(r["latest_stake"]) else 0
        if r["finished"]:
            if r["hit"]:
                status = f"🟢 1着 +¥{int(r['pnl']):,}"
            else:
                status = f"✕ {int(r['winner_lane'])}号艇1着 -¥{stake:,}"
        else:
            status = "⏳ 未確定"
        print(fmt.format(date=r["date"], venue=f"{venue}({parts[1]})",
                         rno=f"{rno}R", seen=seen,
                         ev_max=f"{r['max_ev']:.2f}",
                         odds=f"{r['latest_odds']:.2f}",
                         stake=f"¥{stake:,}", status=status))


def _print_grand_total(df: pd.DataFrame, label: str = "合計") -> None:
    n = len(df)
    fin = int(df["finished"].sum())
    hit = int(df["hit"].sum())
    stake = int(df[df["finished"]]["latest_stake"].sum())
    pnl = int(df["pnl"].sum())
    print()
    print(f"=== {label} ===")
    print(f"レース数: {n} 件 (確定 {fin} / 未確定 {n - fin})")
    if fin:
        print(f"勝率:     {hit}/{fin} = {hit / fin * 100:.1f}%")
        print(f"ステーク: ¥{stake:,}")
        sign = "+" if pnl > 0 else ""
        print(f"PnL:      {sign}¥{pnl:,}")
        if stake:
            print(f"ROI:      {sign}{pnl / stake * 100:.1f}%")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", default=None, help="単日指定 YYYY-MM-DD")
    p.add_argument("--days", type=int, default=None,
                   help="過去N日（今日含む）の集計。指定時は日別サマリも表示")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="集計だけでなく個別レースも全表示")
    args = p.parse_args()

    if args.date:
        targets = [date.fromisoformat(args.date)]
    elif args.days:
        end = date.today()
        targets = [end - timedelta(days=i) for i in range(args.days - 1, -1, -1)]
    else:
        targets = [date.today()]

    winners = _load_winners()
    df = _collect(targets, winners)
    if df.empty:
        print(f"対象期間にシグナルログなし: {targets[0]} 〜 {targets[-1]}")
        return

    period_label = (f"{targets[0]} 〜 {targets[-1]}"
                    if len(targets) > 1 else str(targets[0]))
    print(f"=== シグナル履歴 ({period_label}) — {len(df)} レース ===")
    print()

    multi_day = len(targets) > 1
    if multi_day:
        _print_daily_summary(df)
        print()

    if args.verbose or not multi_day:
        _print_details(df)

    _print_grand_total(df, label="期間合計" if multi_day else "合計")


if __name__ == "__main__":
    main()
