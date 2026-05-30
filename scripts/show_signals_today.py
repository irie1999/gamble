"""シグナル履歴を結果付きで一覧表示。今日 / 任意日 / 過去N日まで対応。

signal_snapshots を集計し、payouts と照合して的中/外れ/PnLを表示。
HTML を開かずにターミナルだけで確認したい時用の軽量ツール。

使い方:
    python -m scripts.show_signals_today                  # 今日のみ
    python -m scripts.show_signals_today --date 2026-05-28
    python -m scripts.show_signals_today --days 7         # 過去7日（今日含む）の日別サマリ
    python -m scripts.show_signals_today --days 7 -v      # サマリ + 個別レース全表示
    python -m scripts.show_signals_today --days 7 --html  # HTMLでブラウザに表示
"""
from __future__ import annotations

import argparse
import webbrowser
from datetime import date, datetime, timedelta
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


_HTML_TEMPLATE = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>シグナル履歴 {period}</title>
<style>
:root {{
  --bg: #0f1117; --bg-card: #1a1d27; --bg-header: #1f2330; --bg-hover: #252937;
  --border: #2a2f3d; --text: #e4e6eb; --text-dim: #9aa0b0;
  --pos: #4ade80; --neg: #f87171; --warn: #fbbf24; --link: #60a5fa; --accent: #818cf8;
}}
* {{ box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Hiragino Sans", "Yu Gothic", sans-serif;
  max-width: 1280px; margin: 0 auto; padding: 24px 16px 60px;
  background: var(--bg); color: var(--text); font-size: 14px;
}}
h1 {{ font-size: 28px; margin: 0 0 4px; font-weight: 700; }}
h1 .accent {{ color: var(--accent); }}
.meta {{ color: var(--text-dim); font-size: 12px; margin-bottom: 24px; }}
.kpi {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 24px; }}
.kpi .card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
.kpi .label {{ color: var(--text-dim); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }}
.kpi .value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
.pos {{ color: var(--pos); }} .neg {{ color: var(--neg); }}
.section-title {{ font-size: 14px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.08em; margin: 28px 0 10px; font-weight: 600; }}
table {{ width: 100%; border-collapse: collapse; background: var(--bg-card); border-radius: 12px; overflow: hidden; }}
th, td {{ text-align: right; padding: 11px 14px; border-bottom: 1px solid var(--border); }}
th:nth-child(-n+2), td:nth-child(-n+2) {{ text-align: left; }}
th:last-child, td:last-child {{ text-align: center; }}
th {{ background: var(--bg-header); font-size: 11px; color: var(--text-dim); font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }}
tbody tr:last-child td {{ border-bottom: none; }}
tbody tr:hover td {{ background: var(--bg-hover); }}
.totals {{ background: var(--bg-header) !important; font-weight: 700; border-top: 2px solid var(--border); }}
.badge {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 11px; font-weight: 700; }}
.badge-hit {{ background: rgba(74,222,128,0.15); color: var(--pos); }}
.badge-miss {{ background: rgba(248,113,113,0.15); color: var(--neg); }}
.badge-pending {{ background: rgba(154,160,176,0.12); color: var(--text-dim); }}
</style></head><body>

<h1>シグナル履歴 <span class="accent">{period}</span></h1>
<div class="meta">生成: {generated} ｜ 戦略: <code>lane1_kelly</code> ｜ レース数: {n_races}件 (確定 {n_finished} / 未確定 {n_pending})</div>

<div class="kpi">
  <div class="card"><div class="label">勝率</div><div class="value">{hit_rate}</div></div>
  <div class="card"><div class="label">合計ステーク</div><div class="value">¥{stake_total}</div></div>
  <div class="card"><div class="label">合計PnL</div><div class="value {pnl_cls}">{pnl_sign}¥{pnl_total}</div></div>
  <div class="card"><div class="label">ROI</div><div class="value {pnl_cls}">{pnl_sign}{roi}</div></div>
</div>

{daily_section}

<div class="section-title">▼ 個別レース</div>
<table><thead><tr>
<th>日付</th><th>場</th><th>R</th><th>観測時間</th><th>最高EV</th><th>オッズ</th><th>ステーク</th><th>結果</th><th>PnL</th>
</tr></thead><tbody>{detail_rows}</tbody></table>

</body></html>
"""


def _to_html(df: pd.DataFrame, period: str) -> str:
    n = len(df)
    fin = int(df["finished"].sum())
    pending = n - fin
    hit = int(df["hit"].sum())
    stake_total = int(df[df["finished"]]["latest_stake"].sum())
    pnl_total = int(df["pnl"].sum())
    pnl_cls = "pos" if pnl_total > 0 else ("neg" if pnl_total < 0 else "")
    pnl_sign = "+" if pnl_total > 0 else ""
    hit_rate = f"{hit / fin * 100:.1f}%" if fin else "-"
    roi = f"{pnl_total / stake_total * 100:.1f}%" if stake_total else "-"

    # 日別サマリ（複数日の時のみ）
    daily_section = ""
    if df["date"].nunique() > 1:
        rows = []
        for d, g in df.groupby("date"):
            n_d = len(g)
            fin_d = int(g["finished"].sum())
            hit_d = int(g["hit"].sum())
            stake_d = int(g[g["finished"]]["latest_stake"].sum())
            pnl_d = int(g["pnl"].sum())
            rate_d = f"{hit_d / fin_d * 100:.1f}%" if fin_d else "-"
            roi_d = f"{pnl_d / stake_d * 100:+.1f}%" if stake_d else "-"
            cls = "pos" if pnl_d > 0 else ("neg" if pnl_d < 0 else "")
            sign = "+" if pnl_d > 0 else ""
            rows.append(
                f"<tr><td>{d}</td><td>{n_d}</td><td>{fin_d}</td><td>{hit_d}</td>"
                f"<td>{rate_d}</td><td>¥{stake_d:,}</td>"
                f"<td class='{cls}'>{sign}¥{pnl_d:,}</td>"
                f"<td class='{cls}'>{roi_d}</td></tr>"
            )
        daily_section = (
            '<div class="section-title">▼ 日別サマリ</div>'
            '<table style="margin-bottom:24px"><thead><tr>'
            '<th>日付</th><th>件数</th><th>確定</th><th>的中</th>'
            '<th>勝率</th><th>ステーク</th><th>PnL</th><th>ROI</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
        )

    # 個別レース行
    detail_rows: list[str] = []
    df_sorted = df.sort_values(["date", "max_ev"], ascending=[True, False])
    for _, r in df_sorted.iterrows():
        rid = str(r["race_id"])
        parts = rid.split("-")
        venue = VENUE_CODES.get(parts[1], "?") if len(parts) > 1 else "?"
        rno = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
        seen = f"{r['first_seen'].strftime('%H:%M')}〜{r['last_seen'].strftime('%H:%M')}"
        stake = int(r["latest_stake"]) if pd.notna(r["latest_stake"]) else 0
        odds = float(r["latest_odds"]) if pd.notna(r["latest_odds"]) else 0.0
        if r["finished"]:
            if r["hit"]:
                status = "<span class='badge badge-hit'>🟢 1着</span>"
                pnl_v = int(r["pnl"])
                pnl_html = f"<span class='pos'>+¥{pnl_v:,}</span>"
            else:
                status = f"<span class='badge badge-miss'>✕ {int(r['winner_lane'])}号艇1着</span>"
                pnl_html = f"<span class='neg'>-¥{stake:,}</span>"
        else:
            status = "<span class='badge badge-pending'>⏳ 未確定</span>"
            pnl_html = "-"
        detail_rows.append(
            "<tr>"
            f"<td>{r['date']}</td>"
            f"<td>{venue}({parts[1]})</td>"
            f"<td>{rno}R</td>"
            f"<td>{seen}</td>"
            f"<td>{r['max_ev']:.3f}</td>"
            f"<td>{odds:.2f}</td>"
            f"<td>¥{stake:,}</td>"
            f"<td>{status}</td>"
            f"<td>{pnl_html}</td>"
            "</tr>"
        )

    return _HTML_TEMPLATE.format(
        period=period,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_races=n,
        n_finished=fin,
        n_pending=pending,
        hit_rate=hit_rate,
        stake_total=f"{stake_total:,}",
        pnl_total=f"{abs(pnl_total):,}",
        pnl_sign=pnl_sign,
        pnl_cls=pnl_cls,
        roi=roi,
        daily_section=daily_section,
        detail_rows="".join(detail_rows),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", default=None, help="単日指定 YYYY-MM-DD")
    p.add_argument("--days", type=int, default=None,
                   help="過去N日（今日含む）の集計。指定時は日別サマリも表示")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="集計だけでなく個別レースも全表示")
    p.add_argument("--html", nargs="?", const="auto", default=None,
                   help="HTML を生成してブラウザで開く。任意で出力パスを指定可")
    p.add_argument("--no-open", action="store_true",
                   help="--html 時にブラウザを自動で開かない")
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

    # HTML 出力モード（ターミナル出力もする）
    if args.html is not None:
        if args.html == "auto":
            if len(targets) > 1:
                fname = f"signals_history_{targets[0]:%Y%m%d}_{targets[-1]:%Y%m%d}.html"
            else:
                fname = f"signals_history_{targets[0]:%Y%m%d}.html"
            html_path = PROCESSED_DIR / fname
        else:
            html_path = Path(args.html)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(_to_html(df, period_label), encoding="utf-8")
        print(f"HTML saved: {html_path}")
        print(f"  open: file://{html_path.resolve()}")
        if not args.no_open:
            try:
                webbrowser.open(html_path.resolve().as_uri())
            except Exception:
                pass
        return

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
