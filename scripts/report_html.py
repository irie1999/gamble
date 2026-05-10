"""バックテストの bets_*.csv を読み、自己完結HTMLレポートを生成する。

CDN の Chart.js を使うので Python 側にプロット依存は無し。
出力ファイルはそのままブラウザで開ける。

使い方:
    # まずバックテストを走らせて bets CSV を吐き出す
    python -m scripts.backtest --strategy lane1_kelly --since 2026-03-01 \
        --exclude-venues 24 --max-odds 10.0 --ev-threshold 1.05 ...

    # それを HTML にする
    python -m scripts.report_html \
        --bets data/models/backtest/bets_lane1_kelly.csv \
        --out data/processed/report.html

ブラウザで report.html を開けば、累積PnL・オッズ帯別・場別の集計が見える。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import VENUE_CODES


def _summary(bets: pd.DataFrame) -> dict:
    n = len(bets)
    if n == 0:
        return {}
    wins = int(bets["hit"].sum())
    stake = float(bets["stake"].sum())
    ret = float(bets["return_yen"].sum())
    pnl = ret - stake
    avg_odds = float(bets["odds_win"].mean()) if "odds_win" in bets.columns else float("nan")
    return {
        "n_bets": n,
        "wins": wins,
        "hit_rate": wins / n,
        "stake": stake,
        "return": ret,
        "pnl": pnl,
        "roi": pnl / stake if stake > 0 else 0.0,
        "avg_odds": avg_odds,
    }


def _cumulative_pnl(bets: pd.DataFrame) -> tuple[list[str], list[float]]:
    bets = bets.sort_values("race_date").reset_index(drop=True)
    cum = (bets["return_yen"] - bets["stake"]).cumsum()
    return bets["race_date"].astype(str).tolist(), cum.round(0).tolist()


def _by_odds_bucket(bets: pd.DataFrame) -> list[dict]:
    if "odds_win" not in bets.columns or bets["odds_win"].isna().all():
        return []
    bins = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 100.0]
    labels = ["1-2", "2-3", "3-5", "5-7", "7-10", "10+"]
    bets = bets.copy()
    bets["bucket"] = pd.cut(bets["odds_win"], bins=bins, labels=labels, include_lowest=True)
    out = []
    for label in labels:
        sub = bets[bets["bucket"] == label]
        if not len(sub):
            continue
        s = float(sub["stake"].sum())
        r = float(sub["return_yen"].sum())
        out.append({
            "bucket": label,
            "n": int(len(sub)),
            "hit_rate": float(sub["hit"].mean()),
            "roi": (r - s) / s if s > 0 else 0,
            "pnl": int(r - s),
        })
    return out


def _by_venue(bets: pd.DataFrame) -> list[dict]:
    bets = bets.copy()
    bets["venue"] = bets["race_id"].astype(str).str.split("-").str[1]
    out = []
    for venue, sub in bets.groupby("venue"):
        s = float(sub["stake"].sum())
        r = float(sub["return_yen"].sum())
        out.append({
            "venue": venue,
            "name": VENUE_CODES.get(venue, "?"),
            "n": int(len(sub)),
            "hit_rate": float(sub["hit"].mean()),
            "roi": (r - s) / s if s > 0 else 0,
            "pnl": int(r - s),
        })
    return sorted(out, key=lambda d: -d["pnl"])


HTML_TEMPLATE = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>Backtest Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 1100px; margin: 24px auto; padding: 0 16px; color: #1a1a1a; }}
  h1 {{ font-size: 22px; }}
  h2 {{ font-size: 17px; margin-top: 32px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  .meta {{ color: #666; font-size: 13px; }}
  .kpi {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 16px 0; }}
  .kpi .card {{ background: #f7f7f9; border-radius: 8px; padding: 12px; }}
  .kpi .label {{ color: #666; font-size: 12px; }}
  .kpi .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
  .pos {{ color: #1f883d; }}
  .neg {{ color: #c0392b; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: right; padding: 6px 10px; border-bottom: 1px solid #eee; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ background: #f7f7f9; }}
  canvas {{ max-height: 300px; }}
  .chart-wrap {{ background: #fff; border: 1px solid #eee; border-radius: 8px; padding: 12px; margin: 12px 0; }}
</style>
</head>
<body>

<h1>Backtest Report</h1>
<div class="meta">
  source: <code>{source}</code><br>
  generated: {generated} &nbsp;|&nbsp; bets: {n_bets} &nbsp;|&nbsp; period: {date_min} → {date_max}
</div>

<h2>サマリ</h2>
<div class="kpi">
  <div class="card"><div class="label">n_bets</div><div class="value">{n_bets}</div></div>
  <div class="card"><div class="label">勝率</div><div class="value">{hit_rate}</div></div>
  <div class="card"><div class="label">ROI</div><div class="value {roi_cls}">{roi}</div></div>
  <div class="card"><div class="label">PnL</div><div class="value {pnl_cls}">¥{pnl}</div></div>
  <div class="card"><div class="label">投入額</div><div class="value">¥{stake}</div></div>
  <div class="card"><div class="label">平均オッズ</div><div class="value">{avg_odds}</div></div>
</div>

<h2>累積 PnL</h2>
<div class="chart-wrap"><canvas id="cumPnl"></canvas></div>

<h2>オッズ帯別</h2>
<div class="chart-wrap"><canvas id="byOdds"></canvas></div>
<table>
  <thead><tr><th>レンジ</th><th>n</th><th>勝率</th><th>ROI</th><th>PnL</th></tr></thead>
  <tbody>{rows_odds}</tbody>
</table>

<h2>場別</h2>
<div class="chart-wrap"><canvas id="byVenue"></canvas></div>
<table>
  <thead><tr><th>場</th><th>名前</th><th>n</th><th>勝率</th><th>ROI</th><th>PnL</th></tr></thead>
  <tbody>{rows_venue}</tbody>
</table>

<h2>個別ベット（全{n_bets}件・新しい順）</h2>
<table>
  <thead><tr><th>日付</th><th>レース</th><th>オッズ</th><th>p_blend</th><th>EV</th><th>stake</th><th>結果</th><th>PnL</th></tr></thead>
  <tbody>{rows_all}</tbody>
</table>

<script>
const data = {data_json};

new Chart(document.getElementById('cumPnl'), {{
  type: 'line',
  data: {{
    labels: data.cumPnl.dates,
    datasets: [{{ label: '累積PnL (¥)', data: data.cumPnl.values, borderColor: '#1f6feb', backgroundColor: 'rgba(31,111,235,0.08)', fill: true, tension: 0.1, pointRadius: 0 }}]
  }},
  options: {{ responsive: true, scales: {{ x: {{ ticks: {{ maxTicksLimit: 12 }} }} }} }}
}});

new Chart(document.getElementById('byOdds'), {{
  type: 'bar',
  data: {{
    labels: data.byOdds.map(d => d.bucket),
    datasets: [
      {{ label: 'ROI', data: data.byOdds.map(d => d.roi), backgroundColor: data.byOdds.map(d => d.roi >= 0 ? '#1f883d' : '#c0392b'), yAxisID: 'y' }},
      {{ label: 'n_bets', data: data.byOdds.map(d => d.n), type: 'line', borderColor: '#666', yAxisID: 'y1', tension: 0 }}
    ]
  }},
  options: {{ scales: {{ y: {{ position: 'left', title: {{ display: true, text: 'ROI' }} }}, y1: {{ position: 'right', grid: {{ drawOnChartArea: false }}, title: {{ display: true, text: 'n_bets' }} }} }} }}
}});

new Chart(document.getElementById('byVenue'), {{
  type: 'bar',
  data: {{
    labels: data.byVenue.map(d => d.name + '(' + d.venue + ')'),
    datasets: [{{ label: 'PnL (¥)', data: data.byVenue.map(d => d.pnl), backgroundColor: data.byVenue.map(d => d.pnl >= 0 ? '#1f883d' : '#c0392b') }}]
  }},
  options: {{ indexAxis: 'y' }}
}});
</script>

</body>
</html>
"""


def _fmt_pct(x: float) -> str:
    return f"{x*100:+.1f}%"


def _fmt_yen(x: float) -> str:
    return f"{int(x):,}"


def _row_odds(b: dict) -> str:
    cls = "pos" if b["roi"] >= 0 else "neg"
    return (f"<tr><td>{b['bucket']}</td><td>{b['n']}</td>"
            f"<td>{b['hit_rate']*100:.1f}%</td>"
            f"<td class='{cls}'>{_fmt_pct(b['roi'])}</td>"
            f"<td class='{cls}'>¥{_fmt_yen(b['pnl'])}</td></tr>")


def _row_venue(b: dict) -> str:
    cls = "pos" if b["pnl"] >= 0 else "neg"
    return (f"<tr><td>{b['venue']}</td><td>{b['name']}</td><td>{b['n']}</td>"
            f"<td>{b['hit_rate']*100:.1f}%</td>"
            f"<td class='{cls}'>{_fmt_pct(b['roi'])}</td>"
            f"<td class='{cls}'>¥{_fmt_yen(b['pnl'])}</td></tr>")


def _cell(value, fmt: str = "{}") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "<td>-</td>"
    return f"<td>{fmt.format(value)}</td>"


def _row_recent(r: pd.Series) -> str:
    cls = "pos" if r["pnl"] >= 0 else "neg"
    venue = str(r["race_id"]).split("-")[1] if "-" in str(r["race_id"]) else "?"
    result = "🟢 的中" if r["hit"] else "✕ 不的中"
    odds = r.get("odds_win") if "odds_win" in r else None
    p_blend = r.get("blended_win_prob") if "blended_win_prob" in r else r.get("pred_win_prob")
    ev = r.get("ev") if "ev" in r else None
    return (f"<tr><td>{r['race_date']}</td>"
            f"<td>{venue}/{r['race_id']}</td>"
            f"{_cell(odds, '{:.2f}')}"
            f"{_cell(p_blend, '{:.3f}')}"
            f"{_cell(ev, '{:.3f}')}"
            f"<td>¥{_fmt_yen(r['stake'])}</td>"
            f"<td>{result}</td>"
            f"<td class='{cls}'>¥{_fmt_yen(r['pnl'])}</td></tr>")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bets", required=True)
    p.add_argument("--out", default="data/processed/report.html")
    args = p.parse_args()

    bets = pd.read_csv(args.bets)
    bets["race_date"] = pd.to_datetime(bets["race_date"]).dt.strftime("%Y-%m-%d")

    summ = _summary(bets)
    cum_dates, cum_values = _cumulative_pnl(bets)
    by_odds = _by_odds_bucket(bets)
    by_venue = _by_venue(bets)

    all_bets = bets.sort_values("race_date", ascending=False)

    payload = {
        "cumPnl": {"dates": cum_dates, "values": cum_values},
        "byOdds": by_odds,
        "byVenue": by_venue,
    }

    html = HTML_TEMPLATE.format(
        source=Path(args.bets).name,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_bets=summ["n_bets"],
        date_min=bets["race_date"].min(),
        date_max=bets["race_date"].max(),
        hit_rate=f"{summ['hit_rate']*100:.1f}%",
        roi=_fmt_pct(summ["roi"]),
        roi_cls="pos" if summ["roi"] >= 0 else "neg",
        pnl=_fmt_yen(summ["pnl"]),
        pnl_cls="pos" if summ["pnl"] >= 0 else "neg",
        stake=_fmt_yen(summ["stake"]),
        avg_odds=f"{summ['avg_odds']:.2f}" if not pd.isna(summ["avg_odds"]) else "-",
        rows_odds="".join(_row_odds(b) for b in by_odds),
        rows_venue="".join(_row_venue(b) for b in by_venue),
        rows_all="".join(_row_recent(r) for _, r in all_bets.iterrows()),
        data_json=json.dumps(payload, ensure_ascii=False),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"saved: {out.resolve()}")
    print(f"open in browser: file://{out.resolve()}")


if __name__ == "__main__":
    main()
