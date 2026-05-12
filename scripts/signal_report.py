"""bets CSV からシグナル向けの見やすいHTMLレポートを生成する。

バックテストの bets_*.csv を入力に、当日（or 任意日）の**推奨ベット一覧**を
ダーク UI の単一テーブルで出力する。run_signal パイプラインの最終ステップ。

backtest 由来の累積PnLチャート・オッズ帯別ヒストグラム等は省略し、
「どのレースの 1号艇に いくら賭けるか」がパッと分かる構成に絞る。

使い方:
    python -m scripts.signal_report --bets data/models/backtest/bets_lane1_kelly.csv \
        --out data/processed/signals.html
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils.config import BASE_URL, VENUE_CODES


HTML = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>当日シグナル {date_label}</title>
<style>
  :root {{
    --bg: #0f1117;
    --bg-card: #1a1d27;
    --bg-header: #1f2330;
    --bg-hover: #252937;
    --border: #2a2f3d;
    --text: #e4e6eb;
    --text-dim: #9aa0b0;
    --pos: #4ade80;
    --neg: #f87171;
    --warn: #fbbf24;
    --link: #60a5fa;
    --accent: #818cf8;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", "Hiragino Sans", "Yu Gothic", sans-serif;
    max-width: 1280px; margin: 0 auto; padding: 24px 16px 60px;
    background: var(--bg); color: var(--text); font-size: 14px;
  }}
  h1 {{ font-size: 28px; margin: 0 0 4px; font-weight: 700; letter-spacing: -0.01em; }}
  h1 .date {{ color: var(--accent); }}
  .meta {{ color: var(--text-dim); font-size: 12px; margin-bottom: 24px; }}
  .meta code {{ background: var(--bg-card); padding: 2px 6px; border-radius: 4px; }}

  .kpi {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .kpi .card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; }}
  .kpi .label {{ color: var(--text-dim); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }}
  .kpi .value {{ font-size: 26px; font-weight: 700; margin-top: 6px; line-height: 1.1; }}
  .kpi .value.big {{ font-size: 32px; }}
  .pos {{ color: var(--pos); }}
  .neg {{ color: var(--neg); }}

  .section-title {{ font-size: 14px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.08em; margin: 28px 0 10px; font-weight: 600; }}

  table {{ width: 100%; border-collapse: collapse; background: var(--bg-card); border-radius: 12px; overflow: hidden; }}
  th, td {{ text-align: right; padding: 12px 14px; border-bottom: 1px solid var(--border); }}
  th:nth-child(-n+2), td:nth-child(-n+2) {{ text-align: left; }}
  th:last-child, td:last-child {{ text-align: center; }}
  th {{ background: var(--bg-header); cursor: pointer; user-select: none; font-size: 11px; color: var(--text-dim); font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }}
  th:hover {{ background: var(--bg-hover); color: var(--text); }}
  tbody tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover td {{ background: var(--bg-hover); }}
  td.venue {{ font-weight: 600; }}
  td.race {{ font-weight: 700; color: var(--accent); font-size: 16px; }}
  td.lane {{ font-weight: 700; font-size: 18px; color: #fff; }}
  td.stake {{ font-weight: 700; font-size: 16px; }}

  .ev-high {{ color: var(--pos); font-weight: 700; }}
  .ev-mid {{ color: var(--warn); font-weight: 600; }}

  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 11px; font-weight: 700; letter-spacing: 0.02em; }}
  .badge-hit {{ background: rgba(74,222,128,0.15); color: var(--pos); }}
  .badge-miss {{ background: rgba(248,113,113,0.15); color: var(--neg); }}
  .badge-pending {{ background: rgba(154,160,176,0.12); color: var(--text-dim); }}

  a {{ color: var(--link); text-decoration: none; font-weight: 600; }}
  a:hover {{ text-decoration: underline; }}

  .totals {{ background: var(--bg-header) !important; font-weight: 700; border-top: 2px solid var(--border); }}
  .totals td {{ padding: 14px; }}
</style>
</head>
<body>

<h1>本日のシグナル <span class="date">{date_label}</span></h1>
<div class="meta">
  生成: {generated} &nbsp;｜&nbsp; 戦略: <code>lane1_kelly</code>（1号艇限定・EV>{ev_threshold}）&nbsp;｜&nbsp;
  全候補: {n_bets}件 ({n_pending}件は未開催/未確定)
</div>

<div class="kpi">
  <div class="card"><div class="label">推奨ベット</div><div class="value big">{n_bets} 件</div></div>
  <div class="card"><div class="label">合計ステーク</div><div class="value big">¥{total_stake}</div></div>
  <div class="card"><div class="label">最高 EV</div><div class="value">{max_ev}</div></div>
  <div class="card"><div class="label">平均オッズ</div><div class="value">{avg_odds}</div></div>
  {result_kpis}
</div>

<div class="section-title">▼ 賭けるレース一覧 — 全て 1号艇 単勝</div>
<table id="bets">
  <thead><tr>
    <th>場</th><th>R</th><th>1号艇</th>
    <th>p_blend</th><th>オッズ</th><th>EV</th><th>推奨ステーク</th>
    <th>結果</th><th>PnL</th><th>公式</th>
  </tr></thead>
  <tbody>{rows}{totals_row}</tbody>
</table>

<script>
document.querySelectorAll('#bets th').forEach((th, idx) => {{
  let asc = false;
  th.addEventListener('click', () => {{
    const tbody = th.closest('table').querySelector('tbody');
    const all = Array.from(tbody.querySelectorAll('tr'));
    const totals = all.filter(r => r.classList.contains('totals'));
    const rows = all.filter(r => !r.classList.contains('totals'));
    rows.sort((a, b) => {{
      const av = a.children[idx].dataset.sort ?? a.children[idx].textContent;
      const bv = b.children[idx].dataset.sort ?? b.children[idx].textContent;
      const an = parseFloat(av), bn = parseFloat(bv);
      const cmp = (!isNaN(an) && !isNaN(bn)) ? an - bn : String(av).localeCompare(String(bv), 'ja');
      return asc ? cmp : -cmp;
    }});
    asc = !asc;
    rows.forEach(r => tbody.appendChild(r));
    totals.forEach(r => tbody.appendChild(r));
  }});
}});
</script>
</body>
</html>
"""


def _ev_class(ev: float) -> str:
    if ev >= 1.30:
        return "ev-high"
    if ev >= 1.10:
        return "ev-mid"
    return ""


def _parse_race_id(rid: str) -> tuple[str, str, str]:
    """20260512-07-01 → ('20260512', '07', '01')"""
    parts = str(rid).split("-")
    if len(parts) != 3:
        return ("", "", "")
    return parts[0], parts[1], parts[2]


def _row(r: pd.Series) -> str:
    date_str, jcd, rno = _parse_race_id(r["race_id"])
    venue_name = VENUE_CODES.get(jcd, "?")
    odds = float(r.get("odds_win", 0)) if pd.notna(r.get("odds_win")) else 0.0
    p_blend = float(r["blended_win_prob"]) if "blended_win_prob" in r and pd.notna(r["blended_win_prob"]) else float(r.get("pred_win_prob", 0))
    ev = float(r.get("ev", odds * p_blend))
    stake = int(r["stake"])

    is_pending = ("race_finished" in r.index) and (not bool(r["race_finished"]))
    if is_pending:
        result_html = "<span class='badge badge-pending'>⏳ 未確定</span>"
        pnl_html = "-"
        pnl_sort = 0
    else:
        hit = bool(r["hit"])
        if hit:
            wl = int(r.get("winner_lane", 1)) if pd.notna(r.get("winner_lane")) else 1
            result_html = f"<span class='badge badge-hit'>🟢 1着</span>"
            pnl = int(r["pnl"])
            ret = int(r["return_yen"])
            pnl_html = f"<span class='pos'>+¥{pnl:,}</span><br><span style='color:var(--text-dim);font-size:11px'>返¥{ret:,}</span>"
            pnl_sort = pnl
        else:
            wl_v = r.get("winner_lane")
            wl_text = f"{int(wl_v)}号艇" if pd.notna(wl_v) else "外れ"
            result_html = f"<span class='badge badge-miss'>✕ {wl_text}1着</span>"
            pnl = int(r["pnl"])
            pnl_html = f"<span class='neg'>¥{pnl:,}</span>"
            pnl_sort = pnl

    url = f"{BASE_URL}/oddstf?rno={int(rno)}&jcd={jcd}&hd={date_str}"
    ev_cls = _ev_class(ev)

    return (
        f"<tr>"
        f"<td class='venue'>{venue_name}({jcd})</td>"
        f"<td class='race' data-sort='{int(rno)}'>{int(rno)}R</td>"
        f"<td class='lane'>1</td>"
        f"<td>{p_blend:.3f}</td>"
        f"<td>{odds:.2f}</td>"
        f"<td class='{ev_cls}'>{ev:.3f}</td>"
        f"<td class='stake'>¥{stake:,}</td>"
        f"<td>{result_html}</td>"
        f"<td data-sort='{pnl_sort}'>{pnl_html}</td>"
        f"<td><a href='{url}' target='_blank'>開く</a></td>"
        f"</tr>"
    )


def _totals_row(bets: pd.DataFrame) -> str:
    if "race_finished" not in bets.columns:
        decided = bets[bets["hit"].notna()]
    else:
        decided = bets[bets["race_finished"]]
    if not len(decided):
        return ""
    n = len(decided)
    n_hit = int(decided["hit"].sum())
    stake = int(decided["stake"].sum())
    pnl = int(decided["pnl"].sum())
    ret = stake + pnl
    cls = "pos" if pnl > 0 else ("neg" if pnl < 0 else "")
    sign = "+" if pnl > 0 else ""
    return (
        f"<tr class='totals'>"
        f"<td colspan='6'>確定済み合計（{n}件・的中{n_hit}件・勝率{n_hit/n*100:.0f}%）</td>"
        f"<td>¥{stake:,}</td>"
        f"<td></td>"
        f"<td class='{cls}'>{sign}¥{pnl:,}<br><span style='color:var(--text-dim);font-size:11px;font-weight:400'>返¥{ret:,}</span></td>"
        f"<td></td>"
        f"</tr>"
    )


def _result_kpis(bets: pd.DataFrame) -> str:
    """確定済み分の集計KPIカード。全件未確定なら空文字。"""
    if "race_finished" not in bets.columns:
        decided = bets
    else:
        decided = bets[bets["race_finished"]]
    if not len(decided):
        return ""
    n = len(decided)
    n_hit = int(decided["hit"].sum())
    stake = int(decided["stake"].sum())
    pnl = int(decided["pnl"].sum())
    roi = (pnl / stake * 100) if stake else 0.0
    pnl_cls = "pos" if pnl > 0 else ("neg" if pnl < 0 else "")
    sign = "+" if pnl > 0 else ""
    return (
        f"<div class='card'><div class='label'>確定済み</div>"
        f"<div class='value'>{n} 件</div></div>"
        f"<div class='card'><div class='label'>実勝率</div>"
        f"<div class='value'>{n_hit/n*100:.0f}%</div></div>"
        f"<div class='card'><div class='label'>実 PnL</div>"
        f"<div class='value {pnl_cls}'>{sign}¥{pnl:,}</div></div>"
        f"<div class='card'><div class='label'>実 ROI</div>"
        f"<div class='value {pnl_cls}'>{sign}{roi:.0f}%</div></div>"
    )


def _summary(bets: pd.DataFrame) -> dict:
    total_stake = int(bets["stake"].sum())
    max_ev = float(bets.get("ev", pd.Series([0])).max()) if "ev" in bets.columns else 0.0
    if max_ev == 0 and "blended_win_prob" in bets.columns and "odds_win" in bets.columns:
        max_ev = float((bets["blended_win_prob"] * bets["odds_win"]).max())
    avg_odds = float(bets["odds_win"].mean()) if "odds_win" in bets.columns else 0.0
    return {
        "n_bets": len(bets),
        "n_pending": int((~bets["race_finished"]).sum()) if "race_finished" in bets.columns else 0,
        "total_stake": total_stake,
        "max_ev": max_ev,
        "avg_odds": avg_odds,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bets", required=True)
    p.add_argument("--out", default="data/processed/signals.html")
    p.add_argument("--ev-threshold", default="1.05",
                   help="表示用のEV閾値ラベル（描画のみ。実際の閾値はbacktestで決まる）")
    p.add_argument("--date-label", default=None,
                   help="ヘッダの日付ラベル。未指定なら CSV から自動抽出")
    args = p.parse_args()

    bets = pd.read_csv(args.bets)
    bets["race_date"] = pd.to_datetime(bets["race_date"]).dt.strftime("%Y-%m-%d")

    # 日付ラベル
    if args.date_label:
        date_label = args.date_label
    elif len(bets):
        unique_dates = sorted(bets["race_date"].unique())
        date_label = unique_dates[0] if len(unique_dates) == 1 else f"{unique_dates[0]} 〜 {unique_dates[-1]}"
    else:
        date_label = "(対象レースなし)"

    # EV を計算（無ければ blend * odds）
    if "ev" not in bets.columns and "blended_win_prob" in bets.columns:
        bets["ev"] = bets["blended_win_prob"] * bets["odds_win"]

    # EV 降順で並べる（妙味の大きい順 = 優先したいレース）
    bets = bets.sort_values(["ev" if "ev" in bets.columns else "stake"], ascending=False)

    summ = _summary(bets)
    html = HTML.format(
        date_label=date_label,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        ev_threshold=args.ev_threshold,
        n_bets=summ["n_bets"],
        n_pending=summ["n_pending"],
        total_stake=f"{summ['total_stake']:,}",
        max_ev=f"{summ['max_ev']:.3f}" if summ["max_ev"] else "-",
        avg_odds=f"{summ['avg_odds']:.2f}" if summ["avg_odds"] else "-",
        result_kpis=_result_kpis(bets),
        rows="".join(_row(r) for _, r in bets.iterrows()),
        totals_row=_totals_row(bets),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"saved: {out.resolve()}")
    print(f"open in browser: file://{out.resolve()}")


if __name__ == "__main__":
    main()
