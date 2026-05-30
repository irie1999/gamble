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
import json
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
  .badge-active {{ background: rgba(96,165,250,0.15); color: var(--link); }}
  .badge-dropped {{ background: rgba(154,160,176,0.10); color: var(--text-dim); }}
  .badge-rec {{ background: rgba(74,222,128,0.18); color: var(--pos); font-weight: 700; }}
  .badge-wait {{ background: rgba(251,191,36,0.18); color: var(--warn); }}

  a {{ color: var(--link); text-decoration: none; font-weight: 600; }}
  a:hover {{ text-decoration: underline; }}

  .totals {{ background: var(--bg-header) !important; font-weight: 700; border-top: 2px solid var(--border); }}
  .totals td {{ padding: 14px; }}

  .history-card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; margin-bottom: 24px; }}
  .history-card .equity-wrap {{ margin-top: 14px; background: var(--bg); border-radius: 8px; padding: 8px; }}
  .history-card .equity-wrap svg {{ display: block; width: 100%; height: auto; }}
  .history-meta {{ color: var(--text-dim); font-size: 12px; margin-top: 4px; }}
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

{signal_log_section}

{history_section}

<div class="section-title">▼ 賭けるレース一覧 — 全て 1号艇 単勝</div>
<table id="bets">
  <thead><tr>
    <th>場</th><th>R</th><th>締切</th><th>1号艇</th>
    <th>p_blend</th><th>オッズ</th><th>EV</th><th>判定</th><th>推奨ステーク</th>
    <th>結果</th><th>PnL</th><th>公式</th>
  </tr></thead>
  <tbody>{rows}{totals_row}</tbody>
</table>

<script>
document.querySelectorAll('table').forEach(table => {{
  const ths = table.querySelectorAll('thead th');
  ths.forEach((th, idx) => {{
    let asc = false;
    th.addEventListener('click', () => {{
      const tbody = table.querySelector('tbody');
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


def _minutes_to_deadline_now(deadline_str: str, date_str: str) -> float | None:
    """deadline ("HH:MM") と date ("YYYYMMDD") から 今 までの分数を返す。"""
    if not deadline_str or ":" not in deadline_str or len(date_str) != 8:
        return None
    try:
        hh, mm = deadline_str.split(":")[:2]
        d = datetime.strptime(date_str, "%Y%m%d").date()
        deadline = datetime.combine(d, datetime.min.time().replace(hour=int(hh), minute=int(mm)))
    except (ValueError, AttributeError):
        return None
    return (deadline - datetime.now()).total_seconds() / 60.0


def _classify_recommendation(ev: float, minutes_to_deadline: float | None) -> str:
    """シグナルを「賭ける」か「様子見」か判定。

    Returns:
        "recommended": 締切10分以内 → 投票推奨
        "wait":        まだ時間あり、オッズ動く可能性 → 観察のみ
        "past":        既に締切過ぎ
    """
    if minutes_to_deadline is None:
        return "wait"
    if minutes_to_deadline < 0:
        return "past"
    if minutes_to_deadline <= 10:
        return "recommended"
    return "wait"


def _parse_race_id(rid: str) -> tuple[str, str, str]:
    """20260512-07-01 → ('20260512', '07', '01')"""
    parts = str(rid).split("-")
    if len(parts) != 3:
        return ("", "", "")
    return parts[0], parts[1], parts[2]


def _row(r: pd.Series, schedule: dict) -> str:
    date_str, jcd, rno = _parse_race_id(r["race_id"])
    venue_name = VENUE_CODES.get(jcd, "?")
    odds_pre = float(r.get("odds_win", 0)) if pd.notna(r.get("odds_win")) else 0.0
    p_blend = float(r["blended_win_prob"]) if "blended_win_prob" in r and pd.notna(r["blended_win_prob"]) else float(r.get("pred_win_prob", 0))
    ev = float(r.get("ev", odds_pre * p_blend))
    stake = int(r["stake"])

    deadline = schedule.get(r["race_id"], "")
    deadline_html = f"<td class='deadline'>{deadline}</td>" if deadline else "<td class='deadline'>-</td>"

    # オッズ表示: 確定オッズがあれば併記（ライブ朝→締切時の動きを見える化）
    settled = r.get("settled_odds")
    has_settled = pd.notna(settled) if settled is not None else False
    if has_settled and abs(float(settled) - odds_pre) > 0.05:
        # 朝のオッズと確定で違いあり → 併記
        odds_html = (f"<td>{odds_pre:.2f}<br>"
                     f"<span style='color:var(--text-dim);font-size:11px'>確定 {float(settled):.2f}</span></td>")
    else:
        odds_html = f"<td>{odds_pre:.2f}</td>"

    is_pending = ("race_finished" in r.index) and (not bool(r["race_finished"]))
    if is_pending:
        result_html = "<span class='badge badge-pending'>⏳ 未確定</span>"
        pnl_html = "-"
        pnl_sort = 0
    else:
        hit = bool(r["hit"])
        if hit:
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

    # 判定セル: 賭ける/様子見/締切過ぎ
    mins_left = _minutes_to_deadline_now(deadline, date_str)
    rec = _classify_recommendation(ev, mins_left)
    if is_pending:
        if rec == "recommended":
            rec_html = "<td><span class='badge badge-rec'>✅ 賭ける</span></td>"
        elif rec == "wait":
            rec_html = "<td><span class='badge badge-wait'>👁 様子見</span></td>"
        else:
            rec_html = "<td><span class='badge badge-pending'>締切過ぎ</span></td>"
    else:
        rec_html = "<td><span class='badge badge-pending'>確定</span></td>"

    return (
        f"<tr>"
        f"<td class='venue'>{venue_name}({jcd})</td>"
        f"<td class='race' data-sort='{int(rno)}'>{int(rno)}R</td>"
        f"{deadline_html}"
        f"<td class='lane'>1</td>"
        f"<td>{p_blend:.3f}</td>"
        f"{odds_html}"
        f"<td class='{ev_cls}'>{ev:.3f}</td>"
        f"{rec_html}"
        f"<td class='stake'>¥{stake:,}</td>"
        f"<td>{result_html}</td>"
        f"<td data-sort='{pnl_sort}'>{pnl_html}</td>"
        f"<td><a href='{url}' target='_blank'>開く</a></td>"
        f"</tr>"
    )


def _totals_row(bets: pd.DataFrame) -> str:
    if bets.empty or "hit" not in bets.columns:
        return ""
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
        # colspan は 12列の左8列を埋める（場/R/締切/1号艇/p_blend/オッズ/EV/判定）
        f"<tr class='totals'>"
        f"<td colspan='8'>確定済み合計（{n}件・的中{n_hit}件・勝率{n_hit/n*100:.0f}%）</td>"
        f"<td>¥{stake:,}</td>"
        f"<td></td>"
        f"<td class='{cls}'>{sign}¥{pnl:,}<br><span style='color:var(--text-dim);font-size:11px;font-weight:400'>返¥{ret:,}</span></td>"
        f"<td></td>"
        f"</tr>"
    )


def _result_kpis(bets: pd.DataFrame) -> str:
    """確定済み分の集計KPIカード。全件未確定なら空文字。"""
    if bets.empty or "hit" not in bets.columns:
        return ""
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
    if bets.empty:
        return {"n_bets": 0, "n_pending": 0, "total_stake": 0,
                "max_ev": 0.0, "avg_odds": 0.0}
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


def _load_schedule(schedule_path: Path) -> dict:
    """race_schedule.csv → {race_id: deadline_time}"""
    if not schedule_path.exists():
        return {}
    df = pd.read_csv(schedule_path)
    return dict(zip(df["race_id"].astype(str), df["deadline_time"].astype(str)))


def _load_equity_points(equity_path: Path) -> list[float]:
    """backtest が書いた equity_<strategy>.csv を読み、エクイティ値のリストを返す。

    Series.to_csv で出力された CSV はヘッダ行 + (index, value) の2列形式。
    値カラム名は固定ではない（Series.name が無いと空文字や "0"）ので、最後の列を使う。
    """
    if not equity_path.exists():
        return []
    try:
        df = pd.read_csv(equity_path)
    except Exception:
        return []
    if df.empty:
        return []
    last_col = df.columns[-1]
    return pd.to_numeric(df[last_col], errors="coerce").dropna().tolist()


def _equity_svg(values: list[float], initial: float) -> str:
    """エクイティ推移を SVG 折れ線で描画。初期バンクロールを点線で示す。"""
    if not values:
        return ""
    w, h = 720, 200
    pad_l, pad_r, pad_t, pad_b = 56, 56, 22, 22
    inner_w = w - pad_l - pad_r
    inner_h = h - pad_t - pad_b

    n = len(values)
    y_lo = min(min(values), initial)
    y_hi = max(max(values), initial)
    if y_hi == y_lo:
        y_hi = y_lo + 1

    def sx(i: int) -> float:
        return pad_l + (inner_w * i / max(1, n - 1))

    def sy(v: float) -> float:
        return pad_t + inner_h - (inner_h * (v - y_lo) / (y_hi - y_lo))

    points = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(values))
    final = values[-1]
    above = final >= initial
    color = "#4ade80" if above else "#f87171"
    fill_color = "rgba(74,222,128,0.10)" if above else "rgba(248,113,113,0.10)"

    base_y = sy(initial)
    final_y = sy(final)

    # 簡易な縦軸ラベル（上端・初期・下端）
    def _yen(v: float) -> str:
        return f"¥{int(v):,}"

    polygon = f"{pad_l:.1f},{(pad_t+inner_h):.1f} {points} {(pad_l+inner_w):.1f},{(pad_t+inner_h):.1f}"

    return (
        f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="エクイティカーブ">'
        f'<polygon points="{polygon}" fill="{fill_color}" stroke="none"/>'
        f'<line x1="{pad_l}" y1="{base_y:.1f}" x2="{pad_l+inner_w}" y2="{base_y:.1f}" '
        f'stroke="#3a3f4d" stroke-dasharray="4,4"/>'
        f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
        f'<circle cx="{sx(n-1):.1f}" cy="{final_y:.1f}" r="3.5" fill="{color}"/>'
        f'<text x="{pad_l-8}" y="{pad_t+6}" text-anchor="end" fill="#9aa0b0" font-size="10">{_yen(y_hi)}</text>'
        f'<text x="{pad_l-8}" y="{base_y+3}" text-anchor="end" fill="#9aa0b0" font-size="10">{_yen(initial)}</text>'
        f'<text x="{pad_l-8}" y="{pad_t+inner_h+4}" text-anchor="end" fill="#9aa0b0" font-size="10">{_yen(y_lo)}</text>'
        f'<text x="{pad_l+inner_w+6}" y="{final_y+4}" fill="{color}" font-size="11" font-weight="700">{_yen(final)}</text>'
        f'</svg>'
    )


def _render_history_section(summary_path: Path, equity_path: Path,
                            since: str, until: str,
                            actual_since: str = "", actual_until: str = "") -> str:
    """過去バックテスト集計セクションのHTMLを返す。データ不足なら空文字。

    since/until は要求された期間。actual_since/actual_until が指定されており、
    要求より狭ければ「実データ範囲が短い」旨の注釈を出す。
    """
    if not summary_path.exists():
        return ""
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    n_bets = int(summary.get("n_bets", 0) or 0)
    if n_bets <= 0:
        return ""

    stake = float(summary.get("stake_total", 0) or 0)
    pnl = float(summary.get("pnl", 0) or 0)
    roi = float(summary.get("roi", 0) or 0)
    hit = float(summary.get("hit_rate", 0) or 0)
    max_dd = float(summary.get("max_drawdown", 0) or 0)
    ending = float(summary.get("ending_bankroll", 0) or 0)
    initial = ending - pnl if pnl != 0 else max(1.0, ending)
    pnl_cls = "pos" if pnl > 0 else ("neg" if pnl < 0 else "")
    roi_cls = pnl_cls
    sign = "+" if pnl > 0 else ""

    equity_values = _load_equity_points(equity_path)
    svg = _equity_svg(equity_values, initial) if equity_values else ""

    # 実データ範囲と要求範囲がズレている場合、注釈を出す
    display_since, display_until = (actual_since or since), (actual_until or until)
    coverage_note = ""
    if actual_since and actual_until and since and until:
        if actual_since > since or actual_until < until:
            coverage_note = (
                f'<div class="history-meta" style="color:var(--warn);margin-top:6px">'
                f'⚠ 要求期間 <code>{since}</code> 〜 <code>{until}</code> のうち、'
                f'実データが揃っているのは上記範囲のみ。'
                f'過去データを追加スクレイプすると延長できます'
                f'（<code>scripts.scrape_dataset / scripts.scrape_odds</code>）。'
                f'</div>'
            )

    return (
        '<div class="section-title">▼ 過去バックテスト</div>'
        '<div class="history-card">'
        f'<div class="history-meta">期間: <code>{display_since}</code> 〜 <code>{display_until}</code>'
        f' &nbsp;｜&nbsp; 戦略: <code>lane1_kelly</code>'
        f' &nbsp;｜&nbsp; 初期バンクロール ¥{int(initial):,}</div>'
        + coverage_note +
        '<div class="kpi" style="margin-top:14px">'
        f'<div class="card"><div class="label">対象ベット数</div><div class="value">{n_bets:,} 件</div></div>'
        f'<div class="card"><div class="label">合計ステーク</div><div class="value">¥{int(stake):,}</div></div>'
        f'<div class="card"><div class="label">累積 PnL</div><div class="value {pnl_cls}">{sign}¥{int(pnl):,}</div></div>'
        f'<div class="card"><div class="label">ROI</div><div class="value {roi_cls}">{sign}{roi*100:.1f}%</div></div>'
        f'<div class="card"><div class="label">勝率</div><div class="value">{hit*100:.1f}%</div></div>'
        f'<div class="card"><div class="label">最大DD</div><div class="value neg">{max_dd*100:.1f}%</div></div>'
        f'<div class="card"><div class="label">最終バンクロール</div><div class="value">¥{int(ending):,}</div></div>'
        '</div>'
        + (f'<div class="equity-wrap">{svg}</div>' if svg else "")
        + '</div>'
    )


def _render_history_bets_section(bets_path: Path) -> str:
    """過去バックテストの個別取引テーブル。確定済みのみ・日付降順で表示。"""
    if not bets_path.exists():
        return ""
    try:
        df = pd.read_csv(bets_path)
    except Exception:
        return ""
    if df.empty:
        return ""

    if "race_finished" in df.columns:
        df = df[df["race_finished"].fillna(False).astype(bool)].copy()
    if df.empty:
        return ""

    # 並び替え用に race_date を確保（無ければ race_id の先頭8文字から）
    if "race_date" not in df.columns or df["race_date"].isna().all():
        df["race_date"] = df["race_id"].astype(str).str[:8]
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")

    # 時系列で累計PnLを出すため昇順で累積 → 表示時に降順に並べる
    sort_cols = ["race_date", "race_id"]
    df = df.sort_values([c for c in sort_cols if c in df.columns]).reset_index(drop=True)
    df["cum_pnl"] = df["pnl"].cumsum()
    df = df.iloc[::-1].reset_index(drop=True)  # 新しい順で表示

    rows_html: list[str] = []
    for _, r in df.iterrows():
        rid = str(r["race_id"])
        date_str, jcd, rno = _parse_race_id(rid)
        venue = VENUE_CODES.get(jcd, "?")
        d_disp = (f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}" if len(date_str) == 8 else "-")

        odds_pre = float(r.get("odds_win", 0) or 0)
        settled_v = r.get("settled_odds")
        has_settled = pd.notna(settled_v)
        if has_settled and abs(float(settled_v) - odds_pre) > 0.05:
            odds_cell = (f"{odds_pre:.2f}"
                         f"<br><span style='color:var(--text-dim);font-size:11px'>"
                         f"確定 {float(settled_v):.2f}</span>")
        else:
            odds_cell = f"{odds_pre:.2f}"

        ev = float(r.get("ev", 0) or 0)
        stake = int(r.get("stake", 0) or 0)
        pnl = int(r.get("pnl", 0) or 0)
        cum = int(r.get("cum_pnl", 0) or 0)

        hit = bool(r.get("hit", False))
        if hit:
            result_html = "<span class='badge badge-hit'>🟢 1着</span>"
            ret = int(r.get("return_yen", 0) or 0)
            pnl_html = (f"<span class='pos'>+¥{pnl:,}</span><br>"
                        f"<span style='color:var(--text-dim);font-size:11px'>返¥{ret:,}</span>")
        else:
            wl_v = r.get("winner_lane")
            wl_text = f"{int(wl_v)}号艇1着" if pd.notna(wl_v) else "外れ"
            result_html = f"<span class='badge badge-miss'>✕ {wl_text}</span>"
            pnl_html = f"<span class='neg'>¥{pnl:,}</span>"

        cum_cls = "pos" if cum > 0 else ("neg" if cum < 0 else "")
        cum_sign = "+" if cum > 0 else ""

        rows_html.append(
            "<tr>"
            f"<td data-sort='{date_str}'>{d_disp}</td>"
            f"<td class='venue'>{venue}({jcd})</td>"
            f"<td class='race' data-sort='{int(rno) if rno.isdigit() else 0}'>{int(rno) if rno.isdigit() else rno}R</td>"
            f"<td>{odds_cell}</td>"
            f"<td class='{_ev_class(ev)}'>{ev:.3f}</td>"
            f"<td class='stake'>¥{stake:,}</td>"
            f"<td>{result_html}</td>"
            f"<td data-sort='{pnl}'>{pnl_html}</td>"
            f"<td class='{cum_cls}' data-sort='{cum}'>{cum_sign}¥{cum:,}</td>"
            "</tr>"
        )

    n = len(df)
    return (
        '<details class="history-bets" open style="margin-bottom:32px">'
        f'<summary style="cursor:pointer;font-size:14px;color:var(--text-dim);'
        f'text-transform:uppercase;letter-spacing:0.08em;margin:28px 0 10px;font-weight:600">'
        f'▼ 過去の取引詳細（{n:,}件・新しい順）</summary>'
        '<table id="history-bets-table">'
        '<thead><tr>'
        '<th>日付</th><th>場</th><th>R</th><th>オッズ</th><th>EV</th>'
        '<th>ステーク</th><th>結果</th><th>PnL</th><th>累計PnL</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        '</table>'
        '</details>'
    )


def _render_signal_log_section(log_path: Path, current_ids: set[str],
                               schedule: dict[str, str]) -> str:
    """今日のシグナル履歴セクション。

    snapshot CSV を race_id ごとに集計し、初回出現〜最終出現、最高EV、最新EV、
    結果、現状（候補中/外れた）を表示する。
    """
    if not log_path.exists():
        return ""
    try:
        df = pd.read_csv(log_path)
    except Exception:
        return ""
    if df.empty or "race_id" not in df.columns:
        return ""

    # 集計: race_id ごとに first/last/max
    df["snapshot_at"] = pd.to_datetime(df["snapshot_at"], errors="coerce")
    if "ev" not in df.columns:
        return ""

    # 各 race_id の「最高EV を観測したスナップ」を抽出
    idx_max_ev = df.groupby("race_id")["ev"].idxmax()
    max_rows = df.loc[idx_max_ev].set_index("race_id")[["ev", "odds_win", "snapshot_at"]]
    max_rows = max_rows.rename(columns={
        "ev": "max_ev", "odds_win": "max_ev_odds", "snapshot_at": "max_ev_at"
    })

    # 各 race_id の「最後のスナップ」
    idx_last = df.groupby("race_id")["snapshot_at"].idxmax()
    last_rows = df.loc[idx_last].set_index("race_id")
    last_rows = last_rows.rename(columns={
        "ev": "latest_ev", "odds_win": "latest_odds",
        "stake": "latest_stake", "snapshot_at": "last_seen",
    })

    # 各 race_id の「最初のスナップ」
    first_at = df.groupby("race_id")["snapshot_at"].min().rename("first_seen")

    # n_iterations
    n_iter = df.groupby("race_id").size().rename("n_iterations")

    agg = pd.concat([max_rows, last_rows[["latest_ev", "latest_odds", "latest_stake",
                                          "race_finished", "hit", "pnl", "winner_lane",
                                          "last_seen"]],
                     first_at, n_iter], axis=1).reset_index()

    # 並び替え: 候補中→最高EV降順、候補外→最終出現の新しい順
    agg["currently_active"] = agg["race_id"].astype(str).isin(current_ids)
    agg = agg.sort_values(
        ["currently_active", "max_ev", "last_seen"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    if agg.empty:
        return ""

    # 結果補完: snapshot 当時に未確定だったレースを races_payouts.parquet で照合
    # （候補外になったあとでレースが終わった場合、snapshot には反映されない）
    from src.utils.config import RAW_DIR
    payouts_path = RAW_DIR / "races_payouts.parquet"
    if payouts_path.exists():
        try:
            payouts = pd.read_parquet(payouts_path,
                                      columns=["race_id", "bet_type", "combo", "payout_yen"])
            win_payouts = payouts[payouts["bet_type"] == "win"].copy()
            win_payouts["winner_lane"] = pd.to_numeric(win_payouts["combo"], errors="coerce")
            win_payouts["payout_yen"] = pd.to_numeric(win_payouts["payout_yen"], errors="coerce")
            win_payouts = win_payouts.dropna(subset=["winner_lane"])
            win_map = win_payouts.set_index("race_id")[["winner_lane", "payout_yen"]].to_dict("index")
        except Exception:
            win_map = {}
    else:
        win_map = {}

    for idx, r in agg.iterrows():
        finished_in_log = bool(r.get("race_finished", False)) if pd.notna(r.get("race_finished")) else False
        if finished_in_log:
            continue  # snapshot で既に確定済み
        rid = str(r["race_id"])
        if rid not in win_map:
            continue  # まだレース未終了 or 払戻データ無し
        info = win_map[rid]
        winner = int(info["winner_lane"])
        payout_per_100 = float(info["payout_yen"]) if pd.notna(info["payout_yen"]) else 0.0
        latest_stake = int(r["latest_stake"]) if pd.notna(r.get("latest_stake")) else 0
        hit = (winner == 1)
        if hit:
            return_yen = latest_stake * payout_per_100 / 100.0
            pnl = int(return_yen - latest_stake)
        else:
            pnl = -latest_stake
        agg.at[idx, "race_finished"] = True
        agg.at[idx, "hit"] = hit
        agg.at[idx, "winner_lane"] = winner
        agg.at[idx, "pnl"] = pnl

    rows_html: list[str] = []
    for _, r in agg.iterrows():
        rid = str(r["race_id"])
        date_str, jcd, rno = _parse_race_id(rid)
        venue = VENUE_CODES.get(jcd, "?")
        rno_int = int(rno) if rno.isdigit() else 0
        deadline = schedule.get(rid, "")
        active = bool(r["currently_active"])
        finished = bool(r.get("race_finished", False)) if pd.notna(r.get("race_finished")) else False

        # 状態セル
        if active and not finished:
            status = "<span class='badge badge-active'>🟢 候補中</span>"
        elif finished:
            hit = bool(r.get("hit", False))
            if hit:
                status = "<span class='badge badge-hit'>🟢 1着</span>"
            else:
                wl = r.get("winner_lane")
                wl_text = f"{int(wl)}号艇1着" if pd.notna(wl) else "外れ"
                status = f"<span class='badge badge-miss'>✕ {wl_text}</span>"
        else:
            status = "<span class='badge badge-dropped'>⚪ 候補外</span>"

        max_ev = float(r["max_ev"])
        latest_ev = float(r["latest_ev"])
        max_odds = float(r["max_ev_odds"]) if pd.notna(r.get("max_ev_odds")) else 0
        latest_odds = float(r["latest_odds"]) if pd.notna(r.get("latest_odds")) else 0
        max_at = pd.to_datetime(r["max_ev_at"]).strftime("%H:%M") if pd.notna(r["max_ev_at"]) else ""
        first_seen = pd.to_datetime(r["first_seen"]).strftime("%H:%M") if pd.notna(r["first_seen"]) else ""
        last_seen = pd.to_datetime(r["last_seen"]).strftime("%H:%M") if pd.notna(r["last_seen"]) else ""
        n_iter_v = int(r.get("n_iterations", 0))
        latest_stake = int(r["latest_stake"]) if pd.notna(r.get("latest_stake")) else 0

        # 結果セル: 確定済みなら PnL、未確定なら -
        if finished:
            pnl = int(r.get("pnl", 0) or 0)
            pnl_cls = "pos" if pnl > 0 else ("neg" if pnl < 0 else "")
            pnl_sign = "+" if pnl > 0 else ""
            pnl_cell = f"<span class='{pnl_cls}'>{pnl_sign}¥{pnl:,}</span>"
        else:
            pnl_cell = "-"

        deadline_disp = deadline if deadline else "-"

        rows_html.append(
            "<tr>"
            f"<td class='venue'>{venue}({jcd})</td>"
            f"<td class='race' data-sort='{rno_int}'>{rno_int}R</td>"
            f"<td>{deadline_disp}</td>"
            f"<td>{status}</td>"
            f"<td data-sort='{max_ev:.3f}'>"
            f"<span class='{_ev_class(max_ev)}'>{max_ev:.3f}</span>"
            f"<br><span style='color:var(--text-dim);font-size:11px'>"
            f"{max_at} (オッズ{max_odds:.2f})</span></td>"
            f"<td data-sort='{latest_ev:.3f}'>"
            f"<span class='{_ev_class(latest_ev)}'>{latest_ev:.3f}</span>"
            f"<br><span style='color:var(--text-dim);font-size:11px'>"
            f"オッズ{latest_odds:.2f}</span></td>"
            f"<td class='stake'>¥{latest_stake:,}</td>"
            f"<td>{first_seen}〜{last_seen}<br>"
            f"<span style='color:var(--text-dim);font-size:11px'>{n_iter_v}回観測</span></td>"
            f"<td data-sort='{int(r.get('pnl', 0) or 0)}'>{pnl_cell}</td>"
            "</tr>"
        )

    n_total = len(agg)
    n_active = int(agg["currently_active"].sum())
    finished_mask = agg["race_finished"].fillna(False).astype(bool)
    n_finished = int(finished_mask.sum())
    n_hit = int(finished_mask.sum() and agg.loc[finished_mask, "hit"].fillna(False).astype(bool).sum())
    n_pending = n_total - n_finished
    hit_rate_str = f"（的中{n_hit}/{n_finished}）" if n_finished else ""
    return (
        f'<details class="signal-log" open style="margin-bottom:24px">'
        f'<summary style="cursor:pointer;font-size:14px;color:var(--text-dim);'
        f'text-transform:uppercase;letter-spacing:0.08em;margin:28px 0 10px;font-weight:600">'
        f'▼ 本日のシグナル履歴（全{n_total}件 / 候補中{n_active}件 / 確定{n_finished}件{hit_rate_str} / 未確定{n_pending}件）</summary>'
        '<table id="signal-log-table">'
        '<thead><tr>'
        '<th>場</th><th>R</th><th>締切</th><th>状態</th>'
        '<th>最高 EV</th><th>最新 EV</th><th>推奨ステーク</th>'
        '<th>観測時刻</th><th>PnL</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        '</table>'
        '</details>'
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bets", required=True)
    p.add_argument("--out", default="data/processed/signals.html")
    p.add_argument("--ev-threshold", default="1.05",
                   help="表示用のEV閾値ラベル（描画のみ。実際の閾値はbacktestで決まる）")
    p.add_argument("--date-label", default=None,
                   help="ヘッダの日付ラベル。未指定なら CSV から自動抽出")
    p.add_argument("--schedule", default="data/raw/race_schedule.csv",
                   help="締切時刻 CSV のパス（あれば締切時刻列に表示）")
    p.add_argument("--history-summary", default=None,
                   help="過去バックテストの summary JSON。あればレポートに集計セクションを追加")
    p.add_argument("--history-equity", default=None,
                   help="過去バックテストの equity CSV（SVG折れ線を描画）")
    p.add_argument("--history-bets", default=None,
                   help="過去バックテストの bets CSV（個別取引テーブルを表示）")
    p.add_argument("--history-since", default="",
                   help="過去バックテストの開始日（表示用）")
    p.add_argument("--history-until", default="",
                   help="過去バックテストの終了日（表示用）")
    p.add_argument("--signal-log", default=None,
                   help="本日のシグナル履歴 snapshot CSV のパス。"
                        "watch_signal の各 iteration で追記されたものを集計")
    args = p.parse_args()

    bets = pd.read_csv(args.bets)
    if "race_date" in bets.columns and len(bets):
        bets["race_date"] = pd.to_datetime(bets["race_date"]).dt.strftime("%Y-%m-%d")
    schedule = _load_schedule(Path(args.schedule))

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
    if len(bets):
        bets = bets.sort_values(["ev" if "ev" in bets.columns else "stake"], ascending=False)

    history_section = ""
    if args.history_summary:
        # 実データ範囲を bets から取得（要求範囲とのギャップを HTML で可視化するため）
        actual_since, actual_until = "", ""
        if args.history_bets and Path(args.history_bets).exists():
            try:
                hb = pd.read_csv(args.history_bets)
            except Exception:
                hb = pd.DataFrame()
            if not hb.empty:
                if "race_finished" in hb.columns:
                    hb = hb[hb["race_finished"].fillna(False).astype(bool)]
                if "race_date" not in hb.columns or hb["race_date"].isna().all():
                    hb["race_date"] = hb["race_id"].astype(str).str[:8]
                d = pd.to_datetime(hb["race_date"], errors="coerce").dropna()
                if len(d):
                    actual_since = d.min().strftime("%Y-%m-%d")
                    actual_until = d.max().strftime("%Y-%m-%d")

        history_section = _render_history_section(
            Path(args.history_summary),
            Path(args.history_equity) if args.history_equity else Path("/dev/null"),
            args.history_since,
            args.history_until,
            actual_since=actual_since,
            actual_until=actual_until,
        )
        if args.history_bets:
            history_section += _render_history_bets_section(Path(args.history_bets))

    summ = _summary(bets)
    signal_log_section = ""
    if args.signal_log:
        current_ids = set(bets["race_id"].astype(str)) if len(bets) else set()
        signal_log_section = _render_signal_log_section(
            Path(args.signal_log), current_ids, schedule,
        )

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
        signal_log_section=signal_log_section,
        history_section=history_section,
        rows="".join(_row(r, schedule) for _, r in bets.iterrows()),
        totals_row=_totals_row(bets),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"saved: {out.resolve()}")
    print(f"open in browser: file://{out.resolve()}")


if __name__ == "__main__":
    main()
