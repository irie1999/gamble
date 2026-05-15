"""
バックテスト結果HTMLレポート生成モジュール（競艇・競輪共通）
"""

import json
import webbrowser
from pathlib import Path
from datetime import datetime


VENUE_CODES = {
    "11": "函館",   "12": "青森",   "13": "いわき平", "21": "弥彦",
    "22": "前橋",   "23": "取手",   "24": "宇都宮",   "25": "大宮",
    "26": "西武園", "27": "京王閣", "28": "立川",     "31": "松戸",
    "32": "千葉",   "34": "川崎",   "35": "平塚",     "36": "小田原",
    "37": "伊東",   "38": "静岡",   "42": "名古屋",   "43": "岐阜",
    "44": "大垣",   "45": "豊橋",   "46": "富山",     "47": "松阪",
    "48": "四日市", "51": "福井",   "53": "奈良",     "54": "向日町",
    "55": "和歌山", "56": "岸和田", "61": "玉野",     "62": "広島",
    "63": "防府",   "71": "高松",   "73": "小松島",   "74": "高知",
    "75": "松山",   "81": "小倉",   "83": "久留米",   "84": "武雄",
    "85": "佐世保", "86": "別府",   "87": "熊本",
}


def _parse_race_id(race_id: str) -> tuple[str, str, str]:
    """race_id から (venue_name, date_str, race_no) を抽出"""
    rid = str(race_id).replace("_", "")
    if len(rid) >= 16:
        venue_code = rid[:2]
        date_str = rid[2:10]   # YYYYMMDD
        race_no  = str(int(rid[12:]))  # 末尾4桁
        venue = VENUE_CODES.get(venue_code, venue_code)
        date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        return venue, date_fmt, race_no
    # 旧形式 YYYYMMDD_VV_R
    parts = rid.split("_") if "_" in race_id else []
    if len(parts) == 3:
        date_str, venue_code, race_no = parts
        venue = VENUE_CODES.get(venue_code, venue_code)
        date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}" if len(date_str)==8 else date_str
        return venue, date_fmt, race_no
    return "", race_id, ""


def _format_race_id(race_id: str) -> str:
    venue, date_fmt, race_no = _parse_race_id(race_id)
    if venue and race_no:
        return f"{venue} R{race_no}"
    return race_id


def generate_html(session_data: dict, title: str = "バックテスト結果") -> str:
    bets = session_data.get("bets", [])
    initial = session_data["initial_bankroll"]
    final = session_data["final_bankroll"]
    profit = session_data["profit"]
    roi = session_data["roi"]
    wins = session_data["wins"]
    losses = session_data["losses"]
    total = wins + losses
    win_rate = wins / total if total > 0 else 0
    total_bet = session_data["total_bet"]

    # 資産推移の計算（actual_payout で正確に計算）
    equity_labels = ["開始"]
    equity_values = [initial]
    running = initial
    for b in bets:
        actual = b.get("actual_payout", None)
        if actual is None:
            # 旧形式フォールバック: win_flag + return_odds
            payout = b["bet_amount"] * b.get("return_odds", b["odds"]) if b.get("win_flag") else 0.0
        else:
            payout = actual
        running += payout - b["bet_amount"]
        equity_labels.append(b["race_id"])
        equity_values.append(round(running, 0))

    profit_color = "#22c55e" if profit >= 0 else "#ef4444"
    roi_color = "#22c55e" if roi >= 0 else "#ef4444"

    bet_rows = ""
    cumulative = 0.0
    for i, b in enumerate(bets):
        is_win = b.get("win_flag", False)
        actual_payout = b.get("actual_payout", None)
        return_odds = b.get("return_odds", 0.0)
        bet_amount = b["bet_amount"]

        # 払戻・損益の計算
        if actual_payout is not None:
            is_refund = (not is_win) and actual_payout == bet_amount
            pnl = actual_payout - bet_amount
        else:
            is_refund = False
            pnl = bet_amount * (b["odds"] - 1) if is_win else -bet_amount

        cumulative += pnl

        row_class = "win-row" if is_win else ""
        if is_win:
            result_badge = f'<span class="badge win">的中 {return_odds:.1f}倍</span>'
        elif is_refund:
            result_badge = '<span class="badge" style="background:rgba(250,204,21,0.15);color:#fbbf24">返金</span>'
        else:
            result_badge = '<span class="badge loss">外れ</span>'

        pnl_color = "#4ade80" if pnl > 0 else ("#94a3b8" if pnl == 0 else "#f87171")
        cum_color = "#4ade80" if cumulative > 0 else ("#94a3b8" if cumulative == 0 else "#f87171")
        venue, date_fmt, race_no = _parse_race_id(b['race_id'])
        race_label = f"{venue} R{race_no}" if venue and race_no else b['race_id']
        bet_rows += f"""
        <tr class="{row_class}">
          <td>{i+1}</td>
          <td style="color:#94a3b8;font-size:0.8rem">{date_fmt}</td>
          <td>{race_label}</td>
          <td>{b['bet_type']}</td>
          <td>{b['selections']}</td>
          <td>{b['predicted_prob']:.3f}</td>
          <td>{b['bet_amount']:,}円</td>
          <td>{result_badge}</td>
          <td style="color:{pnl_color};font-weight:600">{pnl:+,.0f}円</td>
          <td style="color:{cum_color};font-weight:600">{cumulative:+,.0f}円</td>
        </tr>"""

    equity_labels_js = json.dumps(equity_labels[:500])
    equity_values_js = json.dumps(equity_values[:500])

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', 'Helvetica Neue', sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }}
  .header {{ background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 2rem; border-bottom: 1px solid #334155; }}
  .header h1 {{ font-size: 1.8rem; font-weight: 700; color: #f8fafc; }}
  .header .subtitle {{ color: #94a3b8; margin-top: 0.25rem; font-size: 0.9rem; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 2rem; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
  .kpi {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1.25rem; }}
  .kpi .label {{ font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem; }}
  .kpi .value {{ font-size: 1.6rem; font-weight: 700; }}
  .kpi .value.profit {{ color: {profit_color}; }}
  .kpi .value.roi {{ color: {roi_color}; }}
  .kpi .value.neutral {{ color: #f1f5f9; }}
  .kpi .value.blue {{ color: #60a5fa; }}
  .kpi .value.green {{ color: #4ade80; }}
  .section {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }}
  .section h2 {{ font-size: 1rem; font-weight: 600; color: #94a3b8; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  .chart-wrap {{ position: relative; height: 260px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ background: #0f172a; color: #64748b; padding: 0.6rem 0.8rem; text-align: left; font-weight: 500; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; position: sticky; top: 0; }}
  td {{ padding: 0.55rem 0.8rem; border-bottom: 1px solid #1e293b; color: #cbd5e1; }}
  tr:hover td {{ background: #263347; }}
  .win-row td {{ background: rgba(34,197,94,0.05); }}
  .badge {{ display: inline-block; padding: 0.15rem 0.55rem; border-radius: 999px; font-size: 0.72rem; font-weight: 600; }}
  .badge.win {{ background: rgba(34,197,94,0.15); color: #4ade80; }}
  .badge.loss {{ background: rgba(239,68,68,0.12); color: #f87171; }}
  .edge.pos {{ color: #4ade80; }}
  .edge.neg {{ color: #f87171; }}
  .table-wrap {{ overflow-x: auto; max-height: 480px; overflow-y: auto; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
  @media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  .bar-label {{ display: flex; justify-content: space-between; font-size: 0.8rem; color: #64748b; margin-bottom: 0.3rem; }}
  .bar-track {{ background: #0f172a; border-radius: 4px; height: 8px; overflow: hidden; margin-bottom: 0.8rem; }}
  .bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.4s; }}
</style>
</head>
<body>
<div class="header">
  <h1>📊 {title}</h1>
  <div class="subtitle">生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
</div>
<div class="container">

  <!-- KPI -->
  <div class="kpi-grid">
    <div class="kpi">
      <div class="label">初期資金</div>
      <div class="value neutral">¥{initial:,.0f}</div>
    </div>
    <div class="kpi">
      <div class="label">最終資金</div>
      <div class="value neutral">¥{final:,.0f}</div>
    </div>
    <div class="kpi">
      <div class="label">損益</div>
      <div class="value profit">¥{profit:+,.0f}</div>
    </div>
    <div class="kpi">
      <div class="label">ROI</div>
      <div class="value roi">{roi:+.1%}</div>
    </div>
    <div class="kpi">
      <div class="label">賭け回数</div>
      <div class="value blue">{total}回</div>
    </div>
    <div class="kpi">
      <div class="label">的中率</div>
      <div class="value green">{win_rate:.1%}</div>
    </div>
    <div class="kpi">
      <div class="label">総賭け金</div>
      <div class="value neutral">¥{total_bet:,.0f}</div>
    </div>
    <div class="kpi">
      <div class="label">平均EV</div>
      <div class="value blue">{sum(b['expected_value'] for b in bets)/len(bets):.3f}倍</div>
    </div>
  </div>

  <!-- 資産推移チャート -->
  <div class="section">
    <h2>資産推移</h2>
    <div class="chart-wrap">
      <canvas id="equityChart"></canvas>
    </div>
  </div>

  <!-- 統計 + エッジ分布 -->
  <div class="two-col">
    <div class="section">
      <h2>的中 / 外れ内訳</h2>
      <div class="chart-wrap" style="height:200px">
        <canvas id="winLossChart"></canvas>
      </div>
    </div>
    <div class="section">
      <h2>オッズ帯別 的中率</h2>
      {_odds_breakdown_html(bets)}
    </div>
  </div>

  <!-- ベット一覧 -->
  <div class="section">
    <h2>ベット一覧（全{total}件）</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th><th>日付</th><th>レース</th><th>種別</th><th>選択</th>
            <th>予測P</th><th>賭け金</th><th>結果</th><th>損益</th><th>累積損益</th>
          </tr>
        </thead>
        <tbody>{bet_rows}</tbody>
      </table>
    </div>
  </div>

</div>

<script>
const equityLabels = {equity_labels_js};
const equityData = {equity_values_js};
const eq = document.getElementById('equityChart').getContext('2d');
new Chart(eq, {{
  type: 'line',
  data: {{
    labels: equityLabels,
    datasets: [{{
      label: '資産推移',
      data: equityData,
      borderColor: '{profit_color}',
      backgroundColor: '{profit_color}22',
      fill: true,
      tension: 0.3,
      pointRadius: equityData.length > 80 ? 0 : 3,
      borderWidth: 2,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ display: false }},
      y: {{
        ticks: {{ color: '#64748b', callback: v => '¥' + v.toLocaleString() }},
        grid: {{ color: '#1e293b' }}
      }}
    }}
  }}
}});

const wl = document.getElementById('winLossChart').getContext('2d');
new Chart(wl, {{
  type: 'doughnut',
  data: {{
    labels: ['的中 {wins}回', '外れ {losses}回'],
    datasets: [{{
      data: [{wins}, {losses}],
      backgroundColor: ['#4ade80', '#f87171'],
      borderWidth: 0,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ color: '#94a3b8', padding: 16 }} }}
    }}
  }}
}});
</script>
</body>
</html>"""
    return html


def _odds_breakdown_html(bets: list[dict]) -> str:
    """払戻倍率帯別の的中数バーチャート（実際の払戻倍率を使用）"""
    bands = [
        ("〜5倍", 0, 5),
        ("5〜15倍", 5, 15),
        ("15〜30倍", 15, 30),
        ("30〜100倍", 30, 100),
        ("100倍〜", 100, 9999),
    ]
    wins = [b for b in bets if b.get("win_flag", False)]
    if not wins:
        return "<p style='color:#64748b;font-size:0.85rem'>的中なし</p>"

    total_wins = len(wins)
    rows = f"<p style='color:#94a3b8;font-size:0.8rem;margin-bottom:0.8rem'>的中{total_wins}件の払戻倍率分布</p>"
    for label, lo, hi in bands:
        subset = [b for b in wins if lo <= b.get("return_odds", 0) < hi]
        if not subset:
            continue
        avg_odds = sum(b.get("return_odds", 0) for b in subset) / len(subset)
        rate = len(subset) / total_wins
        rows += f"""
        <div class="bar-label"><span>{label} (平均{avg_odds:.1f}倍)</span><span>{len(subset)}件 ({rate:.0%})</span></div>
        <div class="bar-track"><div class="bar-fill" style="width:{rate*100:.0f}%;background:#60a5fa"></div></div>"""
    return rows


def save_and_open(
    session_data: dict,
    output_path: Path,
    title: str = "バックテスト結果",
    auto_open: bool = True,
) -> Path:
    """HTMLを保存してブラウザで開く"""
    html = generate_html(session_data, title=title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"HTMLレポート保存: {output_path}")

    if auto_open:
        webbrowser.open(output_path.as_uri())
        print("ブラウザで開きました")

    return output_path


def from_session(session, output_path: Path, title: str, auto_open: bool = True) -> Path:
    """SessionResultオブジェクトから直接HTMLを生成"""
    bets_data = []
    for b in session.bets:
        bets_data.append({
            "race_id": b.race_id,
            "bet_type": b.bet_type,
            "selections": list(b.selections),
            "predicted_prob": b.predicted_prob,
            "implied_prob": b.implied_prob,
            "edge": b.edge,
            "odds": b.odds,
            "bet_amount": b.bet_amount,
            "expected_value": b.expected_value,
            "win_flag": False,
        })

    # 的中フラグを再計算（SessionResultは wins/losses カウントのみ保持）
    # bets の順序通りに的中フラグを振り直す
    wins_remaining = session.wins
    for b_data, b_obj in zip(bets_data, session.bets):
        # BettingResult に win_flag がなければ推測不可なのでスキップ
        pass

    data = {
        "initial_bankroll": session.initial_bankroll,
        "final_bankroll": session.final_bankroll,
        "profit": session.profit,
        "roi": session.roi,
        "wins": session.wins,
        "losses": session.losses,
        "total_bet": session.total_bet,
        "bets": bets_data,
    }
    return save_and_open(data, output_path, title=title, auto_open=auto_open)


if __name__ == "__main__":
    # サンプルJSON からHTML生成テスト
    import sys
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if path and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        out = path.with_suffix(".html")
        save_and_open(data, out, title=path.stem)
    else:
        print("使い方: python report.py <result.json>")
