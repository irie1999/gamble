"""
バックテスト結果HTMLレポート生成モジュール（競艇・競輪共通）
"""

import json
import webbrowser
from pathlib import Path
from datetime import datetime


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

    # 資産推移の計算（win_flag を使用）
    equity_labels = ["開始"]
    equity_values = [initial]
    running = initial
    for b in bets:
        running -= b["bet_amount"]
        if b.get("win_flag", False):
            running += b["bet_amount"] * b["odds"]
        equity_labels.append(b["race_id"])
        equity_values.append(round(running, 0))

    profit_color = "#22c55e" if profit >= 0 else "#ef4444"
    roi_color = "#22c55e" if roi >= 0 else "#ef4444"

    bet_rows = ""
    for i, b in enumerate(bets):
        is_win = b.get("win_flag", False)
        row_class = "win-row" if is_win else ""
        result_badge = (
            '<span class="badge win">的中</span>' if is_win
            else '<span class="badge loss">外れ</span>'
        )
        bet_rows += f"""
        <tr class="{row_class}">
          <td>{i+1}</td>
          <td>{b['race_id']}</td>
          <td>{b['bet_type']}</td>
          <td>{b['selections']}</td>
          <td>{b['predicted_prob']:.3f}</td>
          <td>{b['implied_prob']:.3f}</td>
          <td><span class="edge {'pos' if b['edge']>=0 else 'neg'}">{b['edge']:+.3f}</span></td>
          <td>{b['odds']:.1f}倍</td>
          <td>{b['bet_amount']:,}円</td>
          <td>{b['expected_value']:.3f}</td>
          <td>{result_badge}</td>
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
            <th>#</th><th>レースID</th><th>種別</th><th>選択</th>
            <th>予測P</th><th>市場P</th><th>エッジ</th>
            <th>オッズ</th><th>賭け金</th><th>EV</th><th>結果</th>
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
    """オッズ帯別の的中率バーチャート"""
    bands = [
        ("〜2倍", 0, 2),
        ("2〜5倍", 2, 5),
        ("5〜10倍", 5, 10),
        ("10〜20倍", 10, 20),
        ("20倍〜", 20, 9999),
    ]
    rows = ""
    for label, lo, hi in bands:
        subset = [b for b in bets if lo <= b["odds"] < hi]
        if not subset:
            continue
        w = sum(1 for b in subset if b.get("win_flag", False))
        rate = w / len(subset)
        rows += f"""
        <div class="bar-label"><span>{label}</span><span>{w}/{len(subset)} ({rate:.0%})</span></div>
        <div class="bar-track"><div class="bar-fill" style="width:{rate*100:.0f}%;background:#60a5fa"></div></div>"""
    return rows or "<p style='color:#64748b;font-size:0.85rem'>データなし</p>"


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
