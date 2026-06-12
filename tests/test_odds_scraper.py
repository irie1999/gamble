"""オッズパーサのテスト（ネットワーク不要、最低限のセレクタ動作確認）。"""
from __future__ import annotations

import textwrap

from src.scraper.odds_scraper import OddsScraper


# 単勝オッズページの最小サンプル（戦略2: 単勝見出し → 直近tableをマッチ）
SAMPLE_ODDS_HTML = textwrap.dedent("""\
    <html><body>
      <h3>単勝オッズ</h3>
      <table>
        <tbody>
          <tr><td>1</td><td>佐々木</td><td>2.5</td></tr>
          <tr><td>2</td><td>石川</td><td>4.8</td></tr>
          <tr><td>3</td><td>松本</td><td>6.1</td></tr>
          <tr><td>4</td><td>原田</td><td>3.2</td></tr>
          <tr><td>5</td><td>坪内</td><td>15.0</td></tr>
          <tr><td>6</td><td>澁川</td><td>40.0</td></tr>
        </tbody>
      </table>
    </body></html>
    """)


def test_parse_win_odds_minimal():
    odds = OddsScraper._parse_win_odds(SAMPLE_ODDS_HTML)
    assert len(odds) == 6, f"expected 6 lanes, got {odds}"
    assert odds[1] == 2.5
    assert odds[2] == 4.8
    assert odds[3] == 6.1
    assert odds[4] == 3.2
    assert odds[5] == 15.0
    assert odds[6] == 40.0
