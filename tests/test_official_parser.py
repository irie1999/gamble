"""公式テキスト形式パーサのテスト（ネットワーク不要）。

固定幅ファイルのサンプルを最小構成で組み立て、艇行・払戻が抽出できることを確認する。
"""
from __future__ import annotations

import textwrap

from src.scraper.official_parser import parse_banzuke, parse_results


SAMPLE_BANZUKE = textwrap.dedent("""\
    24/08/15
    BBGN
    ボートレース住之江
       1R  一般  1800m
        1 4123 山田太郎    35 東京 51.5 A1 6.50 45.50 6.20 43.10 12 35.50 56 30.20
        2 4456 鈴木次郎    28 大阪 52.0 A2 5.50 38.00 5.30 36.00 22 40.00 47 28.10
        3 4789 佐藤三郎    40 福岡 53.0 B1 4.50 30.00 4.40 29.00 33 32.00 12 25.00
        4 4001 田中四郎    32 愛知 51.8 B1 4.20 28.00 4.10 27.50 44 30.00 23 22.10
        5 4234 高橋五郎    25 東京 50.5 A2 5.80 40.00 5.70 39.00 55 36.00 34 27.30
        6 4567 渡辺六郎    45 大阪 52.5 B2 3.80 22.00 3.70 21.00 11 25.00 45 20.00
       2R  一般  1800m
        1 4123 山田太郎    35 東京 51.5 A1 6.50 45.50 6.20 43.10 12 35.50 56 30.20
        2 4456 鈴木次郎    28 大阪 52.0 A2 5.50 38.00 5.30 36.00 22 40.00 47 28.10
        3 4789 佐藤三郎    40 福岡 53.0 B1 4.50 30.00 4.40 29.00 33 32.00 12 25.00
        4 4001 田中四郎    32 愛知 51.8 B1 4.20 28.00 4.10 27.50 44 30.00 23 22.10
        5 4234 高橋五郎    25 東京 50.5 A2 5.80 40.00 5.70 39.00 55 36.00 34 27.30
        6 4567 渡辺六郎    45 大阪 52.5 B2 3.80 22.00 3.70 21.00 11 25.00 45 20.00
    BEND
    """)


SAMPLE_RESULT = textwrap.dedent("""\
    24/08/15
    KBGN
    ボートレース住之江
       1R
        01 1 4123 山田太郎
        02 3 4789 佐藤三郎
        03 2 4456 鈴木次郎
        04 4 4001 田中四郎
        05 5 4234 高橋五郎
        06 6 4567 渡辺六郎
        単勝 1 230
        複勝 1 110
        2連単 1-3 1,230
        2連複 1=3 580
        3連単 1-3-2 4,560
        3連複 1=2=3 1,890
    KEND
    """)


def test_parse_banzuke_basic():
    cards = parse_banzuke(SAMPLE_BANZUKE)
    assert len(cards) == 2
    c = cards[0]
    assert c.race_date == "20240815"
    assert c.venue_code == "12"  # 住之江
    assert c.race_no == 1
    assert len(c.entries) == 6
    e1 = c.entries[0]
    assert e1.lane == 1
    assert e1.racer_id == "4123"
    assert e1.name == "山田太郎"
    assert e1.grade == "A1"
    assert e1.win_rate_national == 6.50
    assert e1.motor_2rate == 35.50


def test_parse_results_basic():
    results = parse_results(SAMPLE_RESULT)
    assert len(results) == 1
    r = results[0]
    assert r.race_date == "20240815"
    assert r.venue_code == "12"
    assert r.race_no == 1
    assert len(r.rows) == 6
    # 1着 lane=1
    winner = next(row for row in r.rows if row.rank == 1)
    assert winner.lane == 1
    assert winner.racer_id == "4123"

    # 払戻
    assert ("1", 230) in r.payouts.get("win", [])
    assert ("1-3", 1230) in r.payouts.get("exacta", [])
    assert ("1-3-2", 4560) in r.payouts.get("trifecta", [])
    assert ("1=2=3", 1890) in r.payouts.get("trio", [])
