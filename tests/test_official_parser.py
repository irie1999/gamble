"""公式テキスト形式パーサのテスト（ネットワーク不要）。

実データの体裁:
  - 場ヘッダ: "ボートレース<場名>" （場名に全角空白を含む）
  - レース見出し: 全角 "　１Ｒ ..."（NFKC後 "1R"）
  - 艇行: 名前・年齢・支部・体重・級別が密着、体重は整数kg
"""
from __future__ import annotations

import textwrap

from src.scraper.official_parser import parse_banzuke, parse_results


# 実データに即したサンプル（"芦　屋" のように全角空白入り、全角数字、密着フォーマット）
SAMPLE_BANZUKE = textwrap.dedent("""\
    ボートレース芦　屋   　８月１５日  九州スポーツ杯争奪オ  第　６日

                                ＊＊＊　番組表　＊＊＊

              九州スポーツ杯争奪オール九州選抜戦

       第　６日          ２０２４年　８月１５日                  ボートレース芦　屋

                   −内容については主催者発行のものと照合して下さい−


    　１Ｒ  サンライズＶ          Ｈ１８００ｍ  電話投票締切予定０８：４７
    -------------------------------------------------------------------------------
    艇 選手 選手  年 支 体級    全国      当地     モーター   ボート   今節成績  早
    番 登番  名   齢 部 重別 勝率  2率  勝率  2率  NO  2率  NO  2率  １２３４５６見
    -------------------------------------------------------------------------------
    1 5018竹下大樹25福岡54A2 6.32 45.97 4.76 23.53 40 30.12 12 36.49 433 5 333    9
    2 4495森晋太郎37福岡54B1 4.89 30.93 4.86 30.14 42 25.00 68 36.78 5 465 454    6
    3 5073上原健次29福岡55B1 4.25 21.43 4.52 19.51 27 30.67 41 29.73 5 655 355    7
    4 5105富田恕生25福岡52B1 5.26 36.28 4.90 30.30 35 41.98 28 37.08 363 461 63   9
    5 5263花田凱成25福岡52B1 2.56 11.11 1.46  1.85 52 27.42 39 40.00 3 522 634    5
    6 5351齊藤　廉20福岡53B2 1.67  2.33 0.00  0.00 10 32.10 56 34.57 6 63S 666    6

    　２Ｒ  サンライズＷ          Ｈ１８００ｍ  電話投票締切予定０９：１３
    -------------------------------------------------------------------------------
    艇 選手 選手  年 支 体級    全国      当地     モーター   ボート   今節成績  早
    番 登番  名   齢 部 重別 勝率  2率  勝率  2率  NO  2率  NO  2率  １２３４５６見
    -------------------------------------------------------------------------------
    1 4136江夏　満44福岡52A1 6.82 57.38 7.14 55.24 24 31.71 32 31.71 23263 4323  11
    2 5271杉山太陽23福岡51B1 2.60 10.34 1.34  0.00 53 14.04 50 27.27 534 666 5    6
    3 5102原田雄次26福岡52B1 4.69 26.53 4.16 18.58 38 16.13 23 28.95 433 414 2    7
    4 4551三川昂暁35福岡53A2 6.16 40.00 5.92 42.16 37 31.17 21 27.59 124 364 1    9
    5 3752宇土泰就53福岡55B1 4.23 22.00 4.63 21.43 50 47.62 24 23.44 316 455 64
    6 3953志道吉和47福岡52B1 3.87 22.58 4.18 23.60 17 35.71 16 40.24 5 454 521    8
    """)


SAMPLE_RESULT = textwrap.dedent("""\
    ボートレース芦　屋   ２０２４年　８月１５日

       １Ｒ
    01 4 5105富田恕生
    02 2 4495森晋太郎
    03 3 5073上原健次
    転 1 5018竹下大樹
    エ 6 5351齊藤　廉
    妨 5 5263花田凱成

    単勝 4 530
    複勝 4 320
    複勝 2 760
    2連単 4-2 3,950
    2連複 2=4 3,190
    3連単 4-2-3 26,420
    3連複 2=3=4 4,480
    """)


def test_parse_banzuke_real_format():
    cards = parse_banzuke(SAMPLE_BANZUKE)
    assert len(cards) == 2, f"expected 2 races, got {len(cards)}"
    c = cards[0]
    assert c.race_date == "20240815"
    assert c.venue_code == "21"  # 芦屋
    assert c.race_no == 1
    assert len(c.entries) == 6, f"expected 6 entries, got {len(c.entries)}: {c.entries}"

    e1 = c.entries[0]
    assert e1.lane == 1
    assert e1.racer_id == "5018"
    assert e1.name == "竹下大樹"
    assert e1.age == 25
    assert e1.branch == "福岡"
    assert e1.weight == 54.0
    assert e1.grade == "A2"
    assert e1.win_rate_national == 6.32
    assert e1.place_rate_national == 45.97
    assert e1.win_rate_local == 4.76
    assert e1.place_rate_local == 23.53
    assert e1.motor_no == 40
    assert e1.motor_2rate == 30.12
    assert e1.boat_no == 12
    assert e1.boat_2rate == 36.49

    # 6艇目（名前に全角空白入り → "齊藤廉" に正規化される）
    e6 = c.entries[5]
    assert e6.lane == 6
    assert e6.racer_id == "5351"
    assert e6.name == "齊藤廉"
    assert e6.grade == "B2"


def test_parse_results_with_disqualification():
    results = parse_results(SAMPLE_RESULT)
    assert len(results) == 1
    r = results[0]
    assert r.venue_code == "21"
    assert r.race_no == 1
    # 6艇分（うち3艇は失格・妨害・転覆で rank=None）
    assert len(r.rows) == 6
    winner = next(row for row in r.rows if row.rank == 1)
    assert winner.lane == 4
    assert winner.racer_id == "5105"
    # 失格系
    dnf_lanes = [row.lane for row in r.rows if row.rank is None]
    assert set(dnf_lanes) == {1, 5, 6}

    # 払戻
    assert ("4", 530) in r.payouts.get("win", [])
    assert ("4-2", 3950) in r.payouts.get("exacta", [])
    assert ("2=4", 3190) in r.payouts.get("quinella", [])
    assert ("4-2-3", 26420) in r.payouts.get("trifecta", [])
    assert ("2=3=4", 4480) in r.payouts.get("trio", [])
