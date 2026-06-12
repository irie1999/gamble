"""preprocessor の最小動作テスト（ネットワーク不要）"""
from __future__ import annotations

from src.data.preprocessor import build_dataset, add_targets, make_race_id
from src.scraper.boatrace_scraper import RaceCard, RacerEntry, RaceResult, RaceResultRow


def _sample_card() -> RaceCard:
    entries = [
        RacerEntry(
            lane=i, racer_id=f"400{i}", name=f"選手{i}", grade="A1",
            branch="東京", age=30, weight=52.0,
            win_rate_national=6.5, win_rate_local=6.0,
            place_rate_national=45.0, place_rate_local=42.0,
            motor_no=10 + i, motor_2rate=35.0,
            boat_no=20 + i, boat_2rate=30.0,
        )
        for i in range(1, 7)
    ]
    return RaceCard(
        race_date="20240815", venue_code="12", race_no=11,
        title="TEST", distance_m=1800, entries=entries,
    )


def _sample_result() -> RaceResult:
    rows = [
        RaceResultRow(lane=i, rank=i, racer_id=f"400{i}", name=f"選手{i}",
                      race_time_sec=110.0 + i, start_timing=0.15)
        for i in range(1, 7)
    ]
    return RaceResult(
        race_date="20240815", venue_code="12", race_no=11, rows=rows,
    )


def test_make_race_id():
    assert make_race_id("20240815", "12", 11) == "20240815-12-11"


def test_build_dataset_and_targets():
    df = build_dataset([(_sample_card(), _sample_result())])
    assert len(df) == 6
    assert set(df["lane"]) == {1, 2, 3, 4, 5, 6}
    assert df["race_id"].nunique() == 1

    df2 = add_targets(df)
    assert df2.loc[df2["lane"] == 1, "is_win"].iloc[0] == 1
    assert df2.loc[df2["lane"] == 1, "is_top3"].iloc[0] == 1
    assert df2.loc[df2["lane"] == 6, "is_win"].iloc[0] == 0
    assert df2.loc[df2["lane"] == 6, "is_top3"].iloc[0] == 0
