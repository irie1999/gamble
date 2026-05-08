"""スクレイプ結果（RaceCard / RaceResult）を pandas DataFrame に変換する。

出力スキーマ（race_id 単位 × lane 単位の long-format）:

  race_id, race_date, venue_code, race_no, lane,
  racer_id, name, grade, age, weight,
  win_rate_national, win_rate_local,
  place_rate_national, place_rate_local,
  motor_no, motor_2rate, boat_no, boat_2rate,
  rank, race_time_sec, start_timing
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import pandas as pd

from src.scraper.boatrace_scraper import RaceCard, RaceResult


def make_race_id(race_date: str, venue_code: str, race_no: int) -> str:
    return f"{race_date}-{venue_code}-{int(race_no):02d}"


def card_to_rows(card: RaceCard) -> list[dict]:
    rows = []
    for e in card.entries:
        d = asdict(e)
        d.update({
            "race_id": make_race_id(card.race_date, card.venue_code, card.race_no),
            "race_date": card.race_date,
            "venue_code": card.venue_code,
            "race_no": card.race_no,
            "race_title": card.title,
            "distance_m": card.distance_m,
        })
        rows.append(d)
    return rows


def result_to_rows(result: RaceResult) -> list[dict]:
    rows = []
    rid = make_race_id(result.race_date, result.venue_code, result.race_no)
    for r in result.rows:
        rows.append({
            "race_id": rid,
            "lane": r.lane,
            "rank": r.rank,
            "race_time_sec": r.race_time_sec,
            "start_timing": r.start_timing,
        })
    return rows


def build_dataset(
    pairs: Iterable[tuple[RaceCard, RaceResult]],
) -> pd.DataFrame:
    """(card, result) のイテラブルから 1艇1行 の DataFrame を生成。"""
    card_rows: list[dict] = []
    result_rows: list[dict] = []
    for card, result in pairs:
        card_rows.extend(card_to_rows(card))
        result_rows.extend(result_to_rows(result))

    if not card_rows:
        return pd.DataFrame()

    cards_df = pd.DataFrame(card_rows)
    if result_rows:
        results_df = pd.DataFrame(result_rows)
        df = cards_df.merge(results_df, on=["race_id", "lane"], how="left")
    else:
        df = cards_df
        df["rank"] = pd.NA
        df["race_time_sec"] = pd.NA
        df["start_timing"] = pd.NA

    # 型の整え
    df["race_date"] = pd.to_datetime(df["race_date"], format="%Y%m%d", errors="coerce")
    for col in ["lane", "race_no", "age", "motor_no", "boat_no"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in [
        "weight", "win_rate_national", "win_rate_local",
        "place_rate_national", "place_rate_local",
        "motor_2rate", "boat_2rate",
        "race_time_sec", "start_timing",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["race_date", "venue_code", "race_no", "lane"]).reset_index(drop=True)
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """着順から学習ターゲットを生成。

    - is_win:   1着になったか (binary)
    - is_top2:  2着以内に入ったか
    - is_top3:  3着以内に入ったか
    """
    out = df.copy()
    rank = pd.to_numeric(out["rank"], errors="coerce")
    out["is_win"] = (rank == 1).astype("Int64")
    out["is_top2"] = (rank <= 2).astype("Int64")
    out["is_top3"] = (rank <= 3).astype("Int64")
    # 失格などで rank が NaN の行はターゲットも NaN に戻す
    mask_na = rank.isna()
    out.loc[mask_na, ["is_win", "is_top2", "is_top3"]] = pd.NA
    return out
