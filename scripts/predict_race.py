"""指定の場×R×日付について、リアルタイムに出走表を取得して1着確率を予測する。

使い方:
    python -m scripts.predict_race --venue 12 --race 11 --date 2024-08-15
"""
from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from src.data.preprocessor import build_dataset
from src.features.feature_engineering import build_features
from src.models.predict import predict_win_probability
from src.scraper.boatrace_scraper import BoatraceScraper, RaceResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue", required=True, help="場コード 例: 12")
    parser.add_argument("--race", type=int, required=True, help="レース番号 1〜12")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    d = date.fromisoformat(args.date)
    scraper = BoatraceScraper()
    card = scraper.fetch_race_card(args.venue, args.race, d)
    # 結果は当日未確定のため空で渡す
    empty = RaceResult(race_date=card.race_date, venue_code=card.venue_code, race_no=card.race_no)
    df = build_dataset([(card, empty)])
    feats = build_features(df)
    pred = predict_win_probability(feats)

    out = pred[["race_id", "lane", "name", "grade", "pred_win_prob"]].sort_values(
        ["race_id", "pred_win_prob"], ascending=[True, False]
    )
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
