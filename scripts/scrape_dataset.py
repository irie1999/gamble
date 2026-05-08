"""指定期間の出走表＋結果をスクレイプして parquet/csv で保存。

使い方:
    python -m scripts.scrape_dataset --from 2024-01-01 --to 2024-01-07
    python -m scripts.scrape_dataset --from 2024-01-01 --to 2024-01-07 --venues 12 22
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.data.preprocessor import build_dataset, add_targets
from src.scraper.boatrace_scraper import BoatraceScraper
from src.utils.config import RAW_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="date_from", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", required=True, help="YYYY-MM-DD")
    parser.add_argument("--venues", nargs="*", default=None, help="場コード (例: 01 12 22)")
    parser.add_argument("--out", default=str(RAW_DIR / "races.parquet"))
    args = parser.parse_args()

    start = date.fromisoformat(args.date_from)
    end = date.fromisoformat(args.date_to)
    venues = args.venues

    scraper = BoatraceScraper()
    pairs = list(scraper.crawl(daterange(start, end), venue_codes=venues))
    logger.info("取得レース数=%d", len(pairs))

    df = build_dataset(pairs)
    df = add_targets(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    logger.info("保存: %s rows=%d", out_path, len(df))


if __name__ == "__main__":
    main()
