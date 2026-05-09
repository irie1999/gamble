"""指定期間の単勝オッズをスクレイプし CSV/parquet で保存する。

使い方:
    python -m scripts.scrape_odds --from 2024-08-15 --to 2024-08-31
    python -m scripts.scrape_odds --from 2024-08-15 --to 2024-08-31 --venues 12

注意:
    - boatrace.jp HTML から取得するため遅い（1リクエスト1秒、1日あたり24場×12R = 288 req）。
    - 過去レースは「締切時オッズ」が表示される。
    - レート制限を守って節度ある利用を。
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.scraper.odds_scraper import OddsScraper
from src.utils.config import RAW_DIR, VENUE_CODES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _race_id(d_str: str, jcd: str, rno: int) -> str:
    return f"{d_str}-{jcd}-{int(rno):02d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="date_from", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", required=True, help="YYYY-MM-DD")
    parser.add_argument("--venues", nargs="*", default=None, help="場コード (例: 12)")
    parser.add_argument("--races", nargs="*", type=int, default=list(range(1, 13)), help="R番号")
    parser.add_argument("--out", default=str(RAW_DIR / "odds_win.csv"))
    args = parser.parse_args()

    start = date.fromisoformat(args.date_from)
    end = date.fromisoformat(args.date_to)
    venues = list(args.venues) if args.venues else list(VENUE_CODES.keys())

    scraper = OddsScraper()
    rows: list[dict] = []
    total_attempts = 0
    total_success = 0
    for d in daterange(start, end):
        d_str = d.strftime("%Y%m%d")
        for jcd in venues:
            for rno in args.races:
                total_attempts += 1
                try:
                    wo = scraper.fetch_win_odds(jcd, rno, d)
                except Exception as e:
                    logger.warning("odds失敗 jcd=%s rno=%s d=%s err=%s", jcd, rno, d, e)
                    continue
                if not wo.odds:
                    continue
                total_success += 1
                rid = _race_id(d_str, jcd, rno)
                for lane, odds in wo.odds.items():
                    rows.append({
                        "race_id": rid,
                        "lane": lane,
                        "odds_win": odds,
                    })
                if total_success % 50 == 0:
                    logger.info("progress: success=%d / attempts=%d", total_success, total_attempts)

    logger.info("collected: %d rows (success races=%d / attempted=%d)",
                len(rows), total_success, total_attempts)
    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    logger.info("保存: %s", out_path)


if __name__ == "__main__":
    main()
