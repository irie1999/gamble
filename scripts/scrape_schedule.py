"""指定日のレース締切時刻を boatrace.jp raceindex から取得して CSV 保存。

bets CSV があれば、その候補レースの場のみ取得（高速化）。
無ければ races.parquet に該当日があれば全場取得。

出力: data/raw/race_schedule.csv (race_id, deadline_time)
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from threading import local

import pandas as pd

from src.scraper.boatrace_scraper import BoatraceScraper
from src.scraper.http_client import HttpClient
from src.utils.config import RAW_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TLS = local()


def _scraper() -> BoatraceScraper:
    if not hasattr(_TLS, "br"):
        _TLS.br = BoatraceScraper(client=HttpClient(interval_sec=0.3, timeout_sec=10))
    return _TLS.br


def _fetch_one(jcd: str, target: date) -> dict[int, str]:
    return _scraper().fetch_race_schedule(jcd, target)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="対象日 YYYY-MM-DD")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--only-bets", default=None,
                   help="bets CSV のパス。指定すると、その CSV にある場のみ取得")
    p.add_argument("--out", default=str(RAW_DIR / "race_schedule.csv"))
    args = p.parse_args()

    target = date.fromisoformat(args.date)
    out_path = Path(args.out)

    # 対象場を決定
    venues: set[str] = set()
    if args.only_bets:
        bets_path = Path(args.only_bets)
        if bets_path.exists():
            bets = pd.read_csv(bets_path, usecols=["race_id"])
            venues = set(bets["race_id"].astype(str).str.split("-").str[1].unique())
            logger.info("--only-bets フィルタ: 場 %s", sorted(venues))
        else:
            logger.warning("--only-bets で指定された CSV が無い: %s", bets_path)

    if not venues:
        # races.parquet からその日の場を抽出
        races_path = RAW_DIR / "races.parquet"
        if not races_path.exists():
            logger.error("races.parquet が無く、--only-bets も指定されていません")
            return
        races = pd.read_parquet(races_path, columns=["race_id", "race_date", "venue_code"])
        races["d"] = races["race_date"].astype(str).str.replace("-", "").str[:8]
        venues = set(
            races[races["d"] == target.strftime("%Y%m%d")]["venue_code"].astype(str).unique()
        )
        logger.info("races.parquet から場 %s を抽出", sorted(venues))

    if not venues:
        logger.info("対象場なし")
        return

    new_rows: list[dict] = []
    venues_list = sorted(venues)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for jcd, sched in zip(venues_list, ex.map(lambda v: _fetch_one(v, target), venues_list)):
            n = len(sched)
            if n == 0:
                logger.warning("⚠ %s: 締切時刻取得失敗（パーサが時刻を抽出できず）", jcd)
            else:
                logger.info("✓ %s: %d レース分取得", jcd, n)
            for rno, t in sched.items():
                rid = f"{target.strftime('%Y%m%d')}-{jcd}-{int(rno):02d}"
                new_rows.append({"race_id": rid, "deadline_time": t})

    if not new_rows:
        logger.warning("取得できた時刻なし。fetch_race_schedule のパーサ要調整")
        return

    new_df = pd.DataFrame(new_rows)
    if out_path.exists():
        old = pd.read_csv(out_path)
        merged = pd.concat([old, new_df], ignore_index=True).drop_duplicates(
            subset=["race_id"], keep="last"
        )
    else:
        merged = new_df
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    logger.info("保存: %s rows=%d (+%d)", out_path, len(merged), len(new_df))


if __name__ == "__main__":
    main()
