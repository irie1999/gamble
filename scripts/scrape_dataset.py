"""指定期間の番組表＋競走成績を取得して parquet/csv で保存。

データソース:
    --source official (デフォルト): boatrace公式LZH（高速・安定）
    --source html: boatrace.jp HTMLスクレイプ（互換用、HTML改修に弱い）

使い方:
    python -m scripts.scrape_dataset --from 2024-08-15 --to 2024-08-15
    python -m scripts.scrape_dataset --from 2024-08-01 --to 2024-08-31 --venues 12 22
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.data.preprocessor import build_dataset, add_targets, build_payouts
from src.scraper.boatrace_scraper import BoatraceScraper
from src.scraper.official_downloader import OfficialDownloader
from src.scraper.official_parser import parse_banzuke, parse_results
from src.utils.config import RAW_DIR, VENUE_CODES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _crawl_official(dates, venue_codes):
    """公式LZHから (RaceCard, RaceResult) ペアをyield。"""
    dl = OfficialDownloader()
    venue_filter = set(venue_codes) if venue_codes else set(VENUE_CODES.keys())

    for d in dates:
        try:
            b_text = dl.download_banzuke(d)
        except Exception as e:
            logger.warning("番組表取得失敗 date=%s err=%s", d, e)
            continue
        try:
            k_text = dl.download_results(d)
        except Exception as e:
            logger.warning("成績取得失敗 date=%s err=%s", d, e)
            k_text = ""

        cards = parse_banzuke(b_text)
        results = parse_results(k_text) if k_text else []
        result_index = {(r.race_date, r.venue_code, r.race_no): r for r in results}

        for card in cards:
            if card.venue_code not in venue_filter:
                continue
            key = (card.race_date, card.venue_code, card.race_no)
            res = result_index.get(key)
            if res is None:
                # 結果未取得の場合は空のRaceResult
                from src.scraper.boatrace_scraper import RaceResult
                res = RaceResult(
                    race_date=card.race_date,
                    venue_code=card.venue_code,
                    race_no=card.race_no,
                )
            yield card, res
        logger.info("[%s] cards=%d results=%d (filtered to venues=%d)",
                    d, len(cards), len(results), len(venue_filter))


def _crawl_html(dates, venue_codes):
    scraper = BoatraceScraper()
    yield from scraper.crawl(dates, venue_codes=venue_codes)


def _merge_with_existing(new_df: pd.DataFrame, out_path: Path,
                         key_cols: list[str]) -> pd.DataFrame:
    """既存ファイルがあれば読み込み、key_cols で重複排除した上で結合。

    new_df 側を優先（同じ key の行は新しいもので上書き）する。
    """
    if not out_path.exists():
        return new_df
    # 空ファイル / 壊れたファイルでも新規取得分を巻き込まないように防御
    if out_path.stat().st_size == 0:
        return new_df
    try:
        if out_path.suffix == ".parquet":
            old = pd.read_parquet(out_path)
        else:
            old = pd.read_csv(out_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as e:
        # 既存ファイルが壊れていても、新規取得分は失わないようにする
        import logging
        logging.getLogger(__name__).warning(
            "既存ファイル %s の読込に失敗: %s — 新規取得分のみで上書き",
            out_path, e,
        )
        return new_df
    combined = pd.concat([old, new_df], ignore_index=True)
    # 後勝ち: new_df 側を keep="last" で残す
    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="date_from", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", required=True, help="YYYY-MM-DD")
    parser.add_argument("--venues", nargs="*", default=None, help="場コード (例: 01 12 22)")
    parser.add_argument("--source", choices=["official", "html"], default="official")
    parser.add_argument("--out", default=str(RAW_DIR / "races.parquet"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存ファイルを破棄して新規作成（デフォルトは差分追記）",
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.date_from)
    end = date.fromisoformat(args.date_to)

    if args.source == "official":
        pairs = list(_crawl_official(daterange(start, end), args.venues))
    else:
        pairs = list(_crawl_html(daterange(start, end), args.venues))
    logger.info("取得レース数=%d", len(pairs))

    df = build_dataset(pairs)
    df = add_targets(df)
    payouts_df = build_payouts(pairs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payouts_path = out_path.with_name(out_path.stem + "_payouts" + out_path.suffix)

    # 既存と統合（明示的に --overwrite された時のみ全置換）
    if not args.overwrite:
        before_main = len(df)
        before_pay = len(payouts_df)
        df = _merge_with_existing(df, out_path, key_cols=["race_id", "lane"])
        payouts_df = _merge_with_existing(
            payouts_df, payouts_path, key_cols=["race_id", "bet_type", "combo"]
        )
        logger.info("差分マージ: races %d→%d / payouts %d→%d",
                    before_main, len(df), before_pay, len(payouts_df))

    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
        payouts_df.to_parquet(payouts_path, index=False)
    else:
        df.to_csv(out_path, index=False)
        payouts_df.to_csv(payouts_path, index=False)
    logger.info("保存: %s rows=%d / payouts=%d", out_path, len(df), len(payouts_df))


if __name__ == "__main__":
    main()
