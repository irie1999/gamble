"""単勝/複勝オッズページの生HTMLを取得して中身を確認する診断ツール。

使い方:
    python -m scripts.inspect_odds --venue 12 --race 1 --date 2024-08-15
    python -m scripts.inspect_odds --venue 12 --race 1 --date 2024-08-15 --raw   # 生HTML全文
"""
from __future__ import annotations

import argparse
from datetime import date

from src.scraper.http_client import HttpClient
from src.scraper.odds_scraper import OddsScraper
from src.utils.config import BASE_URL


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--venue", required=True, help="場コード 例: 12")
    p.add_argument("--race", type=int, required=True)
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--raw", action="store_true", help="生HTML全文を表示")
    args = p.parse_args()

    d = date.fromisoformat(args.date)
    hd = d.strftime("%Y%m%d")
    url = f"{BASE_URL}/oddstf?rno={args.race}&jcd={args.venue}&hd={hd}"
    print(f"URL: {url}\n")

    client = HttpClient()
    html = client.get(url)
    print(f"HTML length: {len(html)} bytes\n")
    if args.raw:
        print(html)
        return

    # パース結果
    scraper = OddsScraper(client=client)
    odds = OddsScraper._parse_win_odds(html)
    print(f"Parsed win odds: {odds}")
    if len(odds) != 6:
        print("\n⚠️ 6艇分そろっていません。--raw で生HTML確認推奨。")


if __name__ == "__main__":
    main()
