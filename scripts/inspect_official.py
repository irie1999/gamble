"""公式LZHをダウンロード&解凍して、生テキストを見るCLI。

パース失敗時の調査用。例:
    python -m scripts.inspect_official --date 2024-08-15 --type B
    python -m scripts.inspect_official --date 2024-08-15 --type K --venue 12 --race 11
"""
from __future__ import annotations

import argparse
import re
from datetime import date

from src.scraper.official_downloader import OfficialDownloader


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--type", choices=["B", "K"], default="B", help="B=番組表, K=競走成績")
    p.add_argument("--venue", default=None, help="場名一部一致でフィルタ (例: 住之江)")
    p.add_argument("--race", type=int, default=None, help="R番号でフィルタ")
    p.add_argument("--head", type=int, default=200, help="先頭から表示する行数")
    args = p.parse_args()

    d = date.fromisoformat(args.date)
    dl = OfficialDownloader()
    text = dl.download_banzuke(d) if args.type == "B" else dl.download_results(d)

    if args.venue:
        # 「ボートレースXXX」までをスキップして該当場のみ抽出
        m = re.search(rf"(ボートレース{re.escape(args.venue)}[^\n]*\n.*?)(?=ボートレース|\Z)",
                      text, flags=re.DOTALL)
        text = m.group(1) if m else text

    if args.race is not None:
        # R番号で切り出し
        m = re.search(rf"({args.race}R[^\n]*\n(?:.*?\n)*?)(?=^\s*\d{{1,2}}R\s|\Z)",
                      text, flags=re.MULTILINE)
        if m:
            text = m.group(1)

    lines = text.splitlines()
    for line in lines[: args.head]:
        print(line)
    print(f"\n--- ({len(lines)} lines total) ---")


if __name__ == "__main__":
    main()
