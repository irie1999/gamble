"""boatrace公式の番組表/競走成績 LZH ファイルをダウンロード&解凍する。

URL（公式・古くから安定）:
    番組表: http://www1.mbrace.or.jp/od2/B/{YYYYMM}/b{YYMMDD}.lzh
    成績:  http://www1.mbrace.or.jp/od2/K/{YYYYMM}/k{YYMMDD}.lzh

LZH内部のテキストはShift-JISの固定幅。1日1ファイルで全24場・全レースが入る。
"""
from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import lhafile  # type: ignore

from src.scraper.http_client import HttpClient
from src.utils.config import RAW_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

OFFICIAL_BASE = "http://www1.mbrace.or.jp/od2"


def banzuke_url(d: date) -> str:
    return f"{OFFICIAL_BASE}/B/{d:%Y%m}/b{d:%y%m%d}.lzh"


def result_url(d: date) -> str:
    return f"{OFFICIAL_BASE}/K/{d:%Y%m}/k{d:%y%m%d}.lzh"


def _extract_lzh(blob: bytes) -> str:
    """LZHバイト列を解凍し、内部の最初のファイルをShift-JISでデコードして返す。"""
    bio = io.BytesIO(blob)
    archive = lhafile.Lhafile(bio)
    # 通常1ファイルだけ
    name = archive.namelist()[0]
    raw = archive.read(name)
    # 公式はShift-JIS。エラー時は cp932 で寛容に
    return raw.decode("shift_jis", errors="replace")


class OfficialDownloader:
    """番組表 / 競走成績 を1日単位で取得。

    cache_dir に LZH を保存し、再実行時はそれを使う（公式サーバへの再アクセスを避ける）。
    """

    def __init__(self, client: HttpClient | None = None, cache_dir: Path | None = None) -> None:
        self.client = client or HttpClient()
        self.cache_dir = Path(cache_dir or RAW_DIR / "official")
        (self.cache_dir / "B").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "K").mkdir(parents=True, exist_ok=True)

    def _fetch_with_cache(self, url: str, cache_path: Path) -> str:
        if cache_path.exists():
            logger.info("cache hit: %s", cache_path)
            return _extract_lzh(cache_path.read_bytes())
        logger.info("download: %s", url)
        # http_client.get は str を返すが、ここではバイナリが必要
        resp = self.client.session.get(url, timeout=(5.0, 30.0))
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
        return _extract_lzh(resp.content)

    def download_banzuke(self, d: date) -> str:
        url = banzuke_url(d)
        cache = self.cache_dir / "B" / f"b{d:%y%m%d}.lzh"
        return self._fetch_with_cache(url, cache)

    def download_results(self, d: date) -> str:
        url = result_url(d)
        cache = self.cache_dir / "K" / f"k{d:%y%m%d}.lzh"
        return self._fetch_with_cache(url, cache)
