"""boatrace公式の番組表/競走成績 LZH ファイルをダウンロード&解凍する。

URL（公式・古くから安定）:
    番組表: http://www1.mbrace.or.jp/od2/B/{YYYYMM}/b{YYMMDD}.lzh
    成績:  http://www1.mbrace.or.jp/od2/K/{YYYYMM}/k{YYMMDD}.lzh

LZH内部のテキストはShift-JISの固定幅。1日1ファイルで全24場・全レースが入る。

解凍バックエンド（自動検出・優先度順）:
    1. lhafile (Python, 推奨)
    2. 7z.exe / 7z (Windows: 7-Zip / Linux: p7zip)
    3. unar (macOS: brew install unar)
"""
from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path

from src.scraper.http_client import HttpClient
from src.utils.config import RAW_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

OFFICIAL_BASE = "http://www1.mbrace.or.jp/od2"


def banzuke_url(d: date) -> str:
    return f"{OFFICIAL_BASE}/B/{d:%Y%m}/b{d:%y%m%d}.lzh"


def result_url(d: date) -> str:
    return f"{OFFICIAL_BASE}/K/{d:%Y%m}/k{d:%y%m%d}.lzh"


# ---------- 解凍バックエンド ----------

def _try_lhafile(blob: bytes) -> str | None:
    try:
        import lhafile  # type: ignore
    except ImportError:
        return None
    archive = lhafile.Lhafile(io.BytesIO(blob))
    name = archive.namelist()[0]
    raw = archive.read(name)
    return raw.decode("shift_jis", errors="replace")


def _try_7z(blob: bytes) -> str | None:
    """7-Zipコマンドで解凍。Windowsなら C:\\Program Files\\7-Zip\\7z.exe など。"""
    sevenz = shutil.which("7z") or shutil.which("7z.exe")
    if not sevenz:
        # Windowsの標準インストール先をチェック
        for cand in (
            r"C:\Program Files\7-Zip\7z.exe",
            r"C:\Program Files (x86)\7-Zip\7z.exe",
        ):
            if Path(cand).exists():
                sevenz = cand
                break
    if not sevenz:
        return None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        lzh_path = tmp_path / "input.lzh"
        lzh_path.write_bytes(blob)
        # -y: 上書き許可, x: 展開（パス保持）, -o: 出力先
        proc = subprocess.run(
            [sevenz, "x", str(lzh_path), f"-o{tmp_path}", "-y"],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            logger.warning("7z exit=%d stderr=%s", proc.returncode, proc.stderr.decode(errors="replace"))
            return None
        # 解凍されたファイルを探す（input.lzh以外）
        files = [p for p in tmp_path.iterdir() if p.is_file() and p.suffix.lower() != ".lzh"]
        if not files:
            return None
        return files[0].read_bytes().decode("shift_jis", errors="replace")


def _try_unar(blob: bytes) -> str | None:
    unar = shutil.which("unar")
    if not unar:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        lzh_path = tmp_path / "input.lzh"
        lzh_path.write_bytes(blob)
        proc = subprocess.run(
            [unar, "-q", "-o", str(tmp_path), str(lzh_path)],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            return None
        files = [p for p in tmp_path.iterdir() if p.is_file() and p.suffix.lower() != ".lzh"]
        if not files:
            return None
        return files[0].read_bytes().decode("shift_jis", errors="replace")


def extract_lzh(blob: bytes) -> str:
    """LZHバイト列を解凍し、内部の最初のファイルをShift-JISでデコードして返す。

    バックエンドを自動選択:
      lhafile -> 7z -> unar
    すべて失敗した場合は RuntimeError。
    """
    for backend in (_try_lhafile, _try_7z, _try_unar):
        try:
            text = backend(blob)
        except Exception as e:
            logger.warning("%s 解凍失敗: %s", backend.__name__, e)
            continue
        if text is not None:
            logger.debug("使用バックエンド: %s", backend.__name__)
            return text
    raise RuntimeError(
        "LZH解凍バックエンドが見つかりません。以下のいずれかを導入してください:\n"
        "  - lhafile (推奨, Python): pip install lhafile  (要 C++ Build Tools)\n"
        "  - 7-Zip (推奨, Windows): https://www.7-zip.org/  → 7z.exe を PATH に追加\n"
        "  - p7zip (Linux): apt install p7zip-full\n"
        "  - unar (macOS): brew install unar"
    )


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
            return extract_lzh(cache_path.read_bytes())
        logger.info("download: %s", url)
        resp = self.client.session.get(url, timeout=(5.0, 30.0))
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
        return extract_lzh(resp.content)

    def download_banzuke(self, d: date) -> str:
        url = banzuke_url(d)
        cache = self.cache_dir / "B" / f"b{d:%y%m%d}.lzh"
        return self._fetch_with_cache(url, cache)

    def download_results(self, d: date) -> str:
        url = result_url(d)
        cache = self.cache_dir / "K" / f"k{d:%y%m%d}.lzh"
        return self._fetch_with_cache(url, cache)
