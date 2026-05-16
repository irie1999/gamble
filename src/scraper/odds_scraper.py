"""boatrace.jp のオッズページから単勝/3連単オッズを取得する。

URL:
  - 単勝/複勝: /oddstf?rno={R}&jcd={場}&hd={YYYYMMDD}
  - 3連単:     /odds3t?rno={R}&jcd={場}&hd={YYYYMMDD}

過去レースについては「締切時オッズ」が表示される。
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from datetime import date
from typing import Optional

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from src.scraper.http_client import HttpClient
from src.utils.config import BASE_URL, LANES
from src.utils.logger import get_logger

logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


@dataclass
class WinOdds:
    race_date: str
    venue_code: str
    race_no: int
    odds: dict[int, float]  # {lane: odds_win}


def _to_float(s: str) -> Optional[float]:
    s = (s or "").replace(",", "").strip()
    m = re.search(r"\d+\.?\d*", s)
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


class OddsScraper:
    """boatrace.jp のオッズページをスクレイプ。"""

    def __init__(self, client: Optional[HttpClient] = None) -> None:
        self.client = client or HttpClient()

    @staticmethod
    def _ymd(d: date | str) -> str:
        if isinstance(d, str):
            return d.replace("-", "").replace("/", "")
        return d.strftime("%Y%m%d")

    def fetch_win_odds(
        self,
        jcd: str,
        race_no: int,
        race_date: date | str,
        *,
        max_retries: int = 1,
        read_timeout: float = 20.0,
    ) -> WinOdds:
        """単勝オッズを取得。

        デフォルトは fail-fast（リトライ無し・timeout 20s）。バッチ用途で
        無駄なリトライを排除して総時間を短縮するため。1レースの取りこぼしを
        許さない用途（recommend_today 等）では呼び出し側で max_retries を増やし、
        timeout を短めにして外側でバックオフリトライするのが望ましい。
        """
        hd = self._ymd(race_date)
        url = f"{BASE_URL}/oddstf?rno={race_no}&jcd={jcd}&hd={hd}"
        html = self.client.get(url, max_retries=max_retries, read_timeout=read_timeout)
        # 中止・休場で「データがありません」を返すページは早期 return
        if "データがありません" in html:
            logger.debug("no-data page jcd=%s rno=%s d=%s", jcd, race_no, hd)
            return WinOdds(race_date=hd, venue_code=jcd, race_no=race_no, odds={})
        odds = self._parse_win_odds(html)
        return WinOdds(race_date=hd, venue_code=jcd, race_no=race_no, odds=odds)

    @staticmethod
    def _parse_win_odds(html: str) -> dict[int, float]:
        """単勝/複勝ページから {lane: 単勝オッズ} を抽出。

        boatrace.jp の単勝/複勝ページは tan/fuku 二つのテーブルが並ぶ構造。
        単勝側のセルには「odds-tan1_1」のようなクラスIDが付くため、
        まずそれで拾い、見つからなければ汎用フォールバックを使う。
        """
        soup = BeautifulSoup(html, "lxml")
        result: dict[int, float] = {}

        # 戦略1: data-rno + odds-tan{lane}_X クラスを持つセル
        for lane in LANES:
            cell = soup.select_one(f"[class*='oddsPoint{lane}'][class*='tan'], "
                                   f".oddsPoint{lane}, "
                                   f"[id*='tan_{lane}']")
            if cell:
                v = _to_float(cell.get_text())
                # 発売前の 0 や仮表示 0.x を除外（実オッズは必ず 1.0 以上）
                if v is not None and v >= 1.0:
                    result[lane] = v

        if len(result) == 6:
            return result

        # 戦略2: 単勝テーブル (h3に「単勝」を含む) の直近tableから6行
        result = {}
        for h in soup.find_all(["h2", "h3", "h4"]):
            if "単勝" in h.get_text():
                table = h.find_next("table")
                if not table:
                    continue
                for tr in table.select("tbody tr"):
                    tds = tr.select("td")
                    if len(tds) < 2:
                        continue
                    lane_text = re.sub(r"\D", "", tds[0].get_text())
                    if not lane_text:
                        continue
                    lane = int(lane_text)
                    if lane not in LANES:
                        continue
                    odds_val = _to_float(tds[-1].get_text())
                    # 0 や 1.0 未満は「発売前」「無効値」なので除外（後段で実オッズと混ざらないように）
                    if odds_val is not None and odds_val >= 1.0:
                        result[lane] = odds_val
                if result:
                    break

        if len(result) == 6:
            return result

        # 戦略3: ページ全体から「[1-6] [数値.数値]」のパターンを行単位で拾う
        result = {}
        for tr in soup.select("tr"):
            tds = [td.get_text(strip=True) for td in tr.select("td")]
            if len(tds) < 2:
                continue
            lane_text = re.sub(r"\D", "", tds[0])
            if not lane_text or len(lane_text) > 1:
                continue
            lane = int(lane_text)
            if lane not in LANES:
                continue
            v = _to_float(tds[-1])
            if v is not None and 1.0 <= v <= 9999.0:
                result[lane] = v

        return result
