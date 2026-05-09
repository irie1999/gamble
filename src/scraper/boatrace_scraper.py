"""boatrace.jp 公式サイトのスクレイパー

提供URL（参考）:
  - 出走表:     /racelist?rno={R}&jcd={場}&hd={YYYYMMDD}
  - 直前情報:   /beforeinfo?rno={R}&jcd={場}&hd={YYYYMMDD}
  - レース結果: /raceresult?rno={R}&jcd={場}&hd={YYYYMMDD}
  - 開催一覧:   /raceindex?jcd={場}&hd={YYYYMMDD}
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Optional

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from src.scraper.http_client import HttpClient
from src.utils.config import BASE_URL, LANES, VENUE_CODES
from src.utils.logger import get_logger

logger = get_logger(__name__)

# boatrace.jp の HTML は XML 風宣言を含むため XMLParsedAsHTMLWarning が出るが
# HTMLとしてパースして問題ないので抑制する。
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


# ---------- データ構造 ----------

@dataclass
class RacerEntry:
    """出走表の選手1人分"""
    lane: int
    racer_id: str
    name: str
    grade: str           # A1/A2/B1/B2
    branch: str          # 支部
    age: Optional[int]
    weight: Optional[float]
    win_rate_national: Optional[float]    # 全国勝率
    win_rate_local: Optional[float]       # 当地勝率
    place_rate_national: Optional[float]  # 全国2連対率
    place_rate_local: Optional[float]     # 当地2連対率
    motor_no: Optional[int]
    motor_2rate: Optional[float]
    boat_no: Optional[int]
    boat_2rate: Optional[float]


@dataclass
class RaceCard:
    """1レース分の出走表"""
    race_date: str   # YYYYMMDD
    venue_code: str  # "01"〜"24"
    race_no: int
    title: Optional[str] = None
    distance_m: Optional[int] = None
    entries: list[RacerEntry] = field(default_factory=list)


@dataclass
class RaceResultRow:
    """レース結果1艇分"""
    lane: int
    rank: Optional[int]    # 着順（失格等は None）
    racer_id: Optional[str]
    name: Optional[str]
    race_time_sec: Optional[float]
    start_timing: Optional[float]


@dataclass
class RaceResult:
    race_date: str
    venue_code: str
    race_no: int
    rows: list[RaceResultRow] = field(default_factory=list)
    payouts: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    # payouts: {"trifecta": [("1-2-3", 1230)], "exacta": [...], ...}


# ---------- 解析ヘルパ ----------

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _to_float(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = _NUM_RE.search(text.replace(",", ""))
    return float(m.group()) if m else None


def _to_int(text: Optional[str]) -> Optional[int]:
    v = _to_float(text)
    return int(v) if v is not None else None


def _clean(text: Optional[str]) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip()


# ---------- スクレイパー本体 ----------

class BoatraceScraper:
    """boatrace.jp 公式の出走表・直前情報・結果を取得する。

    - 公式サイトのHTML構造に依存するため、サイト改修時はパーサ調整が必要。
    - レート制限・User-Agentは HttpClient 側で吸収。
    """

    def __init__(self, client: Optional[HttpClient] = None) -> None:
        self.client = client or HttpClient()

    # ----- URL構築 -----

    def _url(self, path: str, *, rno: Optional[int], jcd: str, hd: str) -> str:
        params = []
        if rno is not None:
            params.append(f"rno={rno}")
        params.append(f"jcd={jcd}")
        params.append(f"hd={hd}")
        return f"{BASE_URL}/{path}?{'&'.join(params)}"

    @staticmethod
    def _ymd(d: date | str) -> str:
        if isinstance(d, str):
            return d.replace("-", "").replace("/", "")
        return d.strftime("%Y%m%d")

    # ----- 開催一覧（その日の各場で何レースまで開催されているか） -----

    def fetch_race_index(self, jcd: str, race_date: date | str) -> list[int]:
        """指定日・指定場で開催されるレース番号一覧を返す。"""
        hd = self._ymd(race_date)
        url = self._url("raceindex", rno=None, jcd=jcd, hd=hd)
        html = self.client.get(url)
        soup = BeautifulSoup(html, "lxml")

        race_nos: set[int] = set()
        for a in soup.select("a[href*='racelist']"):
            href = a.get("href", "")
            m = re.search(r"rno=(\d+)", href)
            if m:
                race_nos.add(int(m.group(1)))
        return sorted(race_nos)

    # ----- 出走表 -----

    def fetch_race_card(self, jcd: str, race_no: int, race_date: date | str) -> RaceCard:
        hd = self._ymd(race_date)
        url = self._url("racelist", rno=race_no, jcd=jcd, hd=hd)
        html = self.client.get(url)
        return self._parse_race_card(html, jcd=jcd, race_no=race_no, hd=hd)

    @staticmethod
    def _parse_race_card(html: str, *, jcd: str, race_no: int, hd: str) -> RaceCard:
        soup = BeautifulSoup(html, "lxml")

        title_el = soup.select_one(".heading2_titleName, h2")
        title = _clean(title_el.get_text()) if title_el else None

        # レース距離（"1800m" 等）
        dist = None
        for el in soup.select(".heading2_title, .is-fs14, .heading2_titleDetail"):
            m = re.search(r"(\d{3,4})\s*m", el.get_text())
            if m:
                dist = int(m.group(1))
                break

        entries: list[RacerEntry] = []
        # 出走表は tbody 単位で1艇分
        for tbody in soup.select("table.is-w748 tbody, table.tbl_racelist tbody, .table1 tbody"):
            cells = tbody.select("td")
            if len(cells) < 4:
                continue
            text = _clean(tbody.get_text(" "))
            # 1艇分かどうかを艇番（1〜6）で簡易判定
            lane_el = tbody.select_one(".is-fs14, td.is-fs14, .table1_boatImage1Number")
            try:
                lane = int(_clean(lane_el.get_text())) if lane_el else None
            except ValueError:
                lane = None
            if lane not in LANES:
                continue

            # 選手登録番号と氏名
            racer_id = ""
            name = ""
            id_name_el = tbody.select_one(".is-fs11, .is-fs12, .is-fs14")
            if id_name_el:
                t = _clean(id_name_el.get_text())
                m = re.search(r"(\d{4})", t)
                if m:
                    racer_id = m.group(1)
                # 氏名はリンクから取れることが多い
                a = tbody.select_one("a")
                if a:
                    name = _clean(a.get_text())

            # 級別 / 支部 / 年齢 / 体重（テキストから抽出）
            grade_m = re.search(r"(A1|A2|B1|B2)", text)
            grade = grade_m.group(1) if grade_m else ""

            age = None
            weight = None
            age_w = re.search(r"(\d{2})歳[^\d]*([\d.]+)\s*kg", text)
            if age_w:
                age = int(age_w.group(1))
                weight = float(age_w.group(2))

            # 勝率類 / モーター / ボート
            nums = [_to_float(s) for s in re.findall(r"\d+\.\d+", text)]
            # サイトレイアウト依存。失敗時は None で埋める。
            win_rate_national = nums[0] if len(nums) > 0 else None
            place_rate_national = nums[1] if len(nums) > 1 else None
            win_rate_local = nums[2] if len(nums) > 2 else None
            place_rate_local = nums[3] if len(nums) > 3 else None
            motor_2rate = nums[4] if len(nums) > 4 else None
            boat_2rate = nums[5] if len(nums) > 5 else None

            motor_no = None
            boat_no = None
            mb = re.search(r"モーター[\s:：]*(\d+).*?ボート[\s:：]*(\d+)", text)
            if mb:
                motor_no = int(mb.group(1))
                boat_no = int(mb.group(2))

            entries.append(RacerEntry(
                lane=lane,
                racer_id=racer_id,
                name=name,
                grade=grade,
                branch="",
                age=age,
                weight=weight,
                win_rate_national=win_rate_national,
                win_rate_local=win_rate_local,
                place_rate_national=place_rate_national,
                place_rate_local=place_rate_local,
                motor_no=motor_no,
                motor_2rate=motor_2rate,
                boat_no=boat_no,
                boat_2rate=boat_2rate,
            ))

        # 念のため艇番でソート＆6艇でなければそのまま返す
        entries.sort(key=lambda e: e.lane)

        return RaceCard(
            race_date=hd,
            venue_code=jcd,
            race_no=race_no,
            title=title,
            distance_m=dist,
            entries=entries,
        )

    # ----- レース結果 -----

    def fetch_race_result(self, jcd: str, race_no: int, race_date: date | str) -> RaceResult:
        hd = self._ymd(race_date)
        url = self._url("raceresult", rno=race_no, jcd=jcd, hd=hd)
        html = self.client.get(url)
        return self._parse_race_result(html, jcd=jcd, race_no=race_no, hd=hd)

    @staticmethod
    def _parse_race_result(html: str, *, jcd: str, race_no: int, hd: str) -> RaceResult:
        soup = BeautifulSoup(html, "lxml")
        rows: list[RaceResultRow] = []

        # 着順テーブル
        for tr in soup.select("table.is-w495 tbody tr, table.is-w748 tbody tr"):
            tds = [_clean(td.get_text(" ")) for td in tr.select("td")]
            if len(tds) < 3:
                continue
            try:
                rank = int(tds[0])
            except ValueError:
                rank = None
            try:
                lane = int(tds[1])
            except (ValueError, IndexError):
                continue
            if lane not in LANES:
                continue

            racer_id = ""
            name = ""
            m = re.search(r"(\d{4})", tds[2] if len(tds) > 2 else "")
            if m:
                racer_id = m.group(1)
                name = tds[2].replace(racer_id, "").strip()

            race_time_sec = None
            start_timing = None
            for c in tds[3:]:
                if race_time_sec is None:
                    mt = re.match(r"(\d+)'(\d{2})\"(\d)", c)
                    if mt:
                        m_, s_, ds_ = map(int, mt.groups())
                        race_time_sec = m_ * 60 + s_ + ds_ / 10.0
                if start_timing is None:
                    mst = re.match(r"\.?(\d{2})$", c)
                    if mst:
                        start_timing = float("0." + mst.group(1))

            rows.append(RaceResultRow(
                lane=lane,
                rank=rank,
                racer_id=racer_id or None,
                name=name or None,
                race_time_sec=race_time_sec,
                start_timing=start_timing,
            ))

        # 払戻
        payouts: dict[str, list[tuple[str, int]]] = {}
        type_map = {
            "単勝": "win",
            "複勝": "place",
            "2連単": "exacta",
            "２連単": "exacta",
            "2連複": "quinella",
            "２連複": "quinella",
            "3連単": "trifecta",
            "３連単": "trifecta",
            "3連複": "trio",
            "３連複": "trio",
        }
        for tr in soup.select("table.is-w495 tr, table.is-w748 tr"):
            tds = [_clean(td.get_text(" ")) for td in tr.select("td")]
            if not tds:
                continue
            head = tds[0]
            key = type_map.get(head)
            if key is None:
                continue
            for combo, amount in zip(tds[1::2], tds[2::2]):
                amt = _to_int(amount)
                if amt is not None and combo:
                    payouts.setdefault(key, []).append((combo, amt))

        return RaceResult(
            race_date=hd, venue_code=jcd, race_no=race_no,
            rows=rows, payouts=payouts,
        )

    # ----- 期間スクレイプ（高水準API） -----

    def crawl(
        self,
        dates: Iterable[date | str],
        venue_codes: Optional[Iterable[str]] = None,
    ):
        """期間×場をループしてカード・結果をyieldする。"""
        venues = list(venue_codes) if venue_codes else list(VENUE_CODES.keys())
        for d in dates:
            for jcd in venues:
                try:
                    rnos = self.fetch_race_index(jcd, d)
                except Exception as e:
                    logger.warning("index失敗 jcd=%s date=%s err=%s", jcd, d, e)
                    continue
                for rno in rnos:
                    try:
                        card = self.fetch_race_card(jcd, rno, d)
                        result = self.fetch_race_result(jcd, rno, d)
                        yield card, result
                    except Exception as e:
                        logger.warning("レース取得失敗 jcd=%s rno=%s date=%s err=%s",
                                       jcd, rno, d, e)
                        continue
