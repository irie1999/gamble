"""公式 番組表/競走成績 の固定幅テキストをパースする。

公開仕様（業界標準）:
  - Shift-JIS / CRLF
  - 番組表は ``BBGN`` で開始、``BEND`` で終了
  - 競走成績は ``KBGN`` で開始、``KEND`` で終了
  - 場ごとのセクションが連続し、各セクション内に12レースが並ぶ
  - レース行: ``  1R  ...`` のような行
  - 出走表（番組表）の艇行は半角スペース区切りで以下の順:
      艇番 / 登番 / 氏名 / 年齢 / 支部 / 体重 / 級別 /
      全国勝率 / 全国2連率 / 当地勝率 / 当地2連率 /
      モーターNo / モーター2連率 / ボートNo / ボート2連率
  - 競走成績の艇行: 着順 / 艇番 / 登番 / 氏名 / モーターNo / ボートNo / 展示タイム / 進入 / ST / レースタイム
  - 払戻はレース後段に表形式

注意:
  公式の固定幅は版によって微妙にズレるため、本実装は **空白区切りの正規表現ベース** で
  寛容にパースする。値が取れない場合は None。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from src.scraper.boatrace_scraper import (
    RaceCard,
    RaceResult,
    RaceResultRow,
    RacerEntry,
)


# 場名 -> 場コード の逆引き
VENUE_NAME_TO_CODE: dict[str, str] = {
    "桐生": "01", "戸田": "02", "江戸川": "03", "平和島": "04",
    "多摩川": "05", "浜名湖": "06", "蒲郡": "07", "常滑": "08",
    "津": "09", "三国": "10", "びわこ": "11", "住之江": "12",
    "尼崎": "13", "鳴門": "14", "丸亀": "15", "児島": "16",
    "宮島": "17", "徳山": "18", "下関": "19", "若松": "20",
    "芦屋": "21", "福岡": "22", "唐津": "23", "大村": "24",
}


# ---------- 共通ヘルパ ----------

def _detect_date(text: str) -> Optional[str]:
    """ヘッダから 'YY/MM/DD' を見つけて 'YYYYMMDD' に。"""
    m = re.search(r"(\d{2})/(\d{1,2})/(\d{1,2})", text)
    if not m:
        return None
    yy, mm, dd = m.groups()
    yyyy = 2000 + int(yy) if int(yy) < 80 else 1900 + int(yy)
    return f"{yyyy:04d}{int(mm):02d}{int(dd):02d}"


def _split_into_venues(text: str, *, begin: str, end: str) -> list[str]:
    """BBGN/KBGN ... BEND/KEND で囲まれたブロックを場単位の小ブロックに分割。"""
    body_match = re.search(rf"{begin}(.*?){end}", text, flags=re.DOTALL)
    body = body_match.group(1) if body_match else text
    # 各場の区切りに使われる「ボートレース<場名>」のような行で split
    chunks = re.split(r"(?=ボートレース\S+)", body)
    return [c for c in chunks if c.strip()]


_RACE_HEADER_RE = re.compile(r"^\s*(\d{1,2})R\b", re.MULTILINE)


def _split_into_races(venue_chunk: str) -> list[tuple[int, str]]:
    """場ブロックを (R番号, レーステキスト) のリストに分割。"""
    parts = []
    matches = list(_RACE_HEADER_RE.finditer(venue_chunk))
    for i, m in enumerate(matches):
        rno = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(venue_chunk)
        parts.append((rno, venue_chunk[start:end]))
    return parts


def _detect_venue_code(chunk: str) -> Optional[str]:
    m = re.search(r"ボートレース(\S+)", chunk)
    if not m:
        return None
    name = m.group(1).strip()
    # 「ボートレース住之江」「ボートレースびわこ」など
    for n, code in VENUE_NAME_TO_CODE.items():
        if name.startswith(n):
            return code
    return None


def _to_float(s: str) -> Optional[float]:
    s = s.strip()
    if not s or s == "0.00":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_int(s: str) -> Optional[int]:
    s = s.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


# ---------- 番組表（Banzuke）パーサ ----------

# 艇行の基本パターン。空白区切りで以下を抽出:
#   1 4123 山田太郎    35 東京 51.5 B1 5.50 35.50 5.20 33.10 12 35.50 56 30.20
_ENTRY_RE = re.compile(
    r"^\s*([1-6])\s+"
    r"(\d{4})\s+"
    r"(\S+?)\s+"
    r"(\d{1,2})\s+"
    r"(\S+?)\s+"
    r"(\d{2}\.\d)\s+"
    r"(A1|A2|B1|B2)\s+"
    r"(\d+\.\d{2})\s+"
    r"(\d+\.\d{2})\s+"
    r"(\d+\.\d{2})\s+"
    r"(\d+\.\d{2})\s+"
    r"(\d+)\s+"
    r"(\d+\.\d{2})\s+"
    r"(\d+)\s+"
    r"(\d+\.\d{2})",
    re.MULTILINE,
)


def parse_banzuke(text: str) -> list[RaceCard]:
    """番組表テキスト → RaceCard の配列。"""
    race_date = _detect_date(text) or ""
    cards: list[RaceCard] = []
    for venue_chunk in _split_into_venues(text, begin="BBGN", end="BEND"):
        venue_code = _detect_venue_code(venue_chunk)
        if not venue_code:
            continue
        for rno, race_text in _split_into_races(venue_chunk):
            entries: list[RacerEntry] = []
            for m in _ENTRY_RE.finditer(race_text):
                lane = int(m.group(1))
                entries.append(RacerEntry(
                    lane=lane,
                    racer_id=m.group(2),
                    name=m.group(3),
                    grade=m.group(7),
                    branch=m.group(5),
                    age=_to_int(m.group(4)),
                    weight=_to_float(m.group(6)),
                    win_rate_national=_to_float(m.group(8)),
                    place_rate_national=_to_float(m.group(9)),
                    win_rate_local=_to_float(m.group(10)),
                    place_rate_local=_to_float(m.group(11)),
                    motor_no=_to_int(m.group(12)),
                    motor_2rate=_to_float(m.group(13)),
                    boat_no=_to_int(m.group(14)),
                    boat_2rate=_to_float(m.group(15)),
                ))
            entries.sort(key=lambda e: e.lane)
            if entries:
                cards.append(RaceCard(
                    race_date=race_date,
                    venue_code=venue_code,
                    race_no=rno,
                    title=None,
                    distance_m=None,
                    entries=entries,
                ))
    return cards


# ---------- 競走成績（Result）パーサ ----------

# 着順行: 01 1 4123 山田太郎 ... 1'48"3 ... .15
_RANK_LINE_RE = re.compile(
    r"^\s*(\d{2})\s+([1-6])\s+(\d{4})\s+(\S+)",
    re.MULTILINE,
)
_TIME_RE = re.compile(r"(\d+)'(\d{2})\"(\d)")
_ST_RE = re.compile(r"(?:^|\s)(\.\d{2})(?:\s|$)")

# 払戻表: 「単勝」「2連単」など見出し+組合せ+金額
# 各エントリ: (label, regex, separator)
_PAYOUT_BLOCKS = [
    ("win", r"単勝\s+([1-6])\s+(\d{1,3}(?:,\d{3})*)", None),
    ("place", r"複勝\s+([1-6])\s+(\d{1,3}(?:,\d{3})*)", None),
    ("exacta", r"2連単\s+([1-6])-([1-6])\s+(\d{1,3}(?:,\d{3})*)", "-"),
    ("quinella", r"2連複\s+([1-6])=([1-6])\s+(\d{1,3}(?:,\d{3})*)", "="),
    ("trifecta", r"3連単\s+([1-6])-([1-6])-([1-6])\s+(\d{1,3}(?:,\d{3})*)", "-"),
    ("trio", r"3連複\s+([1-6])=([1-6])=([1-6])\s+(\d{1,3}(?:,\d{3})*)", "="),
]


def _parse_payouts(race_text: str) -> dict[str, list[tuple[str, int]]]:
    out: dict[str, list[tuple[str, int]]] = {}
    for key, pat, sep in _PAYOUT_BLOCKS:
        for m in re.finditer(pat, race_text):
            groups = m.groups()
            amount = int(groups[-1].replace(",", ""))
            nums = groups[:-1]
            combo = nums[0] if sep is None else sep.join(nums)
            out.setdefault(key, []).append((combo, amount))
    return out


def parse_results(text: str) -> list[RaceResult]:
    race_date = _detect_date(text) or ""
    results: list[RaceResult] = []
    for venue_chunk in _split_into_venues(text, begin="KBGN", end="KEND"):
        venue_code = _detect_venue_code(venue_chunk)
        if not venue_code:
            continue
        for rno, race_text in _split_into_races(venue_chunk):
            rows: list[RaceResultRow] = []
            for m in _RANK_LINE_RE.finditer(race_text):
                rank = _to_int(m.group(1))
                lane = int(m.group(2))
                racer_id = m.group(3)
                name = m.group(4)
                rows.append(RaceResultRow(
                    lane=lane,
                    rank=rank,
                    racer_id=racer_id,
                    name=name,
                    race_time_sec=None,  # 必要なら後で補強
                    start_timing=None,
                ))
            payouts = _parse_payouts(race_text)
            if rows or payouts:
                results.append(RaceResult(
                    race_date=race_date,
                    venue_code=venue_code,
                    race_no=rno,
                    rows=rows,
                    payouts=payouts,
                ))
    return results
