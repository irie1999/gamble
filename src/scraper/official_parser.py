"""公式 番組表/競走成績 の固定幅テキストをパースする。

ファイル特性:
  - Shift-JIS / CRLF
  - 全角数字・全角スペースを含む（前段で NFKC 正規化）
  - 場ヘッダ:   "ボートレース<場名>" （場名に全角空白を含むことあり: "芦　屋"）
  - レース見出し: 全角数字 "  １Ｒ サンライズＶ ..." （NFKC後 "  1R ..."）
  - 艇行（番組表）の実例（NFKC後）:
      1 5018竹下大樹25福岡54A2 6.32 45.97 4.76 23.53 40 30.12 12 36.49 433 5 333    9
      ^ ^   ^   ^ ^  ^ ^      ^                ^                       ^         ^
      lane id  name age branch weight grade   全国勝率  ...     モーター ボート 今節成績  早見

注意:
  - 名前・年齢・支部・体重・級別の間に**スペースが無い**
  - 体重は **整数kg**（小数点なし）
  - 全角空白 "　" は NFKC で半角空白に変換される
"""
from __future__ import annotations

import re
import unicodedata
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


def _normalize(text: str) -> str:
    """全角→半角の正規化（漢字はそのまま）。"""
    return unicodedata.normalize("NFKC", text)


def _detect_date(text: str) -> Optional[str]:
    """日付検出。'YYYY年M月D日' / 'YY/MM/DD' / 'YYYY/MM/DD' に対応。"""
    m = re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
    if m:
        y, mo, d = m.groups()
        return f"{int(y):04d}{int(mo):02d}{int(d):02d}"
    m = re.search(r"(\d{2,4})/(\d{1,2})/(\d{1,2})", text)
    if m:
        y, mo, d = m.groups()
        yyyy = int(y) if int(y) > 1900 else (2000 + int(y) if int(y) < 80 else 1900 + int(y))
        return f"{yyyy:04d}{int(mo):02d}{int(d):02d}"
    return None


def _split_into_venues(text: str) -> list[str]:
    """テキストを「ボートレース<場名>」を境に場単位に分割。"""
    chunks = re.split(r"(?=ボートレース[^\d\n]+)", text)
    return [c for c in chunks if c.strip()]


def _detect_venue_code(chunk: str) -> Optional[str]:
    """場ヘッダから場コード（"01"〜"24"）を判定。"""
    m = re.search(r"ボートレース([^\n]+)", chunk)
    if not m:
        return None
    head = m.group(1)
    # 全角空白・通常空白を除去して比較
    cleaned = re.sub(r"\s+", "", head)
    for n, code in VENUE_NAME_TO_CODE.items():
        if cleaned.startswith(n):
            return code
    return None


_RACE_HEADER_RE = re.compile(r"^\s*(\d{1,2})R\b", re.MULTILINE)


def _split_into_races(venue_chunk: str) -> list[tuple[int, str]]:
    parts = []
    matches = list(_RACE_HEADER_RE.finditer(venue_chunk))
    for i, m in enumerate(matches):
        rno = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(venue_chunk)
        parts.append((rno, venue_chunk[start:end]))
    return parts


def _to_float(s: str) -> Optional[float]:
    s = s.strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if v > 0 else None  # 0.00 はデータ無し扱い


def _to_int(s: str) -> Optional[int]:
    s = s.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


# ---------- 番組表（Banzuke）パーサ ----------

# 艇行（NFKC正規化後）:
#   1 5018竹下大樹25福岡54A2 6.32 45.97 4.76 23.53 40 30.12 12 36.49 ...
#   ^ ^___^_____^__^___^__^^___ ^___ ^___ ^___ ^___ ^___ ^___ ^___
#   lane id    name age br wt grd  全国勝率 全国2率 当地勝率 当地2率 motorNo motor2 boatNo boat2
#
# name と age と branch と weight と grade は密着（スペース無し）
_ENTRY_RE = re.compile(
    r"^([1-6])\s+"
    r"(\d{4})"                # racer_id
    r"([^\d\n]+?)"            # name (Japanese, may contain spaces after NFKC)
    r"(\d{2})"                # age (2 digits)
    r"([^\d\n]+?)"            # branch (Japanese)
    r"(\d{2,3})"              # weight (kg, integer)
    r"(A1|A2|B1|B2)\s+"       # grade
    r"(\d+\.\d{2})\s+"        # 全国勝率
    r"(\d+\.\d{2})\s+"        # 全国2連率
    r"(\d+\.\d{2})\s+"        # 当地勝率
    r"(\d+\.\d{2})\s+"        # 当地2連率
    r"(\d+)\s+"               # motor_no
    r"(\d+\.\d{2})\s+"        # motor 2連率
    r"(\d+)\s+"               # boat_no
    r"(\d+\.\d{2})",          # boat 2連率
    re.MULTILINE,
)


def parse_banzuke(text: str) -> list[RaceCard]:
    text = _normalize(text)
    race_date = _detect_date(text) or ""
    cards: list[RaceCard] = []
    for venue_chunk in _split_into_venues(text):
        venue_code = _detect_venue_code(venue_chunk)
        if not venue_code:
            continue
        for rno, race_text in _split_into_races(venue_chunk):
            entries: list[RacerEntry] = []
            for m in _ENTRY_RE.finditer(race_text):
                # 名前と支部の両端の空白を整える
                name = re.sub(r"\s+", "", m.group(3))
                branch = re.sub(r"\s+", "", m.group(5))
                entries.append(RacerEntry(
                    lane=int(m.group(1)),
                    racer_id=m.group(2),
                    name=name,
                    grade=m.group(7),
                    branch=branch,
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


# ---------- 競走成績（K）パーサ ----------

# 着順行（NFKC後想定）: "01 4 5105富田恕生  ..."
# 形式は番組表に準じ、着順(2桁)・艇番(1桁)・登番(4桁)・氏名... が密着
_RANK_LINE_RE = re.compile(
    r"^\s*(\d{2}|S\d|F\d|失|妨|エ|転)\s+([1-6])\s+(\d{4})([^\d\n]+?)(?=\s|\d|$)",
    re.MULTILINE,
)

# 払戻表（NFKC後）。組番が "1-2-3" / "1=2=3" / "1" のいずれか。
# 例: "3連単 4-2-3 26,420" / "単勝 4 530"
_PAYOUT_BLOCKS = [
    ("trifecta", r"3連単\s+([1-6])-([1-6])-([1-6])\s+(\d{1,3}(?:,\d{3})*)", "-"),
    ("trio", r"3連複\s+([1-6])=([1-6])=([1-6])\s+(\d{1,3}(?:,\d{3})*)", "="),
    ("exacta", r"2連単\s+([1-6])-([1-6])\s+(\d{1,3}(?:,\d{3})*)", "-"),
    ("quinella", r"2連複\s+([1-6])=([1-6])\s+(\d{1,3}(?:,\d{3})*)", "="),
    ("win", r"単勝\s+([1-6])\s+(\d{1,3}(?:,\d{3})*)", None),
    ("place", r"複勝\s+([1-6])\s+(\d{1,3}(?:,\d{3})*)", None),
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


def _rank_token_to_int(s: str) -> Optional[int]:
    s = s.strip()
    if s.isdigit():
        return int(s)
    return None  # 失格・妨害・転覆等は None


def parse_results(text: str) -> list[RaceResult]:
    text = _normalize(text)
    race_date = _detect_date(text) or ""
    results: list[RaceResult] = []
    for venue_chunk in _split_into_venues(text):
        venue_code = _detect_venue_code(venue_chunk)
        if not venue_code:
            continue
        for rno, race_text in _split_into_races(venue_chunk):
            rows: list[RaceResultRow] = []
            for m in _RANK_LINE_RE.finditer(race_text):
                rank = _rank_token_to_int(m.group(1))
                rows.append(RaceResultRow(
                    lane=int(m.group(2)),
                    rank=rank,
                    racer_id=m.group(3),
                    name=re.sub(r"\s+", "", m.group(4)),
                    race_time_sec=None,
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
