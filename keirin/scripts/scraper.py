"""
競輪データ収集モジュール（楽天Kドリームス版）
keirin.kdreams.jp から日別・場別のレース結果を収集する
"""

import os
import re
import time
import json
import signal
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://keirin.kdreams.jp"
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}

# KCDコード -> 場名（日本語）
VENUE_CODES = {
    "11": "函館",   "12": "青森",   "13": "いわき平", "21": "弥彦",
    "22": "前橋",   "23": "取手",   "24": "宇都宮",   "25": "大宮",
    "26": "西武園", "27": "京王閣", "28": "立川",     "31": "松戸",
    "32": "千葉",   "34": "川崎",   "35": "平塚",     "36": "小田原",
    "37": "伊東",   "38": "静岡",   "42": "名古屋",   "43": "岐阜",
    "44": "大垣",   "45": "豊橋",   "46": "富山",     "47": "松阪",
    "48": "四日市", "51": "福井",   "53": "奈良",     "54": "向日町",
    "55": "和歌山", "56": "岸和田", "61": "玉野",     "62": "広島",
    "63": "防府",   "71": "高松",   "73": "小松島",   "74": "高知",
    "75": "松山",   "81": "小倉",   "83": "久留米",   "84": "武雄",
    "85": "佐世保", "86": "別府",   "87": "熊本",
}

BANK_LENGTH = {
    "函館": 333, "青森": 400, "いわき平": 400, "弥彦": 400,
    "前橋": 333, "取手": 400, "宇都宮": 333, "大宮": 400,
    "西武園": 400, "京王閣": 400, "立川": 400, "松戸": 333,
    "千葉": 333, "川崎": 333, "平塚": 500, "小田原": 333,
    "伊東": 333, "静岡": 500, "豊橋": 400, "岐阜": 400,
    "大垣": 400, "四日市": 333, "松阪": 400, "奈良": 400,
    "向日町": 333, "和歌山": 400, "岸和田": 400, "玉野": 400,
    "広島": 400, "防府": 400, "高松": 400, "小松島": 333,
    "高知": 333, "松山": 333, "小倉": 400, "久留米": 400,
    "武雄": 400, "佐世保": 400, "別府": 400, "熊本": 400,
    "名古屋": 400, "富山": 400, "福井": 400,
}

CLASS_ORDER = ["S1", "S2", "A1", "A2", "A3", "B1"]

_local = threading.local()
_stop_event = threading.Event()


def _get_session() -> requests.Session:
    if not hasattr(_local, "session"):
        s = requests.Session()
        s.headers.update(HEADERS)
        _local.session = s
    return _local.session


def fetch(url: str, retries: int = 3, timeout: int = 15) -> BeautifulSoup | None:
    session = _get_session()
    for i in range(retries):
        if _stop_event.is_set():
            return None
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            return BeautifulSoup(resp.text, "lxml")
        except Exception as e:
            if _stop_event.is_set():
                return None
            wait = 2 ** i * 3
            print(f"[fetch] {e} (試行 {i+1}/{retries}, {wait}秒待機)")
            time.sleep(wait)
    return None


def fetch_daily_races(date_str: str) -> list[dict]:
    """日付別の全レース一覧を取得 (date_str: YYYYMMDD)"""
    url = f"{BASE_URL}/raceresult/{date_str[:4]}/{date_str[4:6]}/{date_str[6:8]}/"
    soup = fetch(url)
    if soup is None:
        return []

    races = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "showResult" not in href:
            continue
        m = re.search(r"/(\w+)/racedetail/(\d{16})/", href)
        if not m:
            continue
        slug, race_id = m.group(1), m.group(2)
        if race_id in seen:
            continue
        seen.add(race_id)

        kcd = race_id[:2]
        date = race_id[2:10]
        race_no = int(race_id[14:16])

        races.append({
            "venue_slug": slug,
            "race_id": race_id,
            "venue_code": kcd,
            "venue_name": VENUE_CODES.get(kcd, slug),
            "date": date,
            "race_no": race_no,
        })
    return races


def parse_rider_table(table) -> list[dict]:
    """Table 0 から選手情報を抽出"""
    rows = table.find_all("tr")
    riders = []
    for tr in rows[2:]:  # 最初の2行はヘッダー
        cols = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        if len(cols) < 23:
            continue
        try:
            car_no = int(cols[4])
        except (ValueError, IndexError):
            continue

        name_full = cols[5]
        # "中島 淳埼　玉/26/125" → name + pref/age/term
        m = re.match(r"(.+?)([^/]+)/(\d+)/(\d+)$", name_full)
        if m:
            player_name = m.group(1).strip()
            age = _safe_int(m.group(3))
            term = _safe_int(m.group(4))
        else:
            player_name = name_full
            age, term = None, None

        rider = {
            "car_no": car_no,
            "player_name": player_name,
            "age": age,
            "term": term,
            "class": cols[6],
            "kakushitsu": cols[7],
            "gear": _safe_float(cols[8]),
            "kyosoten": _safe_float(cols[9]),
            "win_rate": _safe_float(cols[20], divisor=100),
            "second_rate": _safe_float(cols[21], divisor=100),
            "third_rate": _safe_float(cols[22], divisor=100),
        }
        riders.append(rider)
    return riders


def parse_result_table(table) -> list[dict]:
    """Table 34 から着順を抽出"""
    rows = table.find_all("tr")
    finish = []
    for tr in rows[1:]:
        cols = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        if len(cols) < 4:
            continue
        try:
            rank = int(cols[1])
            car_no = int(cols[2])
            finish.append({"rank": rank, "car_no": car_no})
        except (ValueError, IndexError):
            continue
    return finish


def fetch_race_detail(race: dict) -> list[dict] | None:
    """レース詳細から選手＋着順データを取得"""
    url = f"{BASE_URL}/{race['venue_slug']}/racedetail/{race['race_id']}/?pageType=showResult"
    soup = fetch(url)
    if soup is None:
        return None

    tables = soup.find_all("table")
    if len(tables) < 35:
        return None

    riders = parse_rider_table(tables[0])
    finish_order = parse_result_table(tables[34])
    if not riders or not finish_order:
        return None

    venue_name = race["venue_name"]
    bank_length = BANK_LENGTH.get(venue_name, 400)

    records = []
    for r in riders:
        rank = next((f["rank"] for f in finish_order if f["car_no"] == r["car_no"]), None)
        records.append({
            **r,
            "line_no": 0,
            "line_size": 1,
            "is_line_leader": 0,
            "venue_code": race["venue_code"],
            "venue_name": venue_name,
            "bank_length": bank_length,
            "date": race["date"],
            "race_no": race["race_no"],
            "rank": rank,
            "win": 1 if rank == 1 else 0,
        })
    return records


def _collect_day(
    date_str: str,
    sleep_sec: float,
    stop_event: threading.Event,
) -> list[dict]:
    """1日分のデータを収集"""
    if stop_event.is_set():
        return []

    races = fetch_daily_races(date_str)
    if not races:
        return []

    records = []
    for race in races:
        if stop_event.is_set():
            break
        race_records = fetch_race_detail(race)
        if race_records:
            records.extend(race_records)
        time.sleep(sleep_sec)
    return records


def load_existing_records(filename: str = "raw_data.json") -> tuple[list[dict], str | None]:
    path = DATA_DIR / filename
    if not path.exists():
        return [], None
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if not records:
        return [], None
    latest_date = max(r["date"] for r in records)
    return records, latest_date


def save_records(records: list[dict], filename: str = "raw_data.json") -> Path:
    path = DATA_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\n保存完了: {path} ({len(records)}件)")
    return path


def collect_data(
    start_date: str,
    end_date: str,
    venue_codes: list[str] | None = None,  # 互換性のため残すが kdreams.jp では使用しない
    sleep_sec: float = 1.0,
    existing_records: list[dict] | None = None,
    checkpoint_days: int = 7,
    filename: str = "raw_data.json",
    workers: int = 4,
) -> list[dict]:
    """期間内の全レースデータを収集（kdreams.jp、日別並列）"""
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    total_days = (end - start).days + 1

    records = list(existing_records) if existing_records else []
    lock = threading.Lock()
    _stop_event.clear()

    def _handler(sig, frame):
        print("\n\n[中断] 保存して終了します...")
        _stop_event.set()
        with lock:
            save_records(records, filename)
        os._exit(0)
    signal.signal(signal.SIGINT, _handler)

    # 1日あたり概算: 5場×7レース×1リクエスト + 1リクエスト(一覧)
    est_sec = total_days * (5 * 7 + 1) * sleep_sec / max(1, workers)
    print(f"収集日数: {total_days}日  並列数: {workers}日  概算所要時間: {est_sec/3600:.1f}時間")

    days_done = 0
    current = start
    date_list = []
    while current <= end:
        date_list.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    # 日付を並列で処理
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for i, date_str in enumerate(date_list):
            if _stop_event.is_set():
                break
            jitter = i * sleep_sec * 0.3 + random.uniform(0, sleep_sec * 0.3)
            futures[executor.submit(_collect_day_with_jitter,
                                     date_str, sleep_sec, _stop_event, jitter)] = date_str

        for future in as_completed(futures):
            date_str = futures[future]
            if _stop_event.is_set():
                break
            try:
                day_records = future.result()
                with lock:
                    records.extend(day_records)
                    n = len(day_records)
                    days_done += 1
                    print(f"  {date_str}: {n}件  [{days_done}/{total_days}日完了, 累計{len(records)}件]")

                    if checkpoint_days > 0 and days_done % checkpoint_days == 0:
                        print(f"  [チェックポイント] 保存中...")
                        save_records(records, filename)
            except Exception as e:
                print(f"  [エラー] {date_str}: {e}")

    save_records(records, filename)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    return records


def _collect_day_with_jitter(date_str, sleep_sec, stop_event, jitter):
    if jitter > 0:
        time.sleep(jitter)
    return _collect_day(date_str, sleep_sec, stop_event)


def _safe_float(text: str, divisor: float = 1.0) -> float | None:
    try:
        v = float(text.replace(",", "").strip())
        return v / divisor
    except Exception:
        return None


def _safe_int(text: str) -> int | None:
    try:
        return int(text.strip())
    except Exception:
        return None


if __name__ == "__main__":
    today = datetime.now()
    start = (today - timedelta(days=2)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    print(f"収集期間: {start} → {end}")
    records = collect_data(start, end, sleep_sec=2.0, workers=2)
    if not records:
        print("データが取得できませんでした")
