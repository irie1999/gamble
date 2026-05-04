"""
競輪データ収集モジュール
keirin.jp の公式サイトからレース情報・選手成績・バンク情報を取得する
"""

import os
import time
import json
import re
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://keirin.jp"
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
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://keirin.jp/",
}

# 全国競輪場（KCDコード: 場名）
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


def fetch(url: str, retries: int = 3, timeout: int = 10) -> BeautifulSoup | None:
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


def fetch_race_list(venue_code: str, date: str) -> list[dict]:
    """指定場・日付のレース番号一覧を取得"""
    url = f"{BASE_URL}/pc/dfw/dataplaza/guest/racelist?KBI={date}&KCD={venue_code}"
    soup = fetch(url)
    if soup is None:
        return []

    seen = set()
    races = []
    # RNO= を含むリンクからレース番号を収集
    for a in soup.select("a[href*='RNO=']"):
        href = a.get("href", "")
        m = re.search(r"RNO=(\d+)", href, re.IGNORECASE)
        if m:
            rno = int(m.group(1))
            if rno not in seen:
                seen.add(rno)
                races.append({
                    "venue_code": venue_code,
                    "venue_name": VENUE_CODES.get(venue_code, ""),
                    "date": date,
                    "race_no": rno,
                })
    return sorted(races, key=lambda r: r["race_no"])


def fetch_race_card(venue_code: str, date: str, race_no: int) -> dict | None:
    """出走表（選手・ライン・クラス）を取得"""
    url = (
        f"{BASE_URL}/pc/dfw/dataplaza/guest/raceprogram"
        f"?KCD={venue_code}&KST={date}&RNO={race_no}"
    )
    soup = fetch(url)
    if soup is None:
        return None

    venue_name = VENUE_CODES.get(venue_code, "")
    bank_length = BANK_LENGTH.get(venue_name, 400)

    riders = []

    # racecard_table クラスを優先、なければ tbody の tr を探す
    table = soup.select_one(".racecard_table")
    rows = table.select("tr") if table else soup.select("tbody tr")

    for row in rows:
        cols = row.select("td")
        if len(cols) < 5:
            continue
        try:
            car_no_text = cols[0].get_text(strip=True)
            if not car_no_text.isdigit():
                continue
            car_no = int(car_no_text)

            line_no_text = cols[1].get_text(strip=True)
            line_no = int(line_no_text) if line_no_text.isdigit() else 0

            player_name = cols[2].get_text(strip=True)
            player_id_tag = cols[2].select_one("a")
            player_id = ""
            if player_id_tag:
                href = player_id_tag.get("href", "")
                m = re.search(r"(\d{4,5})", href)
                if m:
                    player_id = m.group(1)

            class_text = cols[3].get_text(strip=True) if len(cols) > 3 else ""
            win_rate = _safe_float(cols[4].get_text(strip=True)) if len(cols) > 4 else None
            second_rate = _safe_float(cols[5].get_text(strip=True)) if len(cols) > 5 else None
            third_rate = _safe_float(cols[6].get_text(strip=True)) if len(cols) > 6 else None

            riders.append({
                "car_no": car_no,
                "line_no": line_no,
                "player_id": player_id,
                "player_name": player_name,
                "class": class_text,
                "win_rate": win_rate,
                "second_rate": second_rate,
                "third_rate": third_rate,
            })
        except Exception:
            continue

    if not riders:
        return None

    # ライン情報の付与
    line_counts = {}
    for r in riders:
        ln = r["line_no"]
        line_counts[ln] = line_counts.get(ln, 0) + 1

    for r in riders:
        r["line_size"] = line_counts.get(r["line_no"], 1)
        r["is_line_leader"] = 1 if r["line_no"] > 0 and r["car_no"] == min(
            [x["car_no"] for x in riders if x["line_no"] == r["line_no"]]
        ) else 0

    return {
        "venue_code": venue_code,
        "venue_name": venue_name,
        "bank_length": bank_length,
        "date": date,
        "race_no": race_no,
        "riders": riders,
    }


def fetch_race_result(venue_code: str, date: str, race_no: int) -> dict | None:
    """レース結果（着順）を取得"""
    url = (
        f"{BASE_URL}/pc/dfw/dataplaza/guest/raceresult"
        f"?KCD={venue_code}&KBI={date}&RNO={race_no}"
    )
    soup = fetch(url)
    if soup is None:
        return None

    finish_order = []

    table = soup.select_one(".result_table")
    rows = table.select("tr") if table else soup.select("tbody tr")

    for row in rows:
        cols = row.select("td")
        if len(cols) < 2:
            continue
        try:
            rank_text = cols[0].get_text(strip=True)
            car_text = cols[1].get_text(strip=True)
            if not rank_text.isdigit() or not car_text.isdigit():
                continue
            finish_order.append({"rank": int(rank_text), "car_no": int(car_text)})
        except Exception:
            continue

    if not finish_order:
        return None

    winner = next((r["car_no"] for r in finish_order if r["rank"] == 1), None)
    return {
        "venue_code": venue_code,
        "date": date,
        "race_no": race_no,
        "finish_order": finish_order,
        "winner": winner,
    }


def _collect_venue_day(
    venue_code: str,
    date_str: str,
    sleep_sec: float,
    stop_event: threading.Event,
    jitter: float = 0.0,
) -> list[dict]:
    """1場1日分のデータを収集（スレッド内で実行）"""
    if jitter > 0:
        time.sleep(jitter)
    if stop_event.is_set():
        return []

    race_list = fetch_race_list(venue_code, date_str)
    if not race_list:
        return []

    records = []
    for race_info in race_list:
        if stop_event.is_set():
            break
        rno = race_info["race_no"]
        card = fetch_race_card(venue_code, date_str, rno)
        result = fetch_race_result(venue_code, date_str, rno)

        if card is None or result is None:
            time.sleep(sleep_sec)
            continue

        for rider in card["riders"]:
            cn = rider["car_no"]
            rank = next(
                (r["rank"] for r in result["finish_order"] if r["car_no"] == cn),
                None,
            )
            records.append({
                **rider,
                "venue_code": venue_code,
                "venue_name": card["venue_name"],
                "bank_length": card["bank_length"],
                "date": date_str,
                "race_no": rno,
                "rank": rank,
                "win": 1 if rank == 1 else 0,
            })
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
    venue_codes: list[str] | None = None,
    sleep_sec: float = 1.0,
    existing_records: list[dict] | None = None,
    checkpoint_days: int = 7,
    filename: str = "raw_data.json",
    workers: int = 4,
) -> list[dict]:
    if venue_codes is None:
        venue_codes = list(VENUE_CODES.keys())

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

    avg_venues_per_day = 5
    est_sec = total_days * max(1, avg_venues_per_day / workers) * 12 * 2 * sleep_sec
    print(f"収集日数: {total_days}日  並列数: {workers}場  概算所要時間: {est_sec/3600:.1f}時間")

    days_done = 0
    current = start

    while current <= end and not _stop_event.is_set():
        date_str = current.strftime("%Y%m%d")
        print(f"\n=== {date_str}  [{days_done+1}/{total_days}日目]  累計{len(records)}件 ===")

        import random
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _collect_venue_day, vc, date_str, sleep_sec, _stop_event,
                    jitter=i * sleep_sec * 0.5 + random.uniform(0, sleep_sec * 0.5),
                ): vc
                for i, vc in enumerate(venue_codes)
            }
            for future in as_completed(futures):
                vc = futures[future]
                if _stop_event.is_set():
                    break
                try:
                    venue_records = future.result()
                    if venue_records:
                        n_races = len(set(r["race_no"] for r in venue_records))
                        print(f"  {VENUE_CODES.get(vc, vc)}: {n_races}レース ({len(venue_records)}件)")
                        with lock:
                            records.extend(venue_records)
                except Exception as e:
                    print(f"  [エラー] {VENUE_CODES.get(vc, vc)}: {e}")

        days_done += 1

        if checkpoint_days > 0 and days_done % checkpoint_days == 0:
            print(f"  [チェックポイント] {days_done}日完了 → 保存中...")
            with lock:
                save_records(records, filename)

        current += timedelta(days=1)

    save_records(records, filename)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    return records


def _safe_float(text: str) -> float | None:
    try:
        return float(text.replace(",", "").strip())
    except Exception:
        return None


if __name__ == "__main__":
    today = datetime.now()
    start = (today - timedelta(days=3)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    print(f"収集期間: {start} → {end}")
    records = collect_data(start, end, sleep_sec=1.0, workers=4)
    if not records:
        print("データが取得できませんでした（開催なし or ネットワークエラー）")
