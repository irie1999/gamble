"""
競艇データ収集モジュール
boatrace.jp の公式サイトからレース情報・選手成績・モーター成績を取得する
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

BASE_URL = "https://www.boatrace.jp"
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
    "Referer": "https://www.boatrace.jp/",
}

# 全国24場コード
VENUE_CODES = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島",
    "05": "多摩川", "06": "浜名湖", "07": "蒲郡", "08": "常滑",
    "09": "津", "10": "三国", "11": "びわこ", "12": "住之江",
    "13": "尼崎", "14": "鳴門", "15": "丸亀", "16": "児島",
    "17": "宮島", "18": "徳山", "19": "下関", "20": "若松",
    "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村",
}

# スレッドごとに独立したセッションを持つ
_local = threading.local()
_stop_event = threading.Event()  # モジュール全体で共有する停止フラグ


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
    url = f"{BASE_URL}/owpc/pc/race/raceindex?jcd={venue_code}&hd={date}"
    soup = fetch(url)
    if soup is None:
        return []

    races = []
    for a in soup.select("a[href*='racelist']"):
        href = a.get("href", "")
        m = re.search(r"rno=(\d+)", href)
        if m:
            races.append({
                "venue_code": venue_code,
                "venue_name": VENUE_CODES.get(venue_code, ""),
                "date": date,
                "race_no": int(m.group(1)),
            })

    seen = set()
    unique = []
    for r in races:
        key = r["race_no"]
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def fetch_race_card(venue_code: str, date: str, race_no: int) -> dict | None:
    url = (
        f"{BASE_URL}/owpc/pc/race/racelist"
        f"?jcd={venue_code}&hd={date}&rno={race_no}"
    )
    soup = fetch(url)
    if soup is None:
        return None

    boats = []
    for row in soup.select("tbody.is-fs12 tr"):
        cols = row.select("td")
        if len(cols) < 10:
            continue
        try:
            boat_no = int(cols[0].get_text(strip=True))
            player_name = cols[3].get_text(strip=True)
            player_id_tag = cols[3].select_one("a")
            player_id = ""
            if player_id_tag:
                m = re.search(r"toban=(\d+)", player_id_tag.get("href", ""))
                if m:
                    player_id = m.group(1)

            class_text = cols[2].get_text(strip=True)
            national_win_rate = _safe_float(cols[5].get_text(strip=True))
            national_2rate = _safe_float(cols[6].get_text(strip=True))
            national_3rate = _safe_float(cols[7].get_text(strip=True))
            local_win_rate = _safe_float(cols[8].get_text(strip=True))
            local_2rate = _safe_float(cols[9].get_text(strip=True))
            motor_no = cols[10].get_text(strip=True) if len(cols) > 10 else ""
            motor_2rate = _safe_float(cols[11].get_text(strip=True)) if len(cols) > 11 else None
            boat_no_equip = cols[12].get_text(strip=True) if len(cols) > 12 else ""
            boat_2rate = _safe_float(cols[13].get_text(strip=True)) if len(cols) > 13 else None

            boats.append({
                "boat_no": boat_no,
                "player_id": player_id,
                "player_name": player_name,
                "class": class_text,
                "national_win_rate": national_win_rate,
                "national_2rate": national_2rate,
                "national_3rate": national_3rate,
                "local_win_rate": local_win_rate,
                "local_2rate": local_2rate,
                "motor_no": motor_no,
                "motor_2rate": motor_2rate,
                "boat_no_equip": boat_no_equip,
                "boat_2rate": boat_2rate,
            })
        except Exception:
            continue

    if not boats:
        return None

    return {
        "venue_code": venue_code,
        "venue_name": VENUE_CODES.get(venue_code, ""),
        "date": date,
        "race_no": race_no,
        "boats": boats,
    }


def fetch_exhibition_times(venue_code: str, date: str, race_no: int) -> dict[int, float]:
    url = (
        f"{BASE_URL}/owpc/pc/race/beforeinfo"
        f"?jcd={venue_code}&hd={date}&rno={race_no}"
    )
    soup = fetch(url)
    if soup is None:
        return {}

    times = {}
    for row in soup.select("tbody tr"):
        cols = row.select("td")
        if len(cols) < 3:
            continue
        try:
            boat_no = int(cols[0].get_text(strip=True))
            exhibit_time = _safe_float(cols[2].get_text(strip=True))
            if exhibit_time:
                times[boat_no] = exhibit_time
        except Exception:
            continue
    return times


def fetch_race_result(venue_code: str, date: str, race_no: int) -> dict | None:
    url = (
        f"{BASE_URL}/owpc/pc/race/raceresult"
        f"?jcd={venue_code}&hd={date}&rno={race_no}"
    )
    soup = fetch(url)
    if soup is None:
        return None

    finish_order = []
    for row in soup.select("tbody tr"):
        cols = row.select("td")
        if len(cols) < 2:
            continue
        try:
            rank = int(cols[0].get_text(strip=True))
            boat_no = int(cols[1].get_text(strip=True))
            finish_order.append({"rank": rank, "boat_no": boat_no})
        except Exception:
            continue

    if not finish_order:
        return None

    winner = next((r["boat_no"] for r in finish_order if r["rank"] == 1), None)
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
) -> list[dict]:
    """1場1日分のデータを収集（スレッド内で実行）"""
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

        exhibit_times = fetch_exhibition_times(venue_code, date_str, rno)

        for boat in card["boats"]:
            bn = boat["boat_no"]
            rank = next(
                (r["rank"] for r in result["finish_order"] if r["boat_no"] == bn),
                None,
            )
            records.append({
                **boat,
                "venue_code": venue_code,
                "venue_name": card["venue_name"],
                "date": date_str,
                "race_no": rno,
                "exhibit_time": exhibit_times.get(bn),
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
    """
    期間内の全レースデータを並列収集する
    workers: 同時並行で処理する場の数（デフォルト4）
    """
    if venue_codes is None:
        venue_codes = list(VENUE_CODES.keys())

    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    total_days = (end - start).days + 1

    records = list(existing_records) if existing_records else []
    lock = threading.Lock()
    _stop_event.clear()

    # Ctrl+C で即座に保存して終了
    def _handler(sig, frame):
        print("\n\n[中断] 保存して終了します...")
        _stop_event.set()
        with lock:
            save_records(records, filename)
        os._exit(0)
    signal.signal(signal.SIGINT, _handler)

    # 概算所要時間（並列化を考慮）
    avg_venues_per_day = 5
    est_sec = total_days * max(1, avg_venues_per_day / workers) * 12 * 3 * sleep_sec
    print(f"収集日数: {total_days}日  並列数: {workers}場  概算所要時間: {est_sec/3600:.1f}時間")

    days_done = 0
    current = start

    while current <= end and not _stop_event.is_set():
        date_str = current.strftime("%Y%m%d")
        print(f"\n=== {date_str}  [{days_done+1}/{total_days}日目]  累計{len(records)}件 ===")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_collect_venue_day, vc, date_str, sleep_sec, _stop_event): vc
                for vc in venue_codes
            }
            for future in as_completed(futures):
                vc = futures[future]
                if _stop_event.is_set():
                    break
                try:
                    venue_records = future.result()
                    if venue_records:
                        n_races = len(set((r["race_no"]) for r in venue_records))
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
