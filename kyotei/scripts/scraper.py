"""
競艇データ収集モジュール
boatrace.jp の公式サイトからレース情報・選手成績・モーター成績を取得する
"""

import time
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.boatrace.jp"
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; KyoteiResearch/1.0)"
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


def fetch(url: str, retries: int = 3) -> BeautifulSoup | None:
    for i in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except Exception as e:
            print(f"[fetch] {e} (試行 {i+1}/{retries})")
            time.sleep(2 ** i)
    return None


def fetch_race_list(venue_code: str, date: str) -> list[dict]:
    """指定場・日付のレース一覧を取得 (date: YYYYMMDD)"""
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
    """出走表（選手・モーター・コース枠番）を取得"""
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
    """展示タイムを取得 {艇番: タイム}"""
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
    """レース結果（着順・払戻）を取得"""
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


def collect_data(
    start_date: str,
    end_date: str,
    venue_codes: list[str] | None = None,
    sleep_sec: float = 1.0,
) -> list[dict]:
    """
    期間内の全レースデータを収集してリストで返す
    start_date / end_date: YYYYMMDD
    """
    if venue_codes is None:
        venue_codes = list(VENUE_CODES.keys())

    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    records = []

    current = start
    while current <= end:
        date_str = current.strftime("%Y%m%d")
        print(f"\n=== {date_str} ===")

        for venue_code in venue_codes:
            race_list = fetch_race_list(venue_code, date_str)
            if not race_list:
                continue
            print(f"  {VENUE_CODES.get(venue_code, venue_code)}: {len(race_list)}レース")

            for race_info in race_list:
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

        current += timedelta(days=1)

    return records


def save_records(records: list[dict], filename: str = "raw_data.json") -> Path:
    path = DATA_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\n保存完了: {path} ({len(records)}件)")
    return path


def _safe_float(text: str) -> float | None:
    try:
        return float(text.replace(",", "").strip())
    except Exception:
        return None


if __name__ == "__main__":
    # 直近3日分・全場を収集するサンプル実行
    today = datetime.now()
    start = (today - timedelta(days=3)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    print(f"収集期間: {start} → {end}")
    records = collect_data(start, end, sleep_sec=1.5)
    if records:
        save_records(records)
    else:
        print("データが取得できませんでした（開催なし or ネットワークエラー）")
