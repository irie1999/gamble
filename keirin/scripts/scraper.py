"""
競輪データ収集モジュール
keirin.jp の公式サイトからレース情報・選手成績・バンク情報を取得する
"""

import time
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://keirin.jp"
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; KeirinResearch/1.0)"
}

# 全国競輪場（場コード: 場名）
VENUE_CODES = {
    "11": "函館", "12": "青森", "13": "いわき平", "14": "弥彦",
    "15": "前橋", "16": "取手", "17": "宇都宮", "18": "大宮",
    "19": "西武園", "20": "京王閣", "21": "立川", "22": "松戸",
    "23": "千葉", "24": "川崎", "25": "平塚", "26": "小田原",
    "27": "伊東", "28": "静岡", "29": "豊橋", "30": "岐阜",
    "31": "大垣", "32": "四日市", "33": "松阪", "34": "奈良",
    "35": "向日町", "36": "和歌山", "37": "岸和田", "38": "玉野",
    "39": "広島", "40": "防府", "41": "高松", "42": "小松島",
    "43": "高知", "44": "松山", "45": "小倉", "46": "久留米",
    "47": "武雄", "48": "佐世保", "49": "別府", "50": "熊本",
}

# バンク周長別特性（短バンクほどインが有利）
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
}

# 級別
CLASS_ORDER = ["S1", "S2", "A1", "A2", "A3", "B1"]


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
    url = f"{BASE_URL}/pc/guest/keirinschedule/raceprogram/?jcd={venue_code}&hd={date}"
    soup = fetch(url)
    if soup is None:
        return []

    races = []
    seen = set()
    for a in soup.select("a[href*='rno=']"):
        href = a.get("href", "")
        m = re.search(r"rno=(\d+)", href)
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
    return races


def fetch_race_card(venue_code: str, date: str, race_no: int) -> dict | None:
    """出走表（選手・ライン・クラス）を取得"""
    url = (
        f"{BASE_URL}/pc/guest/keirinschedule/raceprogram/racecard/"
        f"?jcd={venue_code}&hd={date}&rno={race_no}"
    )
    soup = fetch(url)
    if soup is None:
        return None

    venue_name = VENUE_CODES.get(venue_code, "")
    bank_length = BANK_LENGTH.get(venue_name, 400)

    riders = []
    for row in soup.select("tbody tr"):
        cols = row.select("td")
        if len(cols) < 6:
            continue
        try:
            car_no = int(cols[0].get_text(strip=True))
            line_no_text = cols[1].get_text(strip=True)
            line_no = int(line_no_text) if line_no_text.isdigit() else 0
            player_name = cols[2].get_text(strip=True)
            player_id_tag = cols[2].select_one("a")
            player_id = ""
            if player_id_tag:
                m = re.search(r"toban=(\d+)", player_id_tag.get("href", ""))
                if m:
                    player_id = m.group(1)

            class_text = cols[3].get_text(strip=True)
            win_rate = _safe_float(cols[4].get_text(strip=True))
            second_rate = _safe_float(cols[5].get_text(strip=True))
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

    # ライン人数を各選手に付与
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
        f"{BASE_URL}/pc/guest/keirinschedule/raceresult/"
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
            car_no = int(cols[1].get_text(strip=True))
            finish_order.append({"rank": rank, "car_no": car_no})
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


def collect_data(
    start_date: str,
    end_date: str,
    venue_codes: list[str] | None = None,
    sleep_sec: float = 1.0,
) -> list[dict]:
    """期間内の全レースデータを収集"""
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
            venue_name = VENUE_CODES.get(venue_code, venue_code)
            print(f"  {venue_name}: {len(race_list)}レース")

            for race_info in race_list:
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
    today = datetime.now()
    start = (today - timedelta(days=3)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    print(f"収集期間: {start} → {end}")
    records = collect_data(start, end, sleep_sec=1.5)
    if records:
        save_records(records)
    else:
        print("データが取得できませんでした（開催なし or ネットワークエラー）")
