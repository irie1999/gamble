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

# ライン先頭役割
LEADER_ROLES = {"先行", "押え先行", "自力", "逃げ"}

# 賭け式 → kdreams.jp kakeshikiType パラメータ
ODDS_KAKESHIKI = {
    "win":      "tanshyo",
    "place":    "hukushyo",
    "exacta":   "2rentan",
    "quinella": "2hukurentan",
    "trifecta": "3rentan",
    "trio":     "3hukurentan",
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


def _parse_race_links(soup, require_page_type: str | None = None) -> list[dict]:
    """ページ内のレースリンクを解析して race dict のリストを返す"""
    races = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if require_page_type and require_page_type not in href:
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


def fetch_daily_races(date_str: str) -> list[dict]:
    """日付別の全レース一覧を取得（結果確定済み）(date_str: YYYYMMDD)"""
    url = f"{BASE_URL}/raceresult/{date_str[:4]}/{date_str[4:6]}/{date_str[6:8]}/"
    soup = fetch(url)
    if soup is None:
        return []
    return _parse_race_links(soup, require_page_type="showResult")


def fetch_daily_schedule(date_str: str) -> list[dict]:
    """日付別の全出走予定を取得（結果未確定も含む）(date_str: YYYYMMDD)"""
    url = f"{BASE_URL}/raceresult/{date_str[:4]}/{date_str[4:6]}/{date_str[6:8]}/"
    soup = fetch(url)
    if soup is None:
        return []
    # まず結果リンクで取得、なければ全racedetailリンクを取得
    races = _parse_race_links(soup, require_page_type="showResult")
    if not races:
        races = _parse_race_links(soup, require_page_type=None)
    return races


def parse_rider_table(table) -> list[dict]:
    """選手情報テーブルを解析。car_no が col[4] にある行を対象とする"""
    rows = table.find_all("tr")
    riders = []
    for tr in rows:
        cols = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        if len(cols) < 23:
            continue
        try:
            car_no = int(cols[4])
            if not (1 <= car_no <= 9):
                continue
        except (ValueError, IndexError):
            continue

        name_full = cols[5]
        # "中島 淳埼　玉/26/125" → name_pref / age / term
        # rsplit で末尾の /age/term を切り離す（姓名と都道府県は連結のまま保持）
        parts = name_full.rsplit("/", 2)
        if len(parts) == 3:
            player_name = parts[0].strip()
            age = _safe_int(parts[1])
            term = _safe_int(parts[2])
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


def _is_valid_rider_set(riders: list[dict]) -> bool:
    """有効なレース選手セットか確認: 車番1始まり連続・重複なし・6〜9人"""
    if not riders:
        return False
    car_nos = sorted(r["car_no"] for r in riders)
    n = len(car_nos)
    return (6 <= n <= 9 and
            len(car_nos) == len(set(car_nos)) and  # 重複なし
            car_nos == list(range(1, n + 1)))        # 1,2,...,N


def _find_rider_table(tables) -> list[dict]:
    """有効な選手テーブル（車番1〜N連続・重複なし）を最初に返す"""
    for table in tables:
        riders = parse_rider_table(table)
        if _is_valid_rider_set(riders):
            return riders
    return []


def parse_result_table(table) -> list[dict]:
    """着順テーブルを解析。

    ヘッダー行から着順・車番の列位置を特定し、データ行を読む。
    ヘッダーが見つからない場合は先頭2列をフォールバックとして使用。
    選手テーブル（列数≥15）は除外する。
    """
    rows = table.find_all("tr")
    if not rows:
        return []

    # 列数が多いテーブルは選手テーブルなので除外
    sample_cols = [c.get_text(strip=True) for c in rows[0].find_all(["td", "th"])]
    if len(sample_cols) >= 15:
        return []

    # ヘッダー行から着順・車番列を特定
    rank_col: int | None = None
    car_col: int | None = None
    for tr in rows:
        cols = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        for ci, text in enumerate(cols):
            normalized = text.replace("\n", "").replace("　", "").strip()
            if normalized in ("着", "着順") and rank_col is None:
                rank_col = ci
            if normalized in ("車", "車番") and car_col is None:
                car_col = ci
        if rank_col is not None and car_col is not None:
            break

    finish = []
    if rank_col is not None and car_col is not None:
        # ヘッダーで特定した列を使用
        for tr in rows:
            cols = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
            if len(cols) <= max(rank_col, car_col):
                continue
            try:
                rank = int(cols[rank_col])
                car_no = int(cols[car_col])
                if 1 <= rank <= 9 and 1 <= car_no <= 9:
                    finish.append({"rank": rank, "car_no": car_no})
            except (ValueError, IndexError):
                continue
    else:
        # フォールバック: 先頭2列を着順・車番として解釈
        for tr in rows:
            cols = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
            if len(cols) < 2:
                continue
            try:
                rank = int(cols[0])
                car_no = int(cols[1])
                if 1 <= rank <= 9 and 1 <= car_no <= 9:
                    finish.append({"rank": rank, "car_no": car_no})
            except (ValueError, IndexError):
                continue
    return finish


def _find_result_table(tables) -> list[dict]:
    """有効な着順テーブル（rank/car_no ともに重複なし・1位あり・3〜9人）を最初に返す"""
    for table in tables:
        finish = parse_result_table(table)
        ranks = {f["rank"] for f in finish}
        cars = {f["car_no"] for f in finish}
        if (len(finish) >= 3 and
                len(ranks) == len(finish) and
                len(cars) == len(finish) and
                1 in ranks):
            return finish
    return []


def parse_lineup_text(text: str) -> dict[int, dict]:
    """
    "3先行 1追込 8追込 ｜ 2押え先行 7追込 9追込" を解析
    Returns: {car_no: {line_no, line_size, is_line_leader}}
    """
    result = {}
    groups = re.split(r'[｜|／]', text)
    for line_no, group in enumerate(groups, 1):
        tokens = re.findall(r'(\d+)(先行|押え先行|追込|自力|マーク|番手|逃げ|差し|捲り)?', group)
        valid = [(int(c), r) for c, r in tokens if c.isdigit() and 1 <= int(c) <= 9]
        for i, (car_no, role) in enumerate(valid):
            is_leader = (i == 0) or bool(role and role in LEADER_ROLES)
            result[car_no] = {
                "line_no": line_no,
                "line_size": len(valid),
                "is_line_leader": 1 if is_leader else 0,
            }
    return result


def parse_lineup_from_soup(soup) -> dict[int, dict]:
    """HTMLから並び予想テキストを探して解析。見つからなければ空dict。"""
    # kdreams.jp の並び予想はテキストノードまたはdivに含まれる
    candidates = []

    # テキストノードから探す
    for node in soup.find_all(string=re.compile(r'\d+(?:先行|追込|押え先行)')):
        candidates.append(node.strip())

    # 要素のテキストから探す（div, p, td, span）
    for tag in soup.find_all(['div', 'p', 'td', 'span', 'li']):
        t = tag.get_text(strip=True)
        if re.search(r'\d+(?:先行|追込)', t) and len(t) < 200:
            candidates.append(t)

    # 最もライン区切りが多いものを選ぶ
    best, best_count = {}, 0
    for text in candidates:
        parsed = parse_lineup_text(text)
        if len(parsed) > best_count:
            best, best_count = parsed, len(parsed)

    return best if best_count >= 3 else {}


def parse_odds_table(soup, bet_type: str) -> dict:
    """
    オッズページのHTMLからオッズを抽出する。
    Returns: {selection_tuple: odds_float}
      単勝/複勝: {(car_no,): odds}
      2連単/2連複: {(car1, car2): odds}
      3連単/3連複: {(car1, car2, car3): odds}
    """
    odds: dict = {}
    tables = soup.find_all("table")

    if bet_type in ("win", "place"):
        # 単勝・複勝はページ別URL。car_no→odds テーブルを探す。
        # 有効条件: 3〜9エントリ (1レース分), オッズは1.0〜99.9
        for table in tables:
            t_odds: dict = {}
            for tr in table.find_all("tr"):
                cols = [td.get_text(strip=True).replace(",", "") for td in tr.find_all(["td", "th"])]
                if len(cols) < 2:
                    continue
                car = _safe_int(cols[0])
                val = _safe_float(cols[-1])
                if car and val and 1.0 < val < 100.0:
                    t_odds[(car,)] = val
            if 3 <= len(t_odds) <= 9:
                return t_odds

    elif bet_type in ("exacta", "quinella"):
        for table in tables:
            rows = table.find_all("tr")
            if len(rows) < 3:
                continue
            # ヘッダー行から列の車番を取得
            header_cells = rows[0].find_all(["td", "th"])
            col_cars = [_safe_int(c.get_text(strip=True)) for c in header_cells]
            for tr in rows[1:]:
                cells = tr.find_all(["td", "th"])
                if not cells:
                    continue
                row_car = _safe_int(cells[0].get_text(strip=True))
                if not row_car:
                    continue
                for j, td in enumerate(cells[1:], 1):
                    if j >= len(col_cars) or not col_cars[j]:
                        continue
                    col_car = col_cars[j]
                    if row_car == col_car:
                        continue
                    val = _safe_float(td.get_text(strip=True).replace(",", ""))
                    if not val or val <= 1.0:
                        continue
                    if bet_type == "exacta":
                        odds[(row_car, col_car)] = val
                    else:
                        key = tuple(sorted([row_car, col_car]))
                        if key not in odds:
                            odds[key] = val
            if len(odds) >= 3:
                return odds

    elif bet_type in ("trifecta", "trio"):
        # 3連系は1着ごとにページが分かれている場合があるため
        # テーブルが巨大な場合はフラットに全セルを走査
        for table in tables:
            rows = table.find_all("tr")
            if len(rows) < 3:
                continue
            header_cells = rows[0].find_all(["td", "th"])
            col_cars = [_safe_int(c.get_text(strip=True)) for c in header_cells]
            first_col_vals = [_safe_int(tr.find(["td","th"]).get_text(strip=True)) if tr.find(["td","th"]) else None for tr in rows[1:]]
            if not any(first_col_vals):
                continue
            for tr, row_car in zip(rows[1:], first_col_vals):
                if not row_car:
                    continue
                cells = tr.find_all(["td", "th"])
                for j, td in enumerate(cells[1:], 1):
                    if j >= len(col_cars) or not col_cars[j]:
                        continue
                    col_car = col_cars[j]
                    val = _safe_float(td.get_text(strip=True).replace(",", ""))
                    if not val or val <= 1.0:
                        continue
                    if bet_type == "trifecta":
                        # row=1着, col=2着 (3着は不明のためスキップ)
                        pass
                    else:
                        key = tuple(sorted([row_car, col_car]))
                        # trio は3人組なので不完全 → スキップ
                        pass
            # trio/trifecta は3着情報が必要のため現時点ではスキップ
            break

    return odds


def fetch_race_odds(race: dict, bet_types: list[str] | None = None) -> dict[str, dict]:
    """
    レースの実オッズを全賭け式分取得する。
    Returns: {bet_type: {selection_tuple: odds}}
    注: trifecta/trio は3着まで取得できないため省略
    """
    if bet_types is None:
        bet_types = ["win", "place", "exacta", "quinella"]
    result: dict[str, dict] = {}
    for bt in bet_types:
        ktype = ODDS_KAKESHIKI.get(bt)
        if not ktype:
            continue
        url = (f"{BASE_URL}/{race['venue_slug']}/racedetail/{race['race_id']}/"
               f"?pageType=odds&kakeshikiType={ktype}")
        soup = fetch(url)
        if soup is None:
            continue
        parsed = parse_odds_table(soup, bt)
        if parsed:
            result[bt] = parsed
    return result


def fetch_entry_detail(race: dict) -> list[dict] | None:
    """出走表から選手情報を取得（レース前・結果なし）"""
    soup_used = None
    for page_type in ["", "?pageType=showEntry", "?pageType=showResult"]:
        url = f"{BASE_URL}/{race['venue_slug']}/racedetail/{race['race_id']}/{page_type}"
        soup = fetch(url)
        if soup is None:
            continue
        tables = soup.find_all("table")
        if not tables:
            continue
        riders = _find_rider_table(tables)
        if riders:
            soup_used = soup
            break
    else:
        return None

    # 並び予想を解析
    lineup = parse_lineup_from_soup(soup_used) if soup_used else {}

    venue_name = race["venue_name"]
    bank_length = BANK_LENGTH.get(venue_name, 400)

    return [
        {
            **r,
            "line_no": lineup.get(r["car_no"], {}).get("line_no", 0),
            "line_size": lineup.get(r["car_no"], {}).get("line_size", 1),
            "is_line_leader": lineup.get(r["car_no"], {}).get("is_line_leader", 0),
            "venue_code": race["venue_code"],
            "venue_name": venue_name,
            "bank_length": bank_length,
            "date": race["date"],
            "race_no": race["race_no"],
            "rank": None,
            "win": 0,
        }
        for r in riders
    ]


def fetch_race_detail(race: dict) -> list[dict] | None:
    """レース詳細から選手＋着順データを取得"""
    url = f"{BASE_URL}/{race['venue_slug']}/racedetail/{race['race_id']}/?pageType=showResult"
    soup = fetch(url)
    if soup is None:
        return None

    tables = soup.find_all("table")
    riders = _find_rider_table(tables)
    finish_order = _find_result_table(tables)
    if not riders or not finish_order:
        return None

    # 並び予想を解析
    lineup = parse_lineup_from_soup(soup)

    venue_name = race["venue_name"]
    bank_length = BANK_LENGTH.get(venue_name, 400)

    records = []
    for r in riders:
        car_no = r["car_no"]
        rank = next((f["rank"] for f in finish_order if f["car_no"] == car_no), None)
        linfo = lineup.get(car_no, {"line_no": 0, "line_size": 1, "is_line_leader": 0})
        records.append({
            **r,
            "line_no": linfo["line_no"],
            "line_size": linfo["line_size"],
            "is_line_leader": linfo["is_line_leader"],
            "venue_code": race["venue_code"],
            "venue_name": venue_name,
            "bank_length": bank_length,
            "date": race["date"],
            "race_no": race["race_no"],
            "rank": rank,
            "win": 1 if rank == 1 else 0,
        })
    return records


def _odds_key_to_str(key: tuple) -> str:
    return "_".join(str(k) for k in key)


def _str_to_odds_key(s: str) -> tuple:
    sep = "_" if "_" in s else ","
    return tuple(int(x) for x in s.split(sep))


def serialize_odds(odds_by_type: dict) -> dict:
    return {bt: {_odds_key_to_str(k): v for k, v in d.items()} for bt, d in odds_by_type.items()}


def deserialize_odds(data: dict) -> dict:
    return {bt: {_str_to_odds_key(k): v for k, v in d.items()} for bt, d in data.items()}


def load_odds_data(filename: str = "odds_data.json") -> dict:
    """保存済みオッズデータを読み込む。戻り値: {race_id: {bet_type: {sel_tuple: odds}}}"""
    path = DATA_DIR / filename
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {rid: deserialize_odds(d) for rid, d in raw.items()}


def save_odds_data(odds: dict, filename: str = "odds_data.json") -> None:
    path = DATA_DIR / filename
    # 既存データとマージ
    existing = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
    existing.update({rid: serialize_odds(d) for rid, d in odds.items()})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False)


def _collect_day(
    date_str: str,
    sleep_sec: float,
    stop_event: threading.Event,
) -> tuple[list[dict], dict, int]:
    """1日分のデータを収集。戻り値: (records, {race_id: odds}, detail_races_count)"""
    if stop_event.is_set():
        return [], {}, 0

    races = fetch_daily_races(date_str)
    if not races:
        return [], {}, 0

    records = []
    odds_day = {}
    detail_races = 0
    for race in races:
        if stop_event.is_set():
            break
        race_records = fetch_race_detail(race)
        if race_records:
            records.extend(race_records)
            detail_races += 1
        # オッズ取得（単勝・複勝・2連単・2連複）
        odds = fetch_race_odds(race, bet_types=["win", "place", "exacta", "quinella"])
        if odds:
            odds_day[race["race_id"]] = odds
        time.sleep(sleep_sec)
    return records, odds_day, detail_races


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

        all_odds: dict = {}
        for future in as_completed(futures):
            date_str = futures[future]
            if _stop_event.is_set():
                break
            try:
                day_records, day_odds, detail_races = future.result()
                with lock:
                    records.extend(day_records)
                    all_odds.update(day_odds)
                    n = len(day_records)
                    days_done += 1
                    print(f"  {date_str}: {n}件 ({detail_races}R結果/オッズ{len(day_odds)}R)  [{days_done}/{total_days}日完了, 累計{len(records)}件]")

                    if checkpoint_days > 0 and days_done % checkpoint_days == 0:
                        print(f"  [チェックポイント] 保存中...")
                        save_records(records, filename)
                        save_odds_data(all_odds)
            except Exception as e:
                print(f"  [エラー] {date_str}: {e}")

    save_records(records, filename)
    save_odds_data(all_odds)
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
