"""
keirin.jp からオッズ・払戻金を取得するモジュール

API構造:
  ページHTML内 jsonData["PC0201"]["C0201data"]["C0201race"][i]["encParaR"] → encp
  オッズAPI: GET /pc/json?kake={N}&mode=0&encp={encp}&type=JST011
  払戻金:   jsonData["PJ0326"]["haraiGakuSubData"] (completed races)

kake番号:
  2=2車単, 4=2車複, 5=ワイド, 6=3連単, 7=3連複
"""

import re
import json
import time
import threading
import requests

KEIRIN_JP_BASE = "https://keirin.jp/pc"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9",
    "Referer": "https://keirin.jp/pc/racelive",
}

# 賭け式 → kake番号
BET_TYPE_TO_KAKE = {
    "exacta":   2,  # 2車単
    "quinella": 4,  # 2車複
    "wide":     5,  # ワイド
    "trifecta": 6,  # 3連単
    "trio":     7,  # 3連複
}

# keirin.jp KCD（JKA標準場コード）= scraper.py の venue_code と同一
VENUE_TO_KCD = {
    "函館":   11, "青森":   12, "いわき平": 13, "弥彦": 21,
    "前橋":   22, "取手":   23, "宇都宮":   24, "大宮": 25,
    "西武園": 26, "京王閣": 27, "立川":     28,
    "松戸":   31, "千葉":   32, "川崎":     34, "平塚": 35,
    "小田原": 36, "伊東":   37, "静岡":     38,
    "名古屋": 42, "岐阜":   43, "大垣":     44, "豊橋": 45,
    "富山":   46, "松阪":   47, "四日市":   48,
    "福井":   51, "奈良":   53, "向日町":   54,
    "和歌山": 55, "岸和田": 56,
    "玉野":   61, "広島":   62, "防府":     63,
    "高松":   71, "小松島": 73, "高知":     74, "松山": 75,
    "小倉":   81, "久留米": 83, "武雄":     84,
    "佐世保": 85, "別府":   86, "熊本":     87,
}

# scraper.py venue_code（文字列）→ KCD（整数）
VENUE_CODE_TO_KCD = {str(kcd): kcd for kcd in VENUE_TO_KCD.values()}


_tls = threading.local()

def _get_session() -> requests.Session:
    """スレッドローカルなセッションを返す（TLSコネクション再利用）"""
    if not hasattr(_tls, "session"):
        s = requests.Session()
        s.headers.update(HEADERS)
        _tls.session = s
    return _tls.session


def _fetch_html(url: str, retries: int = 3) -> str | None:
    s = _get_session()
    for i in range(retries):
        try:
            r = s.get(url, timeout=15)
            if r.status_code == 200:
                return re.sub(r'<\?xml[^>]+\?>', '',
                              r.content.decode("utf-8", errors="replace"))
        except Exception:
            _tls.session = None  # セッションをリセット
            s = _get_session()
            time.sleep(2 ** i)
    return None


def _extract_json_data(html: str) -> dict:
    """HTML内のjsonData[KEY] = {...}; を全て抽出"""
    result = {}
    for m in re.finditer(r'jsonData\["([^"]+)"\]\s*=\s*', html):
        key = m.group(1)
        start = m.end()
        depth = 0
        for i, ch in enumerate(html[start:]):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        result[key] = json.loads(html[start:start + i + 1])
                    except Exception:
                        pass
                    break
    return result


def _parse_kumiban(kumiban: str) -> tuple | None:
    """
    "2-1"   → (2, 1)       2車単/3連単（順序あり）
    "1=2"   → (1, 2)       2車複/ワイド（ソート済み）
    "2-1-9" → (2, 1, 9)    3連単
    "1=2=9" → (1, 2, 9)    3連複/ワイド
    """
    try:
        if "-" in kumiban:
            return tuple(int(x) for x in kumiban.split("-"))
        elif "=" in kumiban:
            return tuple(sorted(int(x) for x in kumiban.split("=")))
    except ValueError:
        pass
    return None


def _payout_to_odds(harai_gaku: str) -> float:
    """払戻金文字列 "1,920" → オッズ 19.2"""
    try:
        return int(harai_gaku.replace(",", "")) / 100
    except ValueError:
        return 0.0


def _parse_harai_data(harai: dict) -> dict[str, dict[tuple, float]]:
    """
    haraiGakuSubData → {bet_type: {sel_tuple: odds}}
    """
    FIELD_TO_BET = {
        "WT2HaraiGakuDispItemSubData": "exacta",
        "WH2HaraiGakuDispItemSubData": "quinella",
        "RT3HaraiGakuDispItemSubData": "trifecta",
        "RH3HaraiGakuDispItemSubData": "trio",
        "WHaraiGakuDispItemSubData":   "wide",
    }
    result: dict[str, dict] = {}
    for field, bet_type in FIELD_TO_BET.items():
        items = harai.get(field, [])
        if not items:
            continue
        d: dict[tuple, float] = {}
        for item in items:
            kb = _parse_kumiban(item.get("kumiBan", ""))
            odds = _payout_to_odds(item.get("haraiGaku", "0"))
            if kb and odds > 1.0:
                d[kb] = odds
        if d:
            result[bet_type] = d
    return result


def _parse_ozz_response(resp_json: dict, kake: int) -> dict[tuple, float]:
    """
    JSON APIレスポンス → {sel_tuple: odds}
    OZZ12=80.7 → {(1,2): 80.7}  (2車単)
    OZZ123=926.3 → {(1,2,3): 926.3}  (3連単)
    """
    unordered_kakes = {4, 5, 7}  # quinella, wide, trio
    data = resp_json.get("data", {})
    result: dict[tuple, float] = {}
    for field_val in data.values():
        if not isinstance(field_val, dict):
            continue
        for ozz_key, odds_val in field_val.items():
            if not ozz_key.startswith("OZZ"):
                continue
            if not isinstance(odds_val, (int, float)):
                continue
            if odds_val <= 1.0 or odds_val >= 9999:
                continue
            cars_str = ozz_key[3:]
            if len(cars_str) == 2:
                sel = (int(cars_str[0]), int(cars_str[1]))
            elif len(cars_str) == 3:
                sel = (int(cars_str[0]), int(cars_str[1]), int(cars_str[2]))
            else:
                continue
            if kake in unordered_kakes:
                sel = tuple(sorted(sel))
            result[sel] = odds_val
    return result


def fetch_race_page(kcd: int, date: str, race_no: int) -> dict | None:
    """
    raceresultページを取得してjsonDataを返す。
    Returns: {"PC0201": {...}, "PJ0326": {...}, ...} or None
    """
    url = (f"{KEIRIN_JP_BASE}/dfw/dataplaza/guest/raceresult"
           f"?KCD={kcd}&KBI={date}&RNO={race_no}")
    html = _fetch_html(url)
    if not html:
        return None
    return _extract_json_data(html)


def fetch_day_payouts(kcd: int, date: str) -> dict[int, dict]:
    """
    1日1会場の全レース払戻を取得する（高速バッチ版）。
    RNO=1 で PC0201 からレース数を取得し、各レースの払戻をまとめて返す。
    Returns: {race_no: payout_dict}  payout_dict は get_payout_odds と同形式
    """
    # まずRNO=1でその日のレース数を取得
    jdata_first = fetch_race_page(kcd, date, 1)
    if not jdata_first:
        return {}

    try:
        race_list = jdata_first["PC0201"]["C0201data"]["C0201race"]
        n_races = len(race_list)
    except (KeyError, TypeError):
        n_races = 12  # フォールバック

    results: dict[int, dict] = {}

    # RNO=1 の払戻があれば保存
    pay1 = get_payout_odds(jdata_first)
    if pay1:
        results[1] = pay1

    # RNO=2 以降を並行取得
    def _fetch_rno(rno: int):
        jdata = fetch_race_page(kcd, date, rno)
        if not jdata:
            return rno, {}
        return rno, get_payout_odds(jdata)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=min(n_races, 8)) as ex:
        futs = {ex.submit(_fetch_rno, rno): rno for rno in range(2, n_races + 1)}
        for fut in as_completed(futs):
            rno, pay = fut.result()
            if pay:
                results[rno] = pay

    return results


def get_race_encp(jdata: dict, race_no: int) -> str | None:
    """jsonData から指定レースのencParaRを取得"""
    try:
        races = jdata["PC0201"]["C0201data"]["C0201race"]
        return races[race_no - 1]["encParaR"]
    except (KeyError, IndexError, TypeError):
        return None


def get_payout_odds(jdata: dict) -> dict[str, dict[tuple, float]]:
    """jsonData["PJ0326"] から払戻金を取得してオッズに変換"""
    try:
        harai = jdata["PJ0326"]["haraiGakuSubData"]
        return _parse_harai_data(harai)
    except (KeyError, TypeError):
        return {}


def fetch_live_odds(encp: str, bet_types: list[str] | None = None) -> dict[str, dict[tuple, float]]:
    """
    JSON APIからライブオッズを取得
    Returns: {bet_type: {sel_tuple: odds}}
    """
    if bet_types is None:
        bet_types = list(BET_TYPE_TO_KAKE.keys())

    result: dict[str, dict] = {}
    session = requests.Session()
    session.headers.update(HEADERS)
    for bt in bet_types:
        kake = BET_TYPE_TO_KAKE.get(bt)
        if kake is None:
            continue
        url = f"{KEIRIN_JP_BASE}/json?kake={kake}&mode=0&encp={encp}&type=JST011"
        try:
            r = session.get(url, timeout=10)
            if r.status_code == 200:
                data = _parse_ozz_response(r.json(), kake)
                if data:
                    result[bt] = data
        except Exception:
            pass
        time.sleep(0.3)
    return result


def fetch_today_races() -> list[dict]:
    """
    raceliveページから本日の全レース一覧を取得
    Returns: [{"venue_name", "kcd", "encp", "race_no_str"}, ...]
    """
    html = _fetch_html(f"{KEIRIN_JP_BASE}/racelive")
    if not html:
        return []
    jdata = _extract_json_data(html)
    try:
        kaisai = jdata["PJ0312"]["kaisaiData"]
    except (KeyError, TypeError):
        return []

    races = []
    for venue in kaisai:
        races.append({
            "venue_name": venue.get("bkname", ""),
            "kcd": int(venue.get("bkCode", 0)),
            "encp": venue.get("raceUrlPrm", ""),
            "race_no_str": venue.get("raceNum", ""),
        })
    return races


def discover_kcd(venue_name: str, test_date: str, known_race_no: int = 1) -> int | None:
    """
    指定会場のKCDを総当たりで探す（初期設定用）
    """
    session = requests.Session()
    session.headers.update(HEADERS)
    for kcd in range(11, 100):
        url = (f"{KEIRIN_JP_BASE}/dfw/dataplaza/guest/raceresult"
               f"?KCD={kcd}&KBI={test_date}&RNO={known_race_no}")
        try:
            r = session.get(url, timeout=10)
            if r.status_code == 200 and len(r.content) > 20000:
                html = re.sub(r'<\?xml[^>]+\?>', '', r.content.decode("utf-8", errors="replace"))
                if venue_name in html:
                    print(f"  {venue_name} → KCD={kcd}")
                    return kcd
        except Exception:
            pass
        time.sleep(0.2)
    return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] == "today":
        print("=== 本日の開催レース ===")
        races = fetch_today_races()
        for r in races:
            print(f"  {r['venue_name']} {r['race_no_str']}  KCD={r['kcd']}  encp={r['encp'][:20]}...")
            # オッズ取得テスト
            if r["encp"]:
                odds = fetch_live_odds(r["encp"], ["exacta", "trifecta"])
                for bt, d in odds.items():
                    if d:
                        vals = list(d.values())
                        print(f"    {bt}: {len(d)}組合せ  MIN={min(vals):.1f}  MAX={max(vals):.1f}")

    elif len(sys.argv) >= 4:
        # 特定レースの払戻確認
        venue = sys.argv[1]
        date = sys.argv[2]
        race_no = int(sys.argv[3])
        kcd = VENUE_TO_KCD.get(venue)
        if kcd is None:
            print(f"KCD不明: {venue}. VENUE_TO_KCDに追加してください")
            sys.exit(1)
        print(f"\n{venue} {date} R{race_no} (KCD={kcd})")
        jdata = fetch_race_page(kcd, date, race_no)
        if jdata:
            # 払戻金
            payouts = get_payout_odds(jdata)
            for bt, d in payouts.items():
                print(f"  {bt}: {d}")
            # ライブオッズ (encParaR使用)
            encp = get_race_encp(jdata, race_no)
            if encp:
                print(f"\n  encp: {encp[:30]}...")
                odds = fetch_live_odds(encp, ["exacta", "quinella"])
                for bt, d in odds.items():
                    if d:
                        vals = list(d.values())
                        print(f"  {bt}: {len(d)}組合せ  AVG={sum(vals)/len(vals):.1f}")
        else:
            print("データ取得失敗")

    else:
        print("Usage:")
        print("  python keirin_jp.py today")
        print("  python keirin_jp.py <venue> <YYYYMMDD> <race_no>")
        print("  例: python keirin_jp.py 高知 20260201 1")
