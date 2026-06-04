"""
特定レースのオッズページHTMLを解析してテーブル構造を確認するデバッグスクリプト
Usage:
  python keirin/scripts/debug_odds.py <race_id>
  例: python keirin/scripts/debug_odds.py 4320260505010001
"""

import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from keirin.scripts.scraper import fetch, VENUE_CODES, BASE_URL, ODDS_KAKESHIKI, _safe_int, _safe_float

import json


def analyze_odds_page(race_id: str, bet_type: str):
    """オッズページのテーブル構造をダンプ"""
    # race_id からvenueスラグを推定
    venue_code = race_id[:2]
    date = race_id[2:10]
    race_no = int(race_id[14:16])

    # venue_slug はkdreams.jpのURLスラグ (例: gifu, aomori)
    # スラグ一覧はodds_data.jsonから推測するか、手動指定
    slug_map = {
        "11": "maebashi", "12": "takasaki", "13": "utsunomiya", "14": "omiya",
        "15": "kishiwada", "16": "nishio", "17": "kyokaiko", "18": "tachikawa",
        "19": "matsudo", "20": "chiba", "21": "kawasaki", "22": "hiratsuka",
        "23": "odawara", "24": "ito", "25": "shizuoka", "26": "toyohashi",
        "27": "gifu", "28": "ogaki", "29": "yokkaichi", "30": "matsusaka",
        "31": "nara", "32": "mukomachi", "33": "wakayama", "34": "kishiwada",
        "35": "tamano", "36": "hiroshima", "37": "hofu", "38": "takamatsu",
        "39": "komatsushima", "40": "kochi", "41": "matsuyama", "42": "kokura",
        "43": "gifu", "44": "kurume", "45": "takeo", "46": "sasebo",
        "47": "beppu", "48": "kumamoto", "49": "nagoya", "50": "toyama",
        "51": "fukui", "52": "hakodate", "53": "aomori", "54": "iwakitaira",
        "55": "yahiko", "56": "fukushima",
    }
    slug = slug_map.get(venue_code, venue_code)

    ktype = ODDS_KAKESHIKI.get(bet_type, "tanshyo")
    url = f"{BASE_URL}/{slug}/racedetail/{race_id}/?pageType=odds&kakeshikiType={ktype}"
    print(f"\nURL: {url}")

    soup = fetch(url)
    if soup is None:
        print("フェッチ失敗")
        return

    tables = soup.find_all("table")
    print(f"テーブル数: {len(tables)}")

    for i, table in enumerate(tables):
        rows = table.find_all("tr")
        # 数値を含む行があるか確認
        numeric_rows = []
        for tr in rows:
            cols = [td.get_text(strip=True).replace(",", "") for td in tr.find_all(["td", "th"])]
            if len(cols) >= 2:
                car = _safe_int(cols[0])
                val = _safe_float(cols[-1])
                if car and val and val > 0:
                    numeric_rows.append((car, val, cols))

        if numeric_rows:
            print(f"\n  Table[{i}]: {len(numeric_rows)}行 × {len(rows[0].find_all(['td','th'])) if rows else 0}列")
            vals = [v for _, v, _ in numeric_rows]
            print(f"    car_nos: {[c for c, _, _ in numeric_rows]}")
            print(f"    values:  {[round(v, 1) for v in vals]}")
            print(f"    MIN={min(vals):.1f}  MAX={max(vals):.1f}  AVG={sum(vals)/len(vals):.1f}")


def main():
    if len(sys.argv) < 2:
        # odds_data.json から最初のrace_idを取得
        data_dir = Path(__file__).parent.parent / "data"
        odds_path = data_dir / "odds_data.json"
        if odds_path.exists():
            with open(odds_path, encoding="utf-8") as f:
                d = json.load(f)
            race_id = list(d.keys())[0]
            print(f"race_id未指定: {race_id} を使用")
        else:
            print("Usage: python debug_odds.py <race_id>")
            sys.exit(1)
    else:
        race_id = sys.argv[1]

    for bt in ["win", "place", "exacta", "quinella"]:
        print(f"\n{'='*60}")
        print(f"  賭け式: {bt}")
        print(f"{'='*60}")
        analyze_odds_page(race_id, bt)


if __name__ == "__main__":
    main()
