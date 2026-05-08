"""
keirin.jp から過去レースの払戻金を一括取得してodds_data.jsonに保存する

Usage:
  python keirin/scripts/collect_payouts.py              # raw_data.jsonにある全レース
  python keirin/scripts/collect_payouts.py --date 20260504   # 特定日のみ
  python keirin/scripts/collect_payouts.py --overwrite       # 既存データも再取得
  python keirin/scripts/collect_payouts.py --workers 16      # 並列数指定（デフォルト8）

出力: keirin/data/odds_data.json
  {race_id: {bet_type: {sel_tuple_str: odds}}}
"""

import sys
import json
import time
import threading
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from keirin.scripts.keirin_jp import (
    fetch_race_page, get_payout_odds, VENUE_CODE_TO_KCD,
)

DATA_DIR = Path(__file__).parent.parent / "data"


def tuple_to_key(t: tuple) -> str:
    return "_".join(str(x) for x in t)


def load_raw() -> list[dict]:
    path = DATA_DIR / "raw_data.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_odds() -> dict:
    path = DATA_DIR / "odds_data.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_odds(data: dict) -> None:
    path = DATA_DIR / "odds_data.json"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def build_race_list(records: list[dict], target_date: str | None) -> list[dict]:
    """raw_dataからユニークなレース一覧を返す。race_idを直接使用する。"""
    seen = set()
    races = []
    for r in records:
        race_id = r.get("race_id")
        if not race_id:
            continue
        date = race_id[2:10]
        if target_date and date != target_date:
            continue
        if race_id not in seen:
            seen.add(race_id)
            races.append({
                "race_id": race_id,
                "kcd": race_id[:2],
                "date": date,
                "race_no": int(race_id[12:16]),
                "venue_name": r.get("venue_name", race_id[:2]),
            })
    return sorted(races, key=lambda x: x["race_id"])


def _fetch_one(race: dict, sleep_sec: float) -> tuple[str, dict | None, str]:
    """1レース分の払戻を取得。戻り値: (race_id, serializable_dict_or_None, status)"""
    race_id = race["race_id"]
    kcd = race["kcd"]
    date = race["date"]
    race_no = race["race_no"]
    try:
        jdata = fetch_race_page(kcd, date, race_no)
        if not jdata:
            time.sleep(sleep_sec)
            return race_id, None, "nodata"
        payouts = get_payout_odds(jdata)
        if not payouts:
            time.sleep(sleep_sec)
            return race_id, None, "nopayout"
        serializable = {
            bt: {tuple_to_key(k): v for k, v in d.items()}
            for bt, d in payouts.items()
        }
        time.sleep(sleep_sec)
        return race_id, serializable, "ok"
    except Exception as e:
        time.sleep(sleep_sec)
        return race_id, None, f"err:{e}"


def fetch_and_store(
    races: list[dict],
    existing: dict,
    overwrite: bool,
    workers: int = 8,
    sleep_sec: float = 0.1,
) -> dict:
    """払戻金を並列取得してodds_dataに追記する"""
    results = dict(existing)
    lock = threading.Lock()
    total = len(races)
    counters = {"ok": 0, "skip": 0, "fail": 0, "done": 0}

    # スキップ対象を除外
    to_fetch = []
    for race in races:
        if not overwrite and race["race_id"] in results and "trifecta" in results[race["race_id"]]:
            counters["skip"] += 1
        else:
            to_fetch.append(race)

    print(f"  取得対象: {len(to_fetch)}件  スキップ: {counters['skip']}件  並列数: {workers}")
    if not to_fetch:
        print("  全件スキップ済み")
        return results

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_fetch_one, race, sleep_sec): race for race in to_fetch}
        for fut in as_completed(futures):
            race_id, data, status = fut.result()
            race = futures[fut]
            venue = race["venue_name"]
            date = race["date"]
            rno = race["race_no"]

            with lock:
                counters["done"] += 1
                done = counters["done"]
                if status == "ok":
                    results[race_id] = data
                    counters["ok"] += 1
                    bet_summary = " ".join(f"{bt}:{len(d)}" for bt, d in data.items())
                    print(f"  [{done}/{len(to_fetch)}] {venue} {date} R{rno:02d} → OK  {bet_summary}")
                elif status == "nodata":
                    counters["fail"] += 1
                elif status == "nopayout":
                    counters["fail"] += 1
                else:
                    counters["fail"] += 1
                    print(f"  [{done}/{len(to_fetch)}] {venue} {date} R{rno:02d} → {status}")

                if done % 200 == 0:
                    save_odds(results)
                    print(f"  -- 中間保存 ({done}/{len(to_fetch)}) ok={counters['ok']} skip={counters['skip']} fail={counters['fail']} --")

    print(f"\nok={counters['ok']} skip={counters['skip']} fail={counters['fail']}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="特定日のみ取得 (YYYYMMDD)")
    parser.add_argument("--overwrite", action="store_true", help="既存データも再取得")
    parser.add_argument("--workers", type=int, default=8, help="並列数（デフォルト: 8）")
    parser.add_argument("--sleep", type=float, default=0.1, help="リクエスト間隔秒（デフォルト: 0.1）")
    args = parser.parse_args()

    records = load_raw()
    existing = load_odds()
    races = build_race_list(records, args.date)

    print(f"対象レース: {len(races)}件  既存オッズ: {len(existing)}件  overwrite={args.overwrite}")
    if args.date:
        print(f"絞り込み: {args.date}")
    print()

    results = fetch_and_store(races, existing, args.overwrite,
                              workers=args.workers, sleep_sec=args.sleep)
    save_odds(results)

    new_count = len(results) - len(existing)
    print(f"\n完了: {len(results)}件保存 (新規+{new_count}件)")
    print(f"保存先: {DATA_DIR / 'odds_data.json'}")


if __name__ == "__main__":
    main()
