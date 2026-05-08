"""
keirin.jp から過去レースの払戻金を一括取得してodds_data.jsonに保存する

高速バッチ版: 1日1会場単位で取得（個別レースではなく）

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
import threading
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from keirin.scripts.keirin_jp import (
    fetch_race_page, get_payout_odds, fetch_day_payouts,
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


def _group_by_day_venue(races: list[dict]) -> dict[tuple, list[dict]]:
    """(kcd, date) でグループ化"""
    groups: dict[tuple, list[dict]] = {}
    for r in races:
        key = (r["kcd"], r["date"])
        groups.setdefault(key, []).append(r)
    return groups


def _serialize_payouts(payouts: dict) -> dict:
    return {bt: {tuple_to_key(k): v for k, v in d.items()} for bt, d in payouts.items()}


def _fetch_day_group(
    kcd: str, date: str, races_in_day: list[dict], existing: dict, overwrite: bool
) -> tuple[dict, int, int, int]:
    """
    1日1会場分の払戻を取得。
    戻り値: (race_id -> serialized_payouts の dict, ok, skip, fail)
    """
    # 全レース取得済みかチェック
    if not overwrite:
        all_done = all(
            r["race_id"] in existing and "trifecta" in existing[r["race_id"]]
            for r in races_in_day
        )
        if all_done:
            return {}, 0, len(races_in_day), 0

    venue = races_in_day[0]["venue_name"]
    race_nos = {r["race_no"]: r["race_id"] for r in races_in_day}

    results = {}
    ok = skip = fail = 0

    # 1日1会場の全払戻を一括取得
    try:
        day_payouts = fetch_day_payouts(int(kcd), date)
    except Exception as e:
        print(f"  {venue} {date} → 一括取得エラー: {e}")
        day_payouts = {}

    for race_no, race_id in race_nos.items():
        if not overwrite and race_id in existing and "trifecta" in existing[race_id]:
            skip += 1
            continue

        pay = day_payouts.get(race_no, {})
        if pay:
            results[race_id] = _serialize_payouts(pay)
            ok += 1
        else:
            fail += 1

    return results, ok, skip, fail


def fetch_and_store(
    races: list[dict],
    existing: dict,
    overwrite: bool,
    workers: int = 8,
    sleep_sec: float = 0.0,  # バッチ版では不要
) -> dict:
    """払戻金を1日1会場単位で並列取得してodds_dataに追記する"""
    results = dict(existing)
    lock = threading.Lock()

    groups = _group_by_day_venue(races)
    total_groups = len(groups)
    counters = {"ok": 0, "skip": 0, "fail": 0, "done": 0}

    print(f"  日×会場グループ数: {total_groups}  (個別レース数: {len(races)})  並列数: {workers}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(_fetch_day_group, kcd, date, day_races, existing, overwrite): (kcd, date, day_races)
            for (kcd, date), day_races in groups.items()
        }
        for fut in as_completed(futs):
            kcd, date, day_races = futs[fut]
            venue = day_races[0]["venue_name"]
            try:
                day_results, ok, skip, fail = fut.result()
            except Exception as e:
                print(f"  {venue} {date} → エラー: {e}")
                ok, skip, fail = 0, 0, len(day_races)
                day_results = {}

            with lock:
                results.update(day_results)
                counters["ok"] += ok
                counters["skip"] += skip
                counters["fail"] += fail
                counters["done"] += 1
                done = counters["done"]

                if ok > 0:
                    print(f"  [{done}/{total_groups}] {venue} {date} → OK {ok}レース  skip={skip} fail={fail}")
                elif skip == len(day_races):
                    pass  # 全スキップは表示しない
                else:
                    print(f"  [{done}/{total_groups}] {venue} {date} → fail={fail} skip={skip}")

                if done % 50 == 0:
                    save_odds(results)
                    print(f"  -- 中間保存 ({done}/{total_groups}グループ) ok={counters['ok']} skip={counters['skip']} fail={counters['fail']} --")

    print(f"\nok={counters['ok']} skip={counters['skip']} fail={counters['fail']}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="特定日のみ取得 (YYYYMMDD)")
    parser.add_argument("--overwrite", action="store_true", help="既存データも再取得")
    parser.add_argument("--workers", type=int, default=8, help="並列グループ数（デフォルト: 8）")
    args = parser.parse_args()

    records = load_raw()
    existing = load_odds()
    races = build_race_list(records, args.date)

    print(f"対象レース: {len(races)}件  既存オッズ: {len(existing)}件  overwrite={args.overwrite}")
    if args.date:
        print(f"絞り込み: {args.date}")
    print()

    results = fetch_and_store(races, existing, args.overwrite, workers=args.workers)
    save_odds(results)

    new_count = len(results) - len(existing)
    print(f"\n完了: {len(results)}件保存 (新規+{new_count}件)")
    print(f"保存先: {DATA_DIR / 'odds_data.json'}")


if __name__ == "__main__":
    main()
