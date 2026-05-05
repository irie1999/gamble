"""
keirin.jp から過去レースの払戻金を一括取得してodds_data.jsonに保存する

Usage:
  python keirin/scripts/collect_payouts.py              # raw_data.jsonにある全レース
  python keirin/scripts/collect_payouts.py --date 20260504   # 特定日のみ
  python keirin/scripts/collect_payouts.py --overwrite       # 既存データも再取得

出力: keirin/data/odds_data.json
  {race_id: {bet_type: {sel_tuple_str: odds}}}
"""

import sys
import json
import time
import argparse
from pathlib import Path

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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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


def fetch_and_store(races: list[dict], existing: dict, overwrite: bool) -> dict:
    """払戻金を取得してodds_dataに追記する"""
    results = dict(existing)
    total = len(races)
    ok = skip = fail = 0

    for i, race in enumerate(races, 1):
        race_id = race["race_id"]
        kcd = race["kcd"]
        date = race["date"]
        race_no = race["race_no"]
        venue_name = race["venue_name"]
        prefix = f"[{i}/{total}] {venue_name} {date} R{race_no:02d}"

        if not overwrite and race_id in results and "trifecta" in results[race_id]:
            skip += 1
            continue

        try:
            jdata = fetch_race_page(kcd, date, race_no)
            if not jdata:
                print(f"  {prefix} → データなし")
                fail += 1
                time.sleep(0.5)
                continue

            payouts = get_payout_odds(jdata)
            if not payouts:
                print(f"  {prefix} → 払戻なし（未完了または非公開）")
                fail += 1
                time.sleep(0.5)
                continue

            serializable = {}
            for bt, d in payouts.items():
                serializable[bt] = {tuple_to_key(k): v for k, v in d.items()}

            results[race_id] = serializable
            bet_summary = " ".join(f"{bt}:{len(d)}" for bt, d in payouts.items())
            print(f"  {prefix} → OK  {bet_summary}")
            ok += 1

        except Exception as e:
            print(f"  {prefix} → エラー: {e}")
            fail += 1

        time.sleep(0.4)

        if i % 50 == 0:
            save_odds(results)
            print(f"  -- 中間保存 ({i}/{total}) ok={ok} skip={skip} fail={fail} --")

    print(f"\nok={ok} skip={skip} fail={fail}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="特定日のみ取得 (YYYYMMDD)")
    parser.add_argument("--overwrite", action="store_true", help="既存データも再取得")
    args = parser.parse_args()

    records = load_raw()
    existing = load_odds()
    races = build_race_list(records, args.date)

    print(f"対象レース: {len(races)}件  既存オッズ: {len(existing)}件  overwrite={args.overwrite}")
    if args.date:
        print(f"絞り込み: {args.date}")
    print()

    results = fetch_and_store(races, existing, args.overwrite)
    save_odds(results)

    new_count = len(results) - len(existing)
    print(f"\n完了: {len(results)}件保存 (新規+{new_count}件)")
    print(f"保存先: {DATA_DIR / 'odds_data.json'}")


if __name__ == "__main__":
    main()
