"""
keirin.jp から過去レースの払戻金を一括取得してodds_data.jsonに保存する

Usage:
  python keirin/scripts/collect_payouts.py              # raw_data.jsonにある全レース
  python keirin/scripts/collect_payouts.py --date 20260504   # 特定日のみ
  python keirin/scripts/collect_payouts.py --overwrite       # 既存データも再取得

出力: keirin/data/odds_data.json
  {race_id: {bet_type: {sel_tuple_str: odds}}}
  ※ sel_tupleはJSONのため文字列キー "1,2" 形式で保存
"""

import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from keirin.scripts.keirin_jp import (
    fetch_race_page, get_payout_odds, VENUE_CODE_TO_KCD,
)

DATA_DIR = Path(__file__).parent.parent / "data"


def tuple_to_key(t: tuple) -> str:
    return ",".join(str(x) for x in t)


def key_to_tuple(s: str) -> tuple:
    return tuple(int(x) for x in s.split(","))


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


def build_race_list(records: list[dict], target_date: str | None) -> list[tuple]:
    """(date, venue_code, venue_name, race_no) の重複なしリストを返す"""
    seen = set()
    races = []
    for r in records:
        if target_date and r["date"] != target_date:
            continue
        key = (r["date"], r["venue_code"], r["race_no"])
        if key not in seen:
            seen.add(key)
            races.append((r["date"], r["venue_code"], r.get("venue_name", ""), r["race_no"]))
    return sorted(races)


def fetch_and_store(races: list[tuple], existing: dict, overwrite: bool) -> dict:
    """払戻金を取得してodds_dataに追記する"""
    results = dict(existing)
    total = len(races)
    ok = skip = fail = 0

    for i, (date, venue_code, venue_name, race_no) in enumerate(races, 1):
        race_id = f"{venue_code}{date}{race_no:02d}0001"  # scraper形式に合わせた16桁キー
        prefix = f"[{i}/{total}] {venue_name or venue_code} {date} R{race_no}"

        if not overwrite and race_id in results:
            skip += 1
            continue

        kcd = VENUE_CODE_TO_KCD.get(str(venue_code))
        if kcd is None:
            print(f"  {prefix} → KCD不明 スキップ")
            fail += 1
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

            # tuple キーを文字列に変換してJSON保存可能な形に
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

        # 50件ごとに中間保存
        if i % 50 == 0:
            save_odds(results)
            print(f"  -- 中間保存 ({i}/{total}) ok={ok} skip={skip} fail={fail} --")

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
