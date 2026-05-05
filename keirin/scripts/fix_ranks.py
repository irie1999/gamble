"""
odds_data.json の3連単データを使って raw_data.json の着順（rank）を修正する。

3連単キー "4_1_6" → 1着=4, 2着=1, 3着=6 として rank を上書き。
4位以下は判明しないため、残り車番に rank=4,5,... を連番で付与する。
win フィールドも再計算する。

Usage:
  python keirin/scripts/fix_ranks.py [--dry-run]
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"


def load_json(path: Path) -> object:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_race_id(venue_code: str, date: str, race_no: int) -> str:
    """scraper 形式の race_id: {venue_code:02d}{date}{race_no:02d}0001"""
    return f"{int(venue_code):02d}{date}{int(race_no):02d}0001"


def extract_top3_from_trifecta(odds_entry: dict) -> tuple[int, int, int] | None:
    """
    odds_entry: {bet_type: {key_str: odds}}
    3連単データから最低オッズ（実際に払い戻しがあった組み合わせ）を探す。
    払い戻し金額が最も低い組み合わせ = 1点のみ存在するはず。
    Returns: (1着車番, 2着車番, 3着車番) or None
    """
    trifecta = odds_entry.get("trifecta", {})
    if not trifecta:
        return None

    # 値が存在するキーをすべて取得（通常は1つのみ）
    entries = [(k, v) for k, v in trifecta.items() if v is not None and v > 0]
    if not entries:
        return None

    # 最小オッズ（払戻金が実際についた組み合わせ）を選ぶ
    best_key = min(entries, key=lambda x: x[1])[0]
    sep = "_" if "_" in best_key else ","
    parts = best_key.split(sep)
    if len(parts) != 3:
        return None
    return tuple(int(p) for p in parts)


def fix_ranks(raw_data: list[dict], odds_data: dict) -> tuple[list[dict], dict]:
    """
    raw_data の rank を odds_data の3連単から修正する。
    Returns: (修正後のレコードリスト, 統計dict)
    """
    # レースごとにグループ化
    races: dict[str, list[dict]] = defaultdict(list)
    for r in raw_data:
        rid = build_race_id(r["venue_code"], r["date"], r["race_no"])
        races[rid].append(r)

    stats = {"total_races": len(races), "fixed": 0, "no_odds": 0, "no_trifecta": 0}

    for rid, records in races.items():
        odds_entry = odds_data.get(rid)
        if not odds_entry:
            stats["no_odds"] += 1
            continue

        top3 = extract_top3_from_trifecta(odds_entry)
        if not top3:
            stats["no_trifecta"] += 1
            continue

        first, second, third = top3
        rank_map = {first: 1, second: 2, third: 3}

        # 4位以下の車番を連番で割り当て
        top3_set = {first, second, third}
        remaining = sorted(r["car_no"] for r in records if r["car_no"] not in top3_set)
        for i, car_no in enumerate(remaining, start=4):
            rank_map[car_no] = i

        for r in records:
            r["rank"] = rank_map.get(r["car_no"])
            r["win"] = 1 if r["rank"] == 1 else 0

        stats["fixed"] += 1

    return raw_data, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="保存せずに統計だけ表示")
    args = parser.parse_args()

    raw_path = DATA_DIR / "raw_data.json"
    odds_path = DATA_DIR / "odds_data.json"

    if not raw_path.exists():
        print(f"エラー: {raw_path} が見つかりません")
        sys.exit(1)
    if not odds_path.exists():
        print(f"エラー: {odds_path} が見つかりません")
        sys.exit(1)

    print("読み込み中...")
    raw_data = load_json(raw_path)
    odds_raw = load_json(odds_path)

    # odds_data は文字列キーのまま使用（extract_top3_from_trifecta が対応）
    print(f"raw_data: {len(raw_data)}件  odds_data: {len(odds_raw)}レース")

    # 修正前の統計
    ranked = [r for r in raw_data if r.get("rank") is not None]
    eq_before = sum(1 for r in ranked if r["car_no"] == r["rank"])
    print(f"\n修正前: rank=car_no {eq_before}/{len(ranked)} ({100*eq_before/max(1,len(ranked)):.1f}%)")

    fixed_data, stats = fix_ranks(raw_data, odds_raw)

    print(f"\n修正結果:")
    print(f"  対象レース数: {stats['total_races']}")
    print(f"  修正完了:     {stats['fixed']}")
    print(f"  odds未収録:   {stats['no_odds']}")
    print(f"  3連単なし:    {stats['no_trifecta']}")

    # 修正後の統計
    ranked2 = [r for r in fixed_data if r.get("rank") is not None]
    eq_after = sum(1 for r in ranked2 if r["car_no"] == r["rank"])
    print(f"\n修正後: rank=car_no {eq_after}/{len(ranked2)} ({100*eq_after/max(1,len(ranked2)):.1f}%)")
    wins = sum(1 for r in ranked2 if r.get("win") == 1)
    print(f"win=1 のレコード数: {wins} (期待値 {len(ranked2)//6}前後)")

    if args.dry_run:
        print("\n--dry-run 指定のため保存しません")
    else:
        # バックアップ
        backup_path = DATA_DIR / "raw_data_backup.json"
        if not backup_path.exists():
            save_json(backup_path, load_json(raw_path))
            print(f"\nバックアップ保存: {backup_path}")
        save_json(raw_path, fixed_data)
        print(f"保存完了: {raw_path}")


if __name__ == "__main__":
    main()
