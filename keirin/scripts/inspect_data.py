"""
収集データの中身を確認するスクリプト
Usage:
  python keirin/scripts/inspect_data.py [YYYYMMDD]
  例: python keirin/scripts/inspect_data.py 20260206
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

DATA_DIR = Path(__file__).parent.parent / "data"


def load_raw(filename="raw_data.json"):
    path = DATA_DIR / filename
    if not path.exists():
        print(f"[エラー] {path} が見つかりません")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_odds(filename="odds_data.json"):
    path = DATA_DIR / filename
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def show_day_summary(records, target_date):
    day = [r for r in records if r.get("date") == target_date]
    if not day:
        available = sorted({r["date"] for r in records})
        print(f"日付 {target_date} のデータがありません")
        print(f"利用可能な日付: {available[0]} 〜 {available[-1]} ({len(available)}日)")
        return

    # 会場・レース別に集計
    venues = defaultdict(lambda: defaultdict(list))
    for r in day:
        venues[r["venue_name"]][r["race_no"]].append(r)

    total_races = sum(len(rs) for rs in venues.values())
    total_riders = len(day)
    print(f"\n{'='*60}")
    print(f"  {target_date} の収集データ")
    print(f"{'='*60}")
    print(f"  レース数: {total_races}  選手数: {total_riders}  会場数: {len(venues)}")
    print(f"  平均選手/レース: {total_riders/total_races:.1f}")

    # 選手数分布
    race_sizes = [len(riders) for races in venues.values() for riders in races.values()]
    size_dist = Counter(race_sizes)
    print(f"\n  【レースあたり選手数の分布】")
    for sz in sorted(size_dist):
        bar = "█" * size_dist[sz]
        print(f"    {sz}人: {size_dist[sz]:>3}レース  {bar}")

    # 欠損値チェック
    key_cols = ["car_no", "player_name", "class", "kyosoten", "win_rate", "rank", "win"]
    print(f"\n  【フィールド欠損チェック】")
    for col in key_cols:
        missing = sum(1 for r in day if r.get(col) is None)
        print(f"    {col:<15}: {missing:>3}件欠損 ({missing/total_riders:.0%})")

    # 会場別サマリー
    print(f"\n  【会場別】")
    print(f"  {'会場':<8} {'レース数':>6} {'選手/R':>6}")
    for venue, races in sorted(venues.items()):
        nr = len(races)
        ns = sum(len(v) for v in races.values())
        print(f"  {venue:<8} {nr:>6} {ns/nr:>6.1f}")

    # 1レース分のサンプル表示（9人レースがあればそれを優先）
    best_venue, best_rno, best_riders = None, None, []
    for venue, races in venues.items():
        for rno, riders in races.items():
            if len(riders) > len(best_riders):
                best_venue, best_rno, best_riders = venue, rno, riders

    print(f"\n  【サンプル: {best_venue} R{best_rno} ({len(best_riders)}人)】")
    print(f"  {'車番':>3} {'選手名':<14} {'クラス':<4} {'競走得点':>6} {'勝率':>5} {'着順':>4}")
    for r in sorted(best_riders, key=lambda x: x.get("car_no", 0)):
        print(f"  {r.get('car_no','?'):>3} {r.get('player_name','?'):<14} "
              f"{r.get('class','?'):<4} {r.get('kyosoten') or '-':>6} "
              f"{(r.get('win_rate') or 0):.1%} {r.get('rank') or '?':>4}")


def show_odds_summary(odds_data, target_date):
    day_odds = {rid: d for rid, d in odds_data.items() if rid[2:10] == target_date}
    if not day_odds:
        print(f"\n  オッズデータ: なし")
        return

    print(f"\n  【オッズデータ: {len(day_odds)}レース】")

    # 全レースの単勝・複勝平均を集計して比較
    win_avgs, place_avgs = [], []
    for d in day_odds.values():
        if "win" in d:
            vals = list(d["win"].values())
            if vals:
                win_avgs.append(sum(vals) / len(vals))
        if "place" in d:
            vals = list(d["place"].values())
            if vals:
                place_avgs.append(sum(vals) / len(vals))

    if win_avgs and place_avgs:
        wa = sum(win_avgs) / len(win_avgs)
        pa = sum(place_avgs) / len(place_avgs)
        status = "✓ 正常" if pa < wa else "✗ 異常（複勝 >= 単勝）"
        print(f"  単勝平均オッズ: {wa:.2f}  複勝平均オッズ: {pa:.2f}  {status}")

    # 1レース詳細サンプル
    sample_rid = list(day_odds.keys())[0]
    sample = day_odds[sample_rid]
    print(f"\n  サンプル race_id: {sample_rid}")
    for bt, d in sample.items():
        vals = list(d.values())
        if vals:
            print(f"    {bt:<10}: {len(vals):>3}組合せ  "
                  f"MIN={min(vals):.1f}  MAX={max(vals):.1f}  AVG={sum(vals)/len(vals):.1f}")


def main():
    records = load_raw()
    odds_data = load_odds()

    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        dates = sorted({r["date"] for r in records}, reverse=True)
        target = dates[0]  # 最新日
        print(f"日付未指定: {target} を表示します")

    show_day_summary(records, target)
    show_odds_summary(odds_data, target)

    print(f"\n  総収集件数: {len(records)}件  オッズ保存済み: {len(odds_data)}レース")


if __name__ == "__main__":
    main()
