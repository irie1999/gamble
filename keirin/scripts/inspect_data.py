"""
収集データの中身を確認するスクリプト
Usage:
  python keirin/scripts/inspect_data.py [YYYYMMDD]
  例: python keirin/scripts/inspect_data.py 20260206
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

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

    # 1レース分のサンプル表示
    sample_venue = list(venues.keys())[0]
    sample_rno = list(venues[sample_venue].keys())[0]
    sample_riders = venues[sample_venue][sample_rno]
    print(f"\n  【サンプル: {sample_venue} R{sample_rno}】")
    print(f"  {'車番':>3} {'選手名':<10} {'クラス':<4} {'競走得点':>6} {'勝率':>5} {'着順':>4}")
    for r in sorted(sample_riders, key=lambda x: x.get("car_no", 0)):
        print(f"  {r.get('car_no','?'):>3} {r.get('player_name','?'):<10} "
              f"{r.get('class','?'):<4} {r.get('kyosoten') or '-':>6} "
              f"{(r.get('win_rate') or 0):.1%} {r.get('rank') or '?':>4}")


def show_odds_summary(odds_data, target_date):
    day_odds = {rid: d for rid, d in odds_data.items() if rid[2:10] == target_date}
    if not day_odds:
        print(f"\n  オッズデータ: なし")
        return

    print(f"\n  【オッズデータ: {len(day_odds)}レース】")
    sample_rid = list(day_odds.keys())[0]
    sample = day_odds[sample_rid]
    print(f"  サンプル race_id: {sample_rid}")
    print(f"  取得済み賭け式: {list(sample.keys())}")
    for bt, d in sample.items():
        vals = list(d.values())
        if vals:
            print(f"    {bt}: {len(vals)}組合せ  "
                  f"MIN={min(vals):.1f}  MAX={max(vals):.1f}  "
                  f"AVG={sum(vals)/len(vals):.1f}")


def main():
    records = load_raw()
    odds_data = load_odds()

    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        # 最新日
        dates = sorted({r["date"] for r in records})
        target = dates[0]  # 最も古い日（最初に収集）
        print(f"日付未指定: {target} を表示します")

    show_day_summary(records, target)
    show_odds_summary(odds_data, target)

    print(f"\n  総収集件数: {len(records)}件  オッズ保存済み: {len(odds_data)}レース")


if __name__ == "__main__":
    main()
