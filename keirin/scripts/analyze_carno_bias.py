"""
車番バイアス分析スクリプト
使い方: python keirin/scripts/analyze_carno_bias.py
"""
import json, sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DATA_DIR = Path(__file__).parent.parent / "data"

def main():
    raw_path = DATA_DIR / "raw_data.json"
    if not raw_path.exists():
        print("raw_data.json が見つかりません。先に collect を実行してください。")
        return

    with open(raw_path, encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    df["rank"] = pd.to_numeric(df.get("rank", np.nan), errors="coerce")

    n_races = df.groupby(["date", "venue_code", "race_no"]).ngroups
    print(f"総データ数: {len(df):,}件  レース数: {n_races:,}レース")
    print(f"期間: {df['date'].min()} 〜 {df['date'].max()}")

    valid = df[df["rank"].notna()].copy()
    valid["top1"] = (valid["rank"] == 1).astype(int)
    valid["top3"] = (valid["rank"] <= 3).astype(int)

    # ---- 1. 車番別 実績 ----
    print("\n" + "=" * 55)
    print(" 【1】 車番別 実績（全期間）")
    print("=" * 55)
    stats = (
        valid.groupby("car_no")
        .agg(races=("rank", "count"), wins=("top1", "sum"), top3=("top3", "sum"))
        .reset_index()
    )
    stats["win_rate"]  = stats["wins"] / stats["races"]
    stats["top3_rate"] = stats["top3"] / stats["races"]
    stats = stats[stats["races"] > 200].sort_values("car_no")

    # ランダム期待値（6〜9車立て混在）
    avg_top3_expected = valid.groupby(["date","venue_code","race_no"])["top3"].transform("sum").mean() / \
                        valid.groupby(["date","venue_code","race_no"])["rank"].transform("count").mean()

    print(f"\n{'車番':>4} {'出走':>6} {'1着率':>7} {'3着以内率':>9}  {'偏差':>6}")
    print("-" * 48)
    for _, r in stats.iterrows():
        diff = r["top3_rate"] - avg_top3_expected
        diff_str = f"{diff:+.1%}"
        bar = "▓" * int(r["top3_rate"] * 25) + "░" * int((1 - r["top3_rate"]) * 25)
        marker = " ← 内側有利" if r["car_no"] <= 2 else ""
        print(f"  {int(r['car_no']):>2}番  {int(r['races']):>5}  {r['win_rate']:>6.1%}  {r['top3_rate']:>8.1%}  {diff_str:>6}{marker}")

    print(f"\n  期待値（均等）: {avg_top3_expected:.1%}")

    # ---- 2. 特徴量と車番の相関 ----
    print("\n" + "=" * 55)
    print(" 【2】 特徴量と car_no の相関（絶対値降順）")
    print("=" * 55)
    try:
        from features import build_features, FEATURE_COLS
        df_feat = build_features(df)
        if "car_no" in df_feat.columns:
            numeric_feats = [c for c in FEATURE_COLS if c in df_feat.columns]
            corrs = df_feat[numeric_feats + ["car_no"]].corr()["car_no"].drop("car_no")
            corrs = corrs.abs().sort_values(ascending=False)
            print(f"\n  特徴量               car_no相関（絶対値）")
            print("  " + "-" * 40)
            for feat, val in corrs.head(15).items():
                bar = "█" * int(val * 30)
                warn = " ⚠ 要注意" if val > 0.15 else ""
                print(f"  {feat:<25} {val:.3f}  {bar}{warn}")
    except Exception as e:
        print(f"  特徴量分析スキップ: {e}")

    # ---- 3. car_no を含む予測の的中率への影響 ----
    print("\n" + "=" * 55)
    print(" 【3】 判断基準")
    print("=" * 55)

    # 1番の実績 vs 期待値
    car1 = stats[stats["car_no"] == 1]
    if not car1.empty:
        w1 = float(car1["win_rate"].values[0])
        t1 = float(car1["top3_rate"].values[0])
        diff_top3 = t1 - avg_top3_expected
        print(f"\n  1番の1着率:     {w1:.1%}")
        print(f"  1番の3着以内率: {t1:.1%}  (期待値比 {diff_top3:+.1%})")

        if diff_top3 > 0.05:
            print("\n  → 1番は構造的有利あり（内側コース）")
            print("     ただし過去のcar_no特徴量除去により現在のモデルは")
            print("     選手能力特徴量から間接的に1番を選びやすい。")
            print("\n  ★ 推奨: car_no を特徴量に【追加】し、正確な内側補正を学習させる")
            print("           → モデルが適切な重みで内側有利を反映できる")
        elif diff_top3 < -0.02:
            print("\n  → 1番に構造的有利なし。現在の除去設定が正しい。")
        else:
            print("\n  → 1番の有利は軽微。モデルに任せて問題なし。")

    # ---- 4. 最終推奨 ----
    print("\n" + "=" * 55)
    print(" 【4】 最終推奨アクション")
    print("=" * 55)
    print("""
  A) car_no を特徴量に戻す
     → 内側有利を正確に学習。ただし過去の[1,2,3]偏りが再発するリスク。
     → 対策: car_no を「単独特徴量」ではなく「バイアス補正係数」として使う

  B) car_no を除外したまま（現状維持）
     → 内側有利を直接学習しないが、win_rateなどで間接的に拾う
     → 1番過大評価は残るが、バックテストROI +14.3% は維持

  C) bank_inner_bonus のみ復活させる（推奨）
     → 車番ではなくバンク形状による物理的有利を学習
     → 特定の車番バイアスなく、コース特性を反映できる
    """)


if __name__ == "__main__":
    main()
