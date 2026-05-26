"""run_signal の B 設定版（4/1〜5/15 バックテスト最適化済み）。

run_signal.py に以下のデフォルトを上書きして呼び出すラッパー:
- --min-odds 3.0       （本命過ぎ 2-3倍帯は -32% ROI で除外）
- --exclude-venues に 10 (三国) を追加（10件 20% 勝率で構造的に弱い）
- --kelly-fraction 0.5（ハーフKelly: 過去365日のバックテストで ROI 最大）
- --odds-shrinkage-max 0.3（時間ベース shrinkage で締切まで遠い場合は保守ステーク）

その他の引数（--date, --autofill, --kelly-fraction 等）はそのまま渡せる。
ユーザが --min-odds や --exclude-venues を明示すれば、後勝ちで上書きできる。

使い方:
    # 今日のシグナル（B 設定）
    python -m scripts.run_signal_v2 --autofill

    # 過去日
    python -m scripts.run_signal_v2 --date 2026-05-11

    # 上書き例: shrinkage を無効化
    python -m scripts.run_signal_v2 --autofill --odds-shrinkage-max 0

バックテスト実績（2025-05-24〜2026-05-23, 49件確定）:
    KF 0.25: ROI +156%, PnL +¥129K, 勝率 55.3%, 最大DD -4.7%
    KF 0.50: ROI +166%, PnL +¥239K, 勝率 55.3%, 最大DD -7.3%  ← 採用
"""
from __future__ import annotations

import subprocess
import sys

# B 設定のオーバーライド引数
B_OVERRIDES = [
    "--min-odds", "3.0",
    "--exclude-venues", "04", "03", "02", "14", "01", "24", "10",
    "--kelly-fraction", "0.5",
    "--odds-shrinkage-max", "0.3",
]


def main() -> None:
    # ユーザ引数を後ろに置くことで、ユーザが同じフラグを再指定すれば後勝ち。
    user_args = sys.argv[1:]
    cmd = [sys.executable, "-m", "scripts.run_signal"] + B_OVERRIDES + user_args
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
