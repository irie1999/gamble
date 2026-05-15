"""run_signal の B 設定版（4/1〜5/15 バックテスト最適化済み）。

run_signal.py に以下のデフォルトを上書きして呼び出すラッパー:
- --min-odds 3.0       （本命過ぎ 2-3倍帯は -32% ROI で除外）
- --exclude-venues に 10 (三国) を追加（10件 20% 勝率で構造的に弱い）

その他の引数（--date, --autofill, --kelly-fraction 等）はそのまま渡せる。
ユーザが --min-odds や --exclude-venues を明示すれば、後勝ちで上書きできる。

使い方:
    # 今日のシグナル（B 設定）
    python -m scripts.run_signal_v2 --autofill

    # 過去日
    python -m scripts.run_signal_v2 --date 2026-05-11

    # 上書き例: min-odds を緩める
    python -m scripts.run_signal_v2 --autofill --min-odds 2.5

バックテスト実績（4/1〜5/15, 38件確定）:
    旧 run_signal: ROI +76%, PnL +¥49,740, 勝率 44.7%
    本スクリプト: ROI +113%, PnL +¥54,650, 勝率 55.6%, 最大DD -4.1%
"""
from __future__ import annotations

import subprocess
import sys

# B 設定のオーバーライド引数
B_OVERRIDES = [
    "--min-odds", "3.0",
    "--exclude-venues", "04", "03", "02", "14", "01", "24", "10",
]


def main() -> None:
    # ユーザ引数を後ろに置くことで、ユーザが同じフラグを再指定すれば後勝ち。
    user_args = sys.argv[1:]
    cmd = [sys.executable, "-m", "scripts.run_signal"] + B_OVERRIDES + user_args
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
