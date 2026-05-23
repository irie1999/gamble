"""run_signal を定期実行し、新規シグナル発生時に通知する監視スクリプト。

オッズは boatrace.jp で順次公開（締切1時間前頃）されるため、朝1回の実行では
拾えないレースが多い。本スクリプトは指定間隔で run_signal を再実行し、新しい
race_id がベット候補に追加された時点で:

  1. ターミナルに 🔔 通知
  2. Windows ビープ音
  3. Windows トースト通知（PowerShell 経由）
  4. ブラウザで最新の HTML レポートを自動オープン

使い方:
    # デフォルト: 30分ごとに v2 を実行
    python -m scripts.watch_signal

    # 15分ごと・通知音なし
    python -m scripts.watch_signal --interval 15 --no-beep

    # 旧設定 (run_signal) を使う
    python -m scripts.watch_signal --variant v1

    # 通知だけ欲しい（ブラウザは開かない）
    python -m scripts.watch_signal --no-open-browser
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
import webbrowser
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.utils.config import MODELS_DIR, PROCESSED_DIR

BETS_CSV = MODELS_DIR / "backtest" / "bets_lane1_kelly.csv"


def _current_bet_set() -> set[str]:
    """bets_lane1_kelly.csv に現在含まれている race_id 集合。"""
    if not BETS_CSV.exists():
        return set()
    try:
        df = pd.read_csv(BETS_CSV, usecols=["race_id"])
    except Exception:
        return set()
    return set(df["race_id"].astype(str).unique())


def _beep() -> None:
    """Windows: システムビープ。それ以外: ベル文字。"""
    try:
        if platform.system() == "Windows":
            import winsound
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass


def _show_toast(title: str, msg: str) -> None:
    """Windows: PowerShell経由でトースト通知（追加依存ライブラリなし）。"""
    if platform.system() != "Windows":
        return
    # PowerShell スクリプトで Windows.UI.Notifications を呼ぶ
    # シングルクォート2回でエスケープ
    safe_title = title.replace("'", "''")
    safe_msg = msg.replace("'", "''")
    ps_script = (
        "$ErrorActionPreference='SilentlyContinue';"
        "[Windows.UI.Notifications.ToastNotificationManager,Windows.UI.Notifications,ContentType=WindowsRuntime]|Out-Null;"
        "[Windows.Data.Xml.Dom.XmlDocument,Windows.Data.Xml.Dom.XmlDocument,ContentType=WindowsRuntime]|Out-Null;"
        "$tpl=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent('ToastText02');"
        "$xml=$tpl.GetXml();"
        f"$xml=$xml -replace '<text id=\"1\">.*?</text>','<text id=\"1\">{safe_title}</text>';"
        f"$xml=$xml -replace '<text id=\"2\">.*?</text>','<text id=\"2\">{safe_msg}</text>';"
        "$doc=New-Object Windows.Data.Xml.Dom.XmlDocument;$doc.LoadXml($xml);"
        "$toast=New-Object Windows.UI.Notifications.ToastNotification $doc;"
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Kyotei').Show($toast);"
    )
    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps_script],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _open_browser(target: date) -> None:
    html = PROCESSED_DIR / f"signals_{target.strftime('%Y%m%d')}.html"
    if html.exists():
        try:
            webbrowser.open(html.resolve().as_uri())
        except Exception:
            pass


def _run_signal_once(variant: str, extra_args: list[str]) -> int:
    """run_signal を1回実行（ブラウザ自動オープンは抑止）。終了コードを返す。"""
    module = "scripts.run_signal_v2" if variant == "v2" else "scripts.run_signal"
    cmd = [sys.executable, "-m", module, "--autofill", "--no-open"] + extra_args
    return subprocess.call(cmd)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interval", type=int, default=30,
                   help="再実行間隔（分）。デフォルト 30")
    p.add_argument("--variant", choices=["v1", "v2"], default="v2",
                   help="run_signal の系統。v1=旧設定 / v2=B設定（デフォルト）")
    p.add_argument("--no-beep", action="store_true")
    p.add_argument("--no-toast", action="store_true")
    p.add_argument("--no-open-browser", action="store_true")
    p.add_argument("--refresh-odds", action="store_true",
                   help="毎回オッズを強制再取得（v1の場合のみ有効）")
    args = p.parse_args()

    target = date.today()
    extra = ["--refresh-odds"] if args.refresh_odds and args.variant == "v1" else []

    print("=" * 60)
    print(f"シグナル監視開始: {args.variant} を {args.interval}分ごとに実行")
    print(f"対象日: {target}")
    print(f"Ctrl+C で停止")
    print("=" * 60)

    prev_ids: set[str] = set()
    first_run = True
    iteration = 0

    try:
        while True:
            iteration += 1
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{now}] === 実行 #{iteration} ===")

            rc = _run_signal_once(args.variant, extra)
            cur_ids = _current_bet_set()
            new_ids = cur_ids - prev_ids
            removed = prev_ids - cur_ids

            if first_run:
                print(f"[{now}] 初回: 候補 {len(cur_ids)}件")
                if cur_ids and not args.no_open_browser:
                    _open_browser(target)
            elif new_ids:
                msg = f"新規 {len(new_ids)}件 / 計 {len(cur_ids)}件: {', '.join(sorted(new_ids))}"
                print(f"[{now}] 🔔 {msg}")
                if not args.no_beep:
                    _beep()
                if not args.no_toast:
                    short = f"{len(new_ids)}件: " + ", ".join(
                        rid.rsplit("-", 1)[0].split("-", 1)[1] + "-" + rid.rsplit("-", 1)[1]
                        for rid in sorted(new_ids)[:3]
                    )
                    _show_toast(f"競艇シグナル新規 {len(new_ids)}件", short)
                if not args.no_open_browser:
                    _open_browser(target)
            elif removed:
                print(f"[{now}] 候補 {len(cur_ids)}件（{len(removed)}件が候補外に）")
            else:
                print(f"[{now}] 候補 {len(cur_ids)}件（変化なし）")

            prev_ids = cur_ids
            first_run = False

            next_t = datetime.now().replace(microsecond=0)
            wait_sec = args.interval * 60
            print(f"[{now}] 次回実行まで {args.interval}分待機...")
            time.sleep(wait_sec)
    except KeyboardInterrupt:
        print("\n\n=== 監視終了 ===")


if __name__ == "__main__":
    main()
