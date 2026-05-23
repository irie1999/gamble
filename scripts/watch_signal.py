"""run_signal を定期実行し、新規シグナル発生時に通知する監視スクリプト。

オッズは boatrace.jp で順次公開（締切1時間前頃）されるため、朝1回の実行では
拾えないレースが多い。本スクリプトは指定間隔で run_signal を再実行し、新しい
race_id がベット候補に追加された時点で:

  1. ターミナルに 🔔 通知
  2. Windows ビープ音
  3. Windows トースト通知（PowerShell 経由）
  4. ブラウザで最新の HTML レポートを自動オープン
  5. スマホへのプッシュ通知（環境変数で LINE/ntfy/Discord/Slack を設定）

スマホ通知のセットアップ例（PowerShell）:
    # ntfy.sh（最も簡単、アカウント不要）:
    $env:NTFY_TOPIC = "kyotei-signal-XXXXXX"  # 推測されにくい文字列

    # LINE Messaging API:
    $env:LINE_ACCESS_TOKEN = "xxxxx"   # チャンネルアクセストークン
    $env:LINE_USER_ID      = "Uxxxxx"  # 自分の userId

設定後に watch_signal を起動すれば、新規シグナル時に自動でスマホに届く。

使い方:
    # デフォルト: 10分ごとに v2 を実行（毎回オッズも強制再取得）
    python -m scripts.watch_signal

    # 15分ごと・通知音なし
    python -m scripts.watch_signal --interval 15 --no-beep

    # 旧設定 (run_signal) を使う
    python -m scripts.watch_signal --variant v1

    # 通知だけ欲しい（ブラウザは開かない）
    python -m scripts.watch_signal --no-open-browser

    # オッズ再取得を止める（サーバー負荷を抑えたい時のみ。EV が古いオッズで
    # 過大評価される副作用に注意）
    python -m scripts.watch_signal --no-refresh-odds
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
import webbrowser
from datetime import date, datetime, time as _time, timedelta
from pathlib import Path

import pandas as pd

from src.notify.push import push_all
from src.utils.config import MODELS_DIR, PROCESSED_DIR, RAW_DIR

BETS_CSV = MODELS_DIR / "backtest" / "bets_lane1_kelly.csv"
SCHEDULE_CSV = RAW_DIR / "race_schedule.csv"


def _hydrate_env_from_registry() -> list[str]:
    """Windows: 永続化された User/Machine 環境変数を現プロセスに取り込む。

    Windows の環境変数は親プロセス起動時にコピーされるため、VSCode などを
    起動した「あと」に SetEnvironmentVariable で設定すると、VSCode 内ターミナル
    からは値が見えない（VSCode 全体を真に再起動しないと反映されない）という
    罠が頻発する。レジストリを直接読んで os.environ にマージすれば、PowerShell
    側に反映されていなくても watch_signal は確実に env を取れる。

    取り込んだキー名のリストを返す（既に os.environ にあるものは上書きしない）。
    """
    if platform.system() != "Windows":
        return []
    try:
        import winreg
    except ImportError:
        return []

    keys_to_check = [
        "LINE_ACCESS_TOKEN", "LINE_USER_ID",
        "NTFY_TOPIC", "NTFY_SERVER",
        "DISCORD_WEBHOOK_URL", "SLACK_WEBHOOK_URL",
    ]
    hydrated: list[str] = []
    sources = [
        (winreg.HKEY_CURRENT_USER, r"Environment"),
        (winreg.HKEY_LOCAL_MACHINE,
         r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    ]
    for hive, subkey in sources:
        try:
            with winreg.OpenKey(hive, subkey) as key:
                for name in keys_to_check:
                    if os.environ.get(name):
                        continue
                    try:
                        value, _ = winreg.QueryValueEx(key, name)
                    except FileNotFoundError:
                        continue
                    if value:
                        os.environ[name] = str(value)
                        hydrated.append(name)
        except OSError:
            continue
    return hydrated


def _current_bets() -> tuple[set[str], dict[str, dict]]:
    """現在の bets CSV から (race_id 集合, race_id → 詳細 dict) を返す。

    詳細 dict は odds / ev / stake / p_blend を含む（push 通知の本文用）。
    """
    if not BETS_CSV.exists():
        return set(), {}
    try:
        df = pd.read_csv(BETS_CSV)
    except Exception:
        return set(), {}
    details: dict[str, dict] = {}
    for _, r in df.iterrows():
        rid = str(r["race_id"])
        p_blend = r.get("blended_win_prob")
        if p_blend is None or pd.isna(p_blend):
            p_blend = r.get("pred_win_prob", 0)
        details[rid] = {
            "odds": float(r.get("odds_win", 0) or 0),
            "ev": float(r.get("ev", 0) or 0),
            "stake": int(r.get("stake", 0) or 0),
            "p_blend": float(p_blend or 0),
        }
    return set(details.keys()), details


def _load_schedule() -> dict[str, str]:
    """race_schedule.csv → {race_id: deadline_time}"""
    if not SCHEDULE_CSV.exists():
        return {}
    try:
        df = pd.read_csv(SCHEDULE_CSV)
        return dict(zip(df["race_id"].astype(str), df["deadline_time"].astype(str)))
    except Exception:
        return {}


def _races_near_deadline(ids: set[str], schedule: dict[str, str],
                         already_reminded: set[str],
                         minutes_threshold: int,
                         now: datetime | None = None) -> set[str]:
    """締切まで minutes_threshold 分以内のレースを返す。

    既にリマインド済み・締切が過ぎたもの・schedule に締切時刻が無いものは除外。
    """
    if now is None:
        now = datetime.now()
    today = now.date()
    soon: set[str] = set()
    for rid in ids:
        if rid in already_reminded:
            continue
        deadline_str = schedule.get(rid)
        if not deadline_str or ":" not in deadline_str:
            continue
        try:
            hh, mm = deadline_str.split(":")[:2]
            deadline = datetime.combine(today, _time(int(hh), int(mm)))
        except (ValueError, AttributeError):
            continue
        delta = deadline - now
        if timedelta(0) < delta <= timedelta(minutes=minutes_threshold):
            soon.add(rid)
    return soon


def _format_push_body(ids: set[str], details: dict[str, dict],
                      new_ids: set[str] | None = None,
                      schedule: dict[str, str] | None = None) -> str:
    """シグナル本文を整形。HTMLレポートと同じ項目を含む。

    new_ids に含まれる race_id は先頭に 🆕 マーカーを付ける。EV 降順で並べ、
    複数行構成で 1レース = 2行（識別行 + 数値行）。
    """
    from src.utils.config import VENUE_CODES
    new_ids = new_ids or set()
    schedule = schedule or {}

    # EV 降順
    ordered = sorted(ids, key=lambda r: -float(details.get(r, {}).get("ev", 0)))

    lines: list[str] = []
    for rid in ordered:
        parts = rid.split("-")
        jcd = parts[1] if len(parts) >= 2 else ""
        venue = VENUE_CODES.get(jcd, jcd) if jcd else rid
        rno = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 0
        d = details.get(rid, {})
        deadline = schedule.get(rid, "")
        deadline_str = f" 締切{deadline}" if deadline else ""
        marker = "🆕 " if rid in new_ids else "・"
        head = f"{marker}{venue}({jcd}) {rno}R{deadline_str}"
        body = (f"  P{d.get('p_blend', 0):.3f} / "
                f"オッズ{d.get('odds', 0):.2f} / "
                f"EV{d.get('ev', 0):.2f} / "
                f"推奨¥{d.get('stake', 0):,}")
        lines.append(head)
        lines.append(body)
    return "\n".join(lines)


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


def _run_signal_once(variant: str, extra_args: list[str], *,
                     skip_features: bool = False, quiet: bool = True) -> int:
    """run_signal を1回実行（ブラウザ自動オープンは抑止）。終了コードを返す。

    過去バックテスト集計は定期実行では不要なので常に --no-backtest-summary で抑止。
    手動で集計を見たい時は `python -m scripts.run_signal_v2 --autofill` を別途叩く。

    quiet=True なら subprocess の stdout/stderr を捕捉してエラー時のみ出力する
    （LightGBM warning / scrape_odds の progress 等の大量出力を隠す）。
    """
    module = "scripts.run_signal_v2" if variant == "v2" else "scripts.run_signal"
    cmd = [sys.executable, "-m", module,
           "--autofill", "--no-open", "--no-backtest-summary"]
    if skip_features:
        cmd.append("--skip-features")
    cmd += extra_args
    if not quiet:
        return subprocess.call(cmd)
    res = subprocess.run(cmd, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if res.returncode != 0:
        # 失敗時のみ捕捉した出力の末尾30行だけ吐く（デバッグ用・全文は --verbose で）
        combined = (res.stdout or "") + (res.stderr or "")
        tail = combined.splitlines()[-30:]
        if tail:
            sys.stderr.write(
                f"⚠ subprocess failed (rc={res.returncode}). 末尾30行:\n"
                + "\n".join(tail) + "\n"
            )
        else:
            sys.stderr.write(f"⚠ subprocess failed (rc={res.returncode}). 出力なし\n")
    return res.returncode


def _all_races_done(target: date, buffer_min: int = 30,
                    now: datetime | None = None) -> tuple[bool, str]:
    """その日の全レースが終了したかを race_schedule.csv から判定。

    最終締切 + buffer_min 分を過ぎていれば True。schedule が無い・パース失敗時は
    False（誤って早期終了するのを避ける）。
    """
    schedule_path = SCHEDULE_CSV
    if not schedule_path.exists():
        return False, ""
    try:
        df = pd.read_csv(schedule_path)
    except Exception:
        return False, ""
    today_str = target.strftime("%Y%m%d")
    today_sched = df[df["race_id"].astype(str).str.startswith(today_str)]
    if today_sched.empty:
        return False, ""
    max_deadline_str = sorted(today_sched["deadline_time"].astype(str).tolist())[-1]
    try:
        hh, mm = max_deadline_str.split(":")[:2]
        max_deadline = datetime.combine(target, _time(int(hh), int(mm)))
    except (ValueError, AttributeError):
        return False, ""
    cur_now = now if now is not None else datetime.now()
    cutoff = max_deadline + timedelta(minutes=buffer_min)
    if cur_now > cutoff:
        return True, f"最終締切 {max_deadline_str} + {buffer_min}分 経過"
    return False, ""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interval", type=int, default=10,
                   help="再実行間隔（分）。デフォルト 10（オッズが揃うのは締切30分前頃）")
    p.add_argument("--variant", choices=["v1", "v2"], default="v2",
                   help="run_signal の系統。v1=旧設定 / v2=B設定（デフォルト）")
    p.add_argument("--no-beep", action="store_true")
    p.add_argument("--no-toast", action="store_true")
    p.add_argument("--no-open-browser", action="store_true")
    p.add_argument("--no-push", action="store_true",
                   help="スマホへのプッシュ通知を無効化（環境変数で設定済みの場合のみ動く）")
    p.add_argument("--no-refresh-odds", action="store_true",
                   help="毎回のオッズ強制再取得を無効化（デフォルトは毎回再取得）。"
                        "オッズは締切まで動くため、再取得しないと早期スクレイプの古い値で EV が過大評価される")
    p.add_argument("--rebuild-features-every", type=int, default=6,
                   help="N回に1回だけ features を再生成（毎回再生成すると重いため）。"
                        "デフォルト6=10分間隔なら1時間に1回")
    p.add_argument("--deadline-reminder-min", type=int, default=10,
                   help="締切のN分前に最新オッズで再通知する（既に通知済の候補も対象）。"
                        "0で無効。実際の発火は iteration タイミング次第で N+α 分前になることもある")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="subprocess(run_signal/scrape_odds/backtest) の詳細ログを表示。"
                        "デフォルトは抑止（quiet モード）で、エラー時のみ全文吐く")
    p.add_argument("--no-auto-exit", action="store_true",
                   help="当日の全レース終了後の自動終了を無効化")
    p.add_argument("--exit-buffer-min", type=int, default=30,
                   help="最終締切からこの分数経過したら自動終了（デフォルト30分）")
    args = p.parse_args()

    target = date.today()
    extra = [] if args.no_refresh_odds else ["--refresh-odds"]

    # Windows: 永続化済みの env をレジストリから取り込み（VSCode 内ターミナル等の救済）
    hydrated = _hydrate_env_from_registry()

    print("=" * 60)
    print(f"シグナル監視開始: {args.variant} を {args.interval}分ごとに実行")
    print(f"対象日: {target}")
    if hydrated:
        print(f"環境変数をレジストリから取り込みました: {', '.join(hydrated)}")
    # LINE 通知の起動時診断（環境変数が無いと sleep 中に黙って失敗するため）
    if not args.no_push:
        if os.environ.get("LINE_ACCESS_TOKEN") and os.environ.get("LINE_USER_ID"):
            print(f"LINE通知: 有効 (user={os.environ['LINE_USER_ID'][:6]}…)")
        else:
            print("⚠ LINE通知: 環境変数 LINE_ACCESS_TOKEN / LINE_USER_ID 未設定。"
                  "Windows のユーザー/システム環境変数に LINE_ACCESS_TOKEN と "
                  "LINE_USER_ID を設定してから再実行してください。")
    print(f"Ctrl+C で停止")
    print("=" * 60)

    prev_ids: set[str] = set()
    reminded_ids: set[str] = set()
    first_run = True
    iteration = 0

    try:
        while True:
            iteration += 1
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{now}] === 実行 #{iteration} ===")

            # 初回は features を必ず再生成、以降は --rebuild-features-every 毎
            skip_features = not (iteration == 1 or iteration % args.rebuild_features_every == 0)
            rc = _run_signal_once(args.variant, extra, skip_features=skip_features,
                                  quiet=not args.verbose)
            cur_ids, details = _current_bets()
            schedule = _load_schedule()
            new_ids = cur_ids - prev_ids
            removed = prev_ids - cur_ids

            if first_run:
                print(f"[{now}] 初回: 候補 {len(cur_ids)}件")
                if cur_ids and not args.no_open_browser:
                    _open_browser(target)
                # 初回のシグナルもスマホ通知（全件・新規マーカーなし）
                if cur_ids and not args.no_push:
                    body = _format_push_body(cur_ids, details, schedule=schedule)
                    push_all(f"競艇シグナル {len(cur_ids)}件 (初回)", body)
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
                # スマホへ: 新規が出た時は「現在出ている全シグナル」を送る。
                # new_ids は本文中で 🆕 マーカーで識別できる。
                if not args.no_push:
                    body = _format_push_body(cur_ids, details,
                                             new_ids=new_ids, schedule=schedule)
                    push_all(
                        f"競艇シグナル 新規{len(new_ids)}件 / 計{len(cur_ids)}件",
                        body,
                    )
            elif removed:
                print(f"[{now}] 候補 {len(cur_ids)}件（{len(removed)}件が候補外に）")
            else:
                print(f"[{now}] 候補 {len(cur_ids)}件（変化なし）")

            # 締切リマインド: 候補のうち締切 N分以内のものを1回だけ通知
            if args.deadline_reminder_min > 0:
                # iteration の隙間で取りこぼさないよう半周期ぶんマージンを足す
                effective_threshold = args.deadline_reminder_min + max(args.interval // 2, 2)
                soon = _races_near_deadline(
                    cur_ids, schedule, reminded_ids,
                    minutes_threshold=effective_threshold,
                )
                if soon:
                    title = f"⏰ 締切{args.deadline_reminder_min}分前 {len(soon)}件"
                    print(f"[{now}] {title}: {', '.join(sorted(soon))}")
                    if not args.no_beep:
                        _beep()
                    if not args.no_push:
                        body = _format_push_body(soon, details, schedule=schedule)
                        push_all(title, body)
                    reminded_ids |= soon

            prev_ids = cur_ids
            first_run = False

            # 自動終了: 全レース終了+バッファ経過なら抜ける
            if not args.no_auto_exit:
                done, reason = _all_races_done(target, buffer_min=args.exit_buffer_min)
                if done:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                          f"=== 全レース終了 ({reason}) → watch_signal 自動終了 ===")
                    break

            wait_sec = args.interval * 60
            print(f"[{now}] 次回実行まで {args.interval}分待機...")
            time.sleep(wait_sec)
    except KeyboardInterrupt:
        print("\n\n=== 監視終了 ===")


if __name__ == "__main__":
    main()
