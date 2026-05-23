"""スマホ通知（プッシュ）の汎用ディスパッチャ。

複数のバックエンドに対応し、環境変数で設定する。設定されたバックエンド全てに送る。

対応:
  - LINE Messaging API
      LINE_ACCESS_TOKEN  : チャンネルアクセストークン（長期）
      LINE_USER_ID       : 通知先の userId（自分の LINE userId）

  - ntfy.sh （アカウント不要・最も簡単）
      NTFY_TOPIC : 任意の文字列（例: kyotei-signal-abc123）
                   スマホで ntfy アプリを入れ、同じトピックを subscribe するだけ

  - Discord Webhook
      DISCORD_WEBHOOK_URL : サーバーで生成した Webhook URL

  - Slack Webhook
      SLACK_WEBHOOK_URL  : ワークスペースの Incoming Webhook URL
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _post_json(url: str, payload: dict, headers: Optional[dict] = None,
               timeout: float = 10.0) -> tuple[bool, str]:
    """POST helper. (ok, error_or_status)"""
    data = json.dumps(payload).encode("utf-8")
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return (200 <= res.status < 300, f"HTTP {res.status}")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _post_text(url: str, text: str, headers: Optional[dict] = None,
               timeout: float = 10.0) -> tuple[bool, str]:
    data = text.encode("utf-8")
    h = headers or {}
    req = urllib.request.Request(url, data=data, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return (200 <= res.status < 300, f"HTTP {res.status}")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def push_line(title: str, message: str) -> Optional[bool]:
    """LINE Messaging API でプッシュ。設定が無ければ None を返す。"""
    token = os.environ.get("LINE_ACCESS_TOKEN")
    user_id = os.environ.get("LINE_USER_ID")
    if not (token and user_id):
        return None
    payload = {
        "to": user_id,
        "messages": [{"type": "text", "text": f"【{title}】\n{message}"}],
    }
    ok, info = _post_json(
        "https://api.line.me/v2/bot/message/push",
        payload,
        headers={"Authorization": f"Bearer {token}"},
    )
    if not ok:
        logger.warning("LINE 送信失敗: %s", info)
    return ok


def push_ntfy(title: str, message: str) -> Optional[bool]:
    """ntfy.sh にプッシュ。設定が無ければ None を返す。"""
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        return None
    server = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    url = f"{server}/{topic}"
    ok, info = _post_text(
        url,
        message,
        headers={"Title": title.encode("utf-8").decode("latin-1", errors="ignore"),
                 "Priority": "high",
                 "Tags": "boat,bell"},
    )
    if not ok:
        logger.warning("ntfy 送信失敗: %s", info)
    return ok


def push_discord(title: str, message: str) -> Optional[bool]:
    """Discord Webhook にプッシュ。設定が無ければ None を返す。"""
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        return None
    payload = {
        "content": None,
        "embeds": [{
            "title": title,
            "description": message,
            "color": 0x60a5fa,
        }],
    }
    ok, info = _post_json(url, payload)
    if not ok:
        logger.warning("Discord 送信失敗: %s", info)
    return ok


def push_slack(title: str, message: str) -> Optional[bool]:
    """Slack Incoming Webhook にプッシュ。設定が無ければ None を返す。"""
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return None
    payload = {"text": f"*{title}*\n{message}"}
    ok, info = _post_json(url, payload)
    if not ok:
        logger.warning("Slack 送信失敗: %s", info)
    return ok


def push_all(title: str, message: str) -> dict[str, Optional[bool]]:
    """設定されている全バックエンドに送信。結果を {backend: ok or None} で返す。"""
    results = {
        "line": push_line(title, message),
        "ntfy": push_ntfy(title, message),
        "discord": push_discord(title, message),
        "slack": push_slack(title, message),
    }
    enabled = {k: v for k, v in results.items() if v is not None}
    if not enabled:
        logger.debug("プッシュ通知バックエンドが1つも設定されていません")
    else:
        ok_list = [k for k, v in enabled.items() if v]
        fail_list = [k for k, v in enabled.items() if not v]
        logger.info("push 送信: 成功=%s 失敗=%s", ok_list, fail_list)
    return results
