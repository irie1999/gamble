"""共通ロガー"""
from __future__ import annotations

import logging
import sys


def _ensure_utf8_console() -> None:
    """Windows: stdout/stderr を UTF-8 に再構成。

    日本語 Windows のコンソールはデフォルトで cp932（Shift-JIS）。print や
    logger に '✓' や '🔔'・'⚠'・'🟢' といった非 CP932 文字を含む文字列を渡すと
    UnicodeEncodeError で落ちる。errors='replace' で未対応文字は '?' に置換され、
    クラッシュは防ぐ。watch_signal の subprocess 経由でも、子側の logger 読み込み
    時に同じ処理が走るため、CSV 取得スクリプトでも有効。
    """
    if sys.platform != "win32":
        return
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        cur_enc = (getattr(stream, "encoding", "") or "").lower()
        if cur_enc == "utf-8":
            continue
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_ensure_utf8_console()


def get_logger(name: str = "gamble", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger
