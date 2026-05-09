"""HTTP取得クライアント（リトライ・スロットリング付き）"""
from __future__ import annotations

import time
from typing import Optional

import requests

from src.utils.config import REQUEST_INTERVAL_SEC, REQUEST_TIMEOUT_SEC, USER_AGENT
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HttpClient:
    def __init__(
        self,
        interval_sec: float = REQUEST_INTERVAL_SEC,
        timeout_sec: int = REQUEST_TIMEOUT_SEC,
        user_agent: str = USER_AGENT,
    ) -> None:
        self.interval_sec = interval_sec
        self.timeout_sec = timeout_sec
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self._last_request_at: float = 0.0

    def get(self, url: str, *, params: Optional[dict] = None, max_retries: int = 2) -> str:
        # リクエスト間隔を担保
        wait = self.interval_sec - (time.time() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)

        last_exc: Optional[Exception] = None
        # (connect timeout, read timeout) - 読み込みが遅い場合に hang しないよう分離
        timeout = (5.0, float(self.timeout_sec))
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=timeout)
                self._last_request_at = time.time()
                resp.raise_for_status()
                resp.encoding = resp.apparent_encoding or "utf-8"
                return resp.text
            except requests.RequestException as e:
                last_exc = e
                backoff = 2 ** (attempt - 1)
                logger.warning("GET failed (%s/%s) url=%s err=%s retry in %ss",
                               attempt, max_retries, url, e, backoff)
                time.sleep(backoff)
        assert last_exc is not None
        raise last_exc
