from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import requests


@dataclass
class RateLimiter:
    min_interval_s: float
    _last_ts: float = 0.0

    def wait(self) -> None:
        now = time.time()
        delta = now - self._last_ts
        if delta < self.min_interval_s:
            time.sleep(self.min_interval_s - delta)
        self._last_ts = time.time()


def request_with_backoff(
    method: str,
    url: str,
    *,
    session: Optional[requests.Session] = None,
    timeout: float = 30,
    max_retries: int = 8,
    backoff_base_s: float = 2.0,
    backoff_max_s: float = 120.0,
    retry_statuses: tuple[int, ...] = (429, 503),
    **kwargs: Any,
) -> requests.Response:
    sess = session or requests.Session()
    attempt = 0
    while True:
        resp = sess.request(method, url, timeout=timeout, **kwargs)
        if resp.status_code not in retry_statuses:
            return resp
        attempt += 1
        if attempt > max_retries:
            return resp
        sleep_s = min(backoff_max_s, backoff_base_s * (2 ** (attempt - 1)))
        time.sleep(sleep_s)

