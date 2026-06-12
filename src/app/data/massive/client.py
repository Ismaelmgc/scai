"""Massive API base client with retries, rate limiting, pagination, and logging.

Design:
  - Exponential backoff for 429, 5xx errors
  - Token-bucket rate limiter respecting plan limits
  - Automatic pagination via next_url
  - Structured logging (never leaks API key)
  - Optional raw-response persistence for auditability
  - Configurable via environment variables
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from app.utils import get_logger

log = get_logger(__name__)

# ── Defaults ────────────────────────────────────────────────
_DEFAULT_BASE_URL = "https://api.polygon.io"
_DEFAULT_TIMEOUT = 30
_DEFAULT_MAX_RETRIES = 5
_DEFAULT_CALLS_PER_MINUTE = 5  # Free plan


class RateLimiter:
    """Simple token-bucket rate limiter."""

    def __init__(self, calls_per_minute: int) -> None:
        self._interval = 60.0 / max(calls_per_minute, 1)
        self._last_call: float = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self._interval:
            sleep_time = self._interval - elapsed
            time.sleep(sleep_time)
        self._last_call = time.monotonic()


class MassiveClient:
    """Production-grade HTTP client for the Massive/Polygon REST API.

    Features:
      - Rate limiting (configurable calls/min)
      - Exponential backoff retries for transient errors
      - Automatic next_url pagination
      - Raw response persistence (optional)
      - Structured logging without leaking credentials
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
        calls_per_minute: int | None = None,
        persist_raw: bool = False,
        raw_dir: Path | None = None,
    ) -> None:
        self._key = api_key or os.environ.get("MASSIVE_API_KEY") or self._load_from_settings()
        if not self._key:
            raise ValueError(
                "Massive API key required. Set MASSIVE_API_KEY env var "
                "or SCAI_POLYGON_API_KEY in .env"
            )

        self._base_url = (
            base_url
            or os.environ.get("MASSIVE_BASE_URL")
            or _DEFAULT_BASE_URL
        )
        self._timeout = timeout or int(os.environ.get("MASSIVE_TIMEOUT", _DEFAULT_TIMEOUT))
        self._max_retries = max_retries or _DEFAULT_MAX_RETRIES
        self._persist_raw = persist_raw
        self._raw_dir = raw_dir

        cpm = calls_per_minute or int(
            os.environ.get("MASSIVE_CALLS_PER_MINUTE", _DEFAULT_CALLS_PER_MINUTE)
        )
        self._limiter = RateLimiter(cpm)

        self._client = httpx.Client(
            timeout=self._timeout,
            headers={"User-Agent": "SCAI/1.0 (small-cap-ai-platform)"},
        )
        self._request_count = 0

    @staticmethod
    def _load_from_settings() -> str:
        """Try loading from SCAI settings as fallback."""
        try:
            from app.config import get_settings
            return get_settings().polygon_api_key
        except Exception:
            return ""

    # ── Core request method ─────────────────────────────────
    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        _is_next_url: bool = False,
    ) -> dict[str, Any]:
        """Execute a GET request with rate limiting and retries.

        Parameters
        ----------
        path : str
            API path (e.g., "/v3/reference/tickers") or full next_url.
        params : dict, optional
            Query parameters (apiKey is added automatically).
        _is_next_url : bool
            If True, path is treated as a full URL (from pagination).

        Returns
        -------
        dict : Parsed JSON response body.

        Raises
        ------
        httpx.HTTPStatusError : On non-retryable 4xx errors.
        """
        params = dict(params or {})
        if not _is_next_url:
            params["apiKey"] = self._key
            url = f"{self._base_url}{path}"
        else:
            # next_url already contains apiKey
            url = path
            if "apiKey" not in url:
                sep = "&" if "?" in url else "?"
                url = f"{url}{sep}apiKey={self._key}"

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            self._limiter.wait()
            self._request_count += 1

            try:
                resp = self._client.get(url, params=params if not _is_next_url else None)

                if resp.status_code == 200:
                    data = resp.json()
                    if self._persist_raw:
                        self._save_raw(path, params, data)
                    return data

                if resp.status_code == 429:
                    wait = min(2 ** attempt * 15, 120)
                    log.warning("rate_limited", attempt=attempt, wait_s=wait)
                    time.sleep(wait)
                    last_error = httpx.HTTPStatusError(
                        "429 Too Many Requests", request=resp.request, response=resp
                    )
                    continue

                if resp.status_code in (500, 502, 503, 504):
                    wait = min(2 ** attempt * 5, 60)
                    log.warning("server_error", status=resp.status_code,
                                attempt=attempt, wait_s=wait)
                    time.sleep(wait)
                    last_error = httpx.HTTPStatusError(
                        f"{resp.status_code}", request=resp.request, response=resp
                    )
                    continue

                if resp.status_code in (401, 403):
                    log.error("auth_error", status=resp.status_code, path=path)
                    resp.raise_for_status()

                # Other 4xx
                resp.raise_for_status()

            except httpx.TimeoutException as e:
                wait = min(2 ** attempt * 5, 60)
                log.warning("timeout", attempt=attempt, wait_s=wait)
                time.sleep(wait)
                last_error = e
                continue

            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as e:
                wait = min(2 ** attempt * 5, 60)
                log.warning("connection_error", attempt=attempt, wait_s=wait, error=str(e)[:80])
                time.sleep(wait)
                last_error = e
                continue

            except httpx.HTTPStatusError:
                raise

        if last_error:
            raise last_error
        raise RuntimeError(f"Request failed after {self._max_retries} retries: {path}")

    # ── Pagination helper ───────────────────────────────────
    def get_all_pages(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        max_pages: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch all pages of a paginated endpoint.

        Follows next_url until exhausted or max_pages reached.
        Returns combined list of all 'results' arrays.
        """
        all_results: list[dict[str, Any]] = []
        data = self.get(path, params)
        all_results.extend(data.get("results", []))

        pages = 1
        while pages < max_pages:
            next_url = data.get("next_url")
            if not next_url:
                break
            data = self.get(next_url, _is_next_url=True)
            all_results.extend(data.get("results", []))
            pages += 1

        log.info("paginated_fetch", path=path, total_results=len(all_results), pages=pages)
        return all_results

    def iter_pages(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        """Yield pages one at a time for memory-efficient processing."""
        data = self.get(path, params)
        results = data.get("results", [])
        if results:
            yield results

        while True:
            next_url = data.get("next_url")
            if not next_url:
                break
            data = self.get(next_url, _is_next_url=True)
            results = data.get("results", [])
            if results:
                yield results

    # ── Raw persistence ─────────────────────────────────────
    def _save_raw(self, path: str, params: dict[str, Any], data: dict[str, Any]) -> None:
        if not self._raw_dir:
            return
        import json
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        safe_path = path.replace("/", "_").strip("_")[:80]
        out = self._raw_dir / f"{ts}_{safe_path}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        # Strip apiKey from saved params
        safe_params = {k: v for k, v in params.items() if k != "apiKey"}
        payload = {"path": path, "params": safe_params, "response": data, "ingested_at": ts}
        out.write_text(json.dumps(payload, default=str))

    # ── Metadata ────────────────────────────────────────────
    @property
    def request_count(self) -> int:
        return self._request_count

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> MassiveClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
