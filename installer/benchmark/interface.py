"""HTTP client wrapper for the ComfyUI server API.

Single responsibility: speak to the ComfyUI server. This module does NOT
orchestrate benchmarks (runner.py), collect hardware metrics (snapshot.py),
or download models (installer.py).

Typical usage::

    client = ComfyUIClient("http://127.0.0.1:8188")
    if client.is_alive():
        stats = client.system_stats()

Each ``ComfyUIClient`` instance is single-client and NOT thread-safe.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

import requests

logger = logging.getLogger(__name__)


class ComfyUIError(Exception):
    """Base class for all ComfyUI client errors."""


class ComfyUIConnectionError(ComfyUIError):
    """Raised when the client cannot reach the ComfyUI server (network failure)."""


class ComfyUITimeoutError(ComfyUIError):
    """Raised when a request to the ComfyUI server exceeds its timeout."""


class ComfyUIAPIError(ComfyUIError):
    """Raised when the ComfyUI server returns a 4xx or 5xx HTTP response."""


class ComfyUIQueueError(ComfyUIError):
    """Raised when the ComfyUI server rejects a workflow at ``/prompt``."""


def _safe_request[T](
    fn: Callable[[], T],
    errors: list[str],
    err_msg: str,
) -> T | None:
    """Call ``fn`` and append a contextualized error to ``errors`` on failure.

    EN equivalent of ``audit._safe_call`` (private repo, PT). Same pattern:
    wrap external calls (HTTP requests, JSON parsing) for granular fault
    tolerance. Returns the value of ``fn`` on success, ``None`` if an
    exception was raised.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — intentional broad catch for fault tolerance
        errors.append(f"{err_msg}: {exc}")
        logger.debug("Suppressed exception in _safe_request (%s): %r", err_msg, exc)
        return None


class ComfyUIClient:
    """HTTP wrapper for the ComfyUI server API.

    A ``ComfyUIClient`` instance is single-client and NOT thread-safe. Spawn
    one client per orchestration context; do not share an instance across
    threads or processes.
    """

    def __init__(self, base_url: str, timeout: int = 30) -> None:
        """Initialize the client.

        Args:
            base_url: Base URL of the ComfyUI server, e.g.
                ``http://127.0.0.1:8188``. Trailing slashes are stripped.
            timeout: Default per-request timeout in seconds. Used for
                short-lived GET endpoints; long-poll methods accept their
                own timeout argument.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def _get(self, path: str, timeout: int | None = None) -> dict[str, Any]:
        """Issue an HTTP GET and decode the JSON object body.

        Args:
            path: URL path starting with ``/``, e.g. ``/system_stats``.
            timeout: Optional override of the client's default timeout.

        Returns:
            Parsed JSON body as a dict.

        Raises:
            ComfyUIConnectionError: Network failure (connection refused,
                DNS error, etc.).
            ComfyUITimeoutError: Request exceeded the timeout.
            ComfyUIAPIError: Server returned a 4xx/5xx response, or the
                body could not be parsed as a JSON object.
            ComfyUIError: Any other ``requests`` failure.
        """
        url = f"{self.base_url}{path}"
        effective_timeout = timeout if timeout is not None else self.timeout
        try:
            response = self._session.get(url, timeout=effective_timeout)
        except requests.Timeout as exc:
            raise ComfyUITimeoutError(
                f"GET {url} timed out after {effective_timeout}s"
            ) from exc
        except requests.ConnectionError as exc:
            raise ComfyUIConnectionError(f"GET {url} failed to connect: {exc}") from exc
        except requests.RequestException as exc:
            raise ComfyUIError(f"GET {url} failed: {exc}") from exc

        if not response.ok:
            raise ComfyUIAPIError(
                f"GET {url} returned HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            data: object = response.json()
        except ValueError as exc:
            raise ComfyUIAPIError(f"GET {url} returned non-JSON body: {exc}") from exc

        if not isinstance(data, dict):
            raise ComfyUIAPIError(
                f"GET {url} returned non-object JSON: {type(data).__name__}"
            )
        return cast(dict[str, Any], data)

    def system_stats(self) -> dict[str, Any]:
        """Fetch ``/system_stats`` from the ComfyUI server.

        Returns:
            JSON dict with ComfyUI version info, system info, and per-device
            VRAM totals. See ComfyUI source for the exact schema.

        Raises:
            ComfyUIConnectionError, ComfyUITimeoutError, ComfyUIAPIError,
            ComfyUIError: See :meth:`_get`.
        """
        return self._get("/system_stats")

    def is_alive(self) -> bool:
        """Check whether the ComfyUI server is responsive.

        Issues ``/system_stats`` with a short fixed timeout (5 seconds),
        regardless of the client's default timeout. Returns ``False`` on
        any failure: connection refused, timeout, HTTP error, invalid JSON.
        Never raises ``ComfyUIError`` subclasses.
        """
        try:
            self._get("/system_stats", timeout=5)
        except ComfyUIError as exc:
            logger.debug("is_alive() returning False: %r", exc)
            return False
        return True
