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
