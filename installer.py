"""Idempotent model installer for the Comfy Workflow Benchmark CGs.

Single responsibility: read a manifest of required models and ensure each
file is present on every target executor (cg-3060 / cg-4090 / cg-5090).
This module dispatches via SSH; the executor's local ``curl.exe`` performs
the actual download against HuggingFace, so model bytes never traverse
the coordinator (Itapoá).

Operating principle (DA-011, "additive"): the installer **never removes,
renames, or overwrites** existing files. If a file with the same path is
present but its byte size does not match the manifest, the installer
records :class:`FileResult` with ``status="size_mismatch"`` and the
caller (``--strict``) decides whether to halt or continue with a
warning. Resolving size mismatches (corrupted partial downloads,
version drift) is a Nível 3 human action — the installer surfaces the
issue but does not delete data.

V1 validation: byte size only. SHA256 verification (HuggingFace exposes
sha256 via LFS pointer) is catalogued as V2 débito #10.

Architecture:
    1. ``installer.py`` runs on the coordinator (Itapoá), reads the
       manifest, and iterates over ``--ssh-hosts``.
    2. For each host, ``ssh <host> powershell ...`` checks whether the
       file exists and matches the expected size.
    3. If absent, ``ssh <host> cmd /c "curl.exe -L ..."`` downloads the
       file directly into the executor's ``ComfyUI/models/`` tree from
       HuggingFace via the executor's own uplink.
    4. Per-host results are aggregated into a JSON ``InstallerSummary``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class InstallerError(Exception):
    """Base class for all installer module errors."""


class InstallerSSHError(InstallerError):
    """Raised when an SSH command fails (non-zero exit, timeout, etc.)."""


class InstallerManifestError(InstallerError):
    """Raised when the manifest file is missing, malformed, or fails schema validation."""


class InstallerSizeMismatchError(InstallerError):
    """Raised when an existing file's size does not match the manifest.

    Surfaces the mismatch to the operator (Nível 3) — the installer never
    deletes or overwrites the file. Caller (e.g. :func:`main` with
    ``--strict``) decides whether to halt or continue with a warning.
    """


@dataclass(frozen=True, slots=True)
class FileEntry:
    """A single file declaration from the manifest.

    Attributes:
        path: Path relative to ComfyUI's ``models/`` directory.
        url: HTTP(S) URL to download from (typically a HuggingFace LFS
            pointer that follows redirects).
        size_bytes: Expected byte size, used for V1 idempotency and
            post-download verification.
    """

    path: str
    url: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """A logical model from the manifest, possibly composed of multiple files.

    Attributes:
        name: Logical name used for log messages and ``--filter-tier``.
        tier: DA-008 tier (``basic``, ``offload``, or ``high_vram``).
        files: List of :class:`FileEntry` belonging to this model.
    """

    name: str
    tier: str
    files: list[FileEntry]


@dataclass(frozen=True, slots=True)
class FileResult:
    """Per-file outcome of one installer invocation against one host.

    Attributes:
        path: ``files[].path`` from the manifest, echoed for traceability.
        status: One of ``"skipped"`` (already present, size match),
            ``"downloaded"`` (newly downloaded, size verified), or
            ``"size_mismatch"`` (existed but with wrong size; not
            overwritten per DA-011).
        size_bytes_actual: Size observed on the remote after any
            download. ``0`` if the file does not exist and was not
            downloaded (e.g. ``--dry-run``).
        error_message: Human-readable explanation when ``status`` is
            ``"size_mismatch"``; ``None`` otherwise.
    """

    path: str
    status: str
    size_bytes_actual: int
    error_message: str | None


@dataclass(frozen=True, slots=True)
class HostResult:
    """Per-host aggregation of file outcomes.

    Attributes:
        host: SSH host alias (e.g. ``"cg-3060"``).
        files: List of :class:`FileResult` produced for this host.
    """

    host: str
    files: list[FileResult]


@dataclass(frozen=True, slots=True)
class InstallerSummary:
    """Schema-versioned installer output (``schema_version=1``).

    Attributes:
        schema_version: Output schema version. Currently ``1``.
        timestamp_utc: ISO-8601 UTC timestamp of the invocation.
        manifest_path: Path to the manifest YAML used for this run.
        hosts: One :class:`HostResult` per ``--ssh-hosts`` argument.
    """

    schema_version: int
    timestamp_utc: str
    manifest_path: str
    hosts: list[HostResult]
