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
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# Canonical path to the public repo on each executor (set up by the
# bootstrap audit, Phase 0). Identical to the constant in runner.py.
REMOTE_REPO_PATH = "C:/ComfyWorkflowVS/comfy-workflow"

# Canonical models tree on each executor (Windows portable layout).
# Forward slashes accepted by both cmd and PowerShell.
REMOTE_MODELS_BASE = "C:/ComfyUI_windows_portable/ComfyUI/models"


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


def _load_manifest(manifest_path: Path) -> list[ModelEntry]:
    """Load and validate the manifest YAML against the V1 schema.

    Validation is structural and value-level: file existence, valid
    YAML, ``schema_version == 1``, ``models`` list of mappings each
    with non-empty ``name`` / ``tier`` / ``files``, and each file
    mapping with non-empty ``path`` / ``url`` and positive integer
    ``size_bytes``. The first failure raises with a path-qualified
    error message.

    Returns:
        Parsed list of :class:`ModelEntry` (one per manifest entry).

    Raises:
        InstallerManifestError: missing file, invalid YAML, or any
            schema mismatch.
    """
    if not manifest_path.is_file():
        raise InstallerManifestError(f"manifest file not found: {manifest_path}")

    try:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise InstallerManifestError(
            f"manifest {manifest_path} is not valid YAML: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise InstallerManifestError(
            f"manifest {manifest_path} top-level must be a mapping, "
            f"got {type(data).__name__}"
        )

    schema_version = data.get("schema_version")
    if schema_version != 1:
        raise InstallerManifestError(
            f"manifest {manifest_path} schema_version must be 1, "
            f"got {schema_version!r}"
        )

    raw_models = data.get("models")
    if not isinstance(raw_models, list):
        raise InstallerManifestError(
            f"manifest {manifest_path} 'models' must be a list, "
            f"got {type(raw_models).__name__}"
        )

    models: list[ModelEntry] = []
    for i, raw_model in enumerate(raw_models):
        if not isinstance(raw_model, dict):
            raise InstallerManifestError(
                f"manifest models[{i}] must be a mapping"
            )
        name = raw_model.get("name")
        tier = raw_model.get("tier")
        raw_files = raw_model.get("files")
        if not isinstance(name, str) or not name:
            raise InstallerManifestError(
                f"manifest models[{i}] 'name' must be a non-empty string"
            )
        if not isinstance(tier, str) or not tier:
            raise InstallerManifestError(
                f"manifest models[{i}] 'tier' must be a non-empty string"
            )
        if not isinstance(raw_files, list) or not raw_files:
            raise InstallerManifestError(
                f"manifest models[{i}] 'files' must be a non-empty list"
            )

        files: list[FileEntry] = []
        for j, raw_file in enumerate(raw_files):
            if not isinstance(raw_file, dict):
                raise InstallerManifestError(
                    f"manifest models[{i}].files[{j}] must be a mapping"
                )
            path = raw_file.get("path")
            url = raw_file.get("url")
            size_bytes = raw_file.get("size_bytes")
            if not isinstance(path, str) or not path:
                raise InstallerManifestError(
                    f"manifest models[{i}].files[{j}] "
                    "'path' must be a non-empty string"
                )
            if not isinstance(url, str) or not url:
                raise InstallerManifestError(
                    f"manifest models[{i}].files[{j}] "
                    "'url' must be a non-empty string"
                )
            if not isinstance(size_bytes, int) or size_bytes <= 0:
                raise InstallerManifestError(
                    f"manifest models[{i}].files[{j}] "
                    "'size_bytes' must be a positive integer"
                )
            files.append(FileEntry(path=path, url=url, size_bytes=size_bytes))

        models.append(ModelEntry(name=name, tier=tier, files=files))

    return models


def _ssh_run(host: str, command: str, timeout: int = 30) -> str:
    """Run a command on a remote host via SSH and return its stdout.

    Mirrors :func:`installer.benchmark.runner._ssh_run` (copy, not import,
    to keep the top-level installer self-contained).

    Args:
        host: SSH host alias.
        command: Single command line passed verbatim to ``ssh``. Quoting
            must be valid for the remote shell (cmd by default on
            Windows OpenSSH).
        timeout: Seconds to wait before raising :class:`InstallerSSHError`.

    Returns:
        Captured stdout, decoded as UTF-8.

    Raises:
        InstallerSSHError: ``ssh`` exited non-zero or the call timed out.
    """
    args = ["ssh", host, command]
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise InstallerSSHError(
            f"ssh {host}: command timed out after {timeout}s: {command[:100]!r}"
        ) from exc
    if result.returncode != 0:
        raise InstallerSSHError(
            f"ssh {host}: exit {result.returncode}: "
            f"stderr={result.stderr.strip()!r}"
        )
    return result.stdout


def _ssh_pull(host: str, repo_path: str = REMOTE_REPO_PATH) -> str:
    """Run ``git pull`` on the remote host's clone. Returns stdout for logging.

    Args:
        host: SSH host alias.
        repo_path: Absolute path to the repo on the remote (forward
            slashes). Defaults to :data:`REMOTE_REPO_PATH`.

    Raises:
        InstallerSSHError: ``git pull`` failed or timed out.
    """
    return _ssh_run(host, f'git -C "{repo_path}" pull', timeout=60)


def _check_remote_file(host: str, remote_path: str) -> tuple[bool, int]:
    """Check whether ``remote_path`` exists on ``host`` and report its size.

    Uses PowerShell ``Test-Path`` + ``(Get-Item ...).Length`` via SSH.
    The remote default shell is cmd on Windows OpenSSH; the PowerShell
    command is wrapped accordingly with cmd-style outer quotes and
    PowerShell-style inner single quotes.

    Args:
        host: SSH host alias.
        remote_path: Absolute path on the remote. Forward slashes are
            accepted by PowerShell.

    Returns:
        Tuple ``(exists, size_bytes)``. ``size_bytes`` is ``0`` when
        ``exists`` is ``False``.

    Raises:
        InstallerSSHError: SSH failed, or the output could not be parsed
            as ``MISSING`` or a non-negative integer.
    """
    remote_cmd = (
        f"powershell -NoProfile -Command "
        f"\"if (Test-Path '{remote_path}') "
        f"{{ (Get-Item '{remote_path}').Length }} "
        f"else {{ 'MISSING' }}\""
    )
    output = _ssh_run(host, remote_cmd, timeout=30).strip()
    if output == "MISSING":
        return (False, 0)
    try:
        size = int(output)
    except ValueError as exc:
        raise InstallerSSHError(
            f"_check_remote_file({host}, {remote_path!r}): "
            f"cannot parse PowerShell output as int: {output!r}"
        ) from exc
    if size < 0:
        raise InstallerSSHError(
            f"_check_remote_file({host}, {remote_path!r}): "
            f"PowerShell returned negative size: {size}"
        )
    return (True, size)
