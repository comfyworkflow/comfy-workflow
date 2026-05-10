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

import argparse
import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

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


def _download_remote(
    host: str,
    url: str,
    remote_path: str,
    expected_size: int,
    timeout: int = 1800,
) -> int:
    """Download ``url`` to ``remote_path`` on ``host`` and verify size.

    Uses the executor's local ``curl.exe`` so the bytes flow directly
    from HuggingFace to the executor (the coordinator is bypassed). The
    parent directory is ensured to exist via cmd ``mkdir`` (silent if
    already present). After download, :func:`_check_remote_file` is
    called and the observed size is compared to ``expected_size``.

    Args:
        host: SSH host alias.
        url: Source URL (HuggingFace LFS pointers are resolved by curl
            ``-L``).
        remote_path: Destination path on the remote (forward slashes
            accepted by both cmd and curl).
        expected_size: Expected byte size from the manifest, used for
            post-download verification.
        timeout: Seconds to wait for curl to finish. Default 1800
            (30 min) accommodates ~17 GB downloads on moderate links.

    Returns:
        Observed size on the remote after the download.

    Raises:
        InstallerSSHError: ssh / curl failed or timed out.
        InstallerSizeMismatchError: downloaded file is missing or has
            unexpected size.
    """
    parent_dir = remote_path.rsplit("/", 1)[0]
    remote_cmd = (
        f'mkdir "{parent_dir}" 2>nul & '
        f"curl.exe -L --fail --retry 3 --retry-delay 5 "
        f"--silent --show-error "
        f'-o "{remote_path}" "{url}"'
    )
    logger.info(
        "[%s] downloading %s -> %s (expected %d bytes)",
        host, url, remote_path, expected_size,
    )
    _ssh_run(host, remote_cmd, timeout=timeout)

    exists, actual_size = _check_remote_file(host, remote_path)
    if not exists:
        raise InstallerSizeMismatchError(
            f"download to {remote_path!r} on {host}: "
            "curl reported success but file does not exist post-download"
        )
    if actual_size != expected_size:
        raise InstallerSizeMismatchError(
            f"download to {remote_path!r} on {host}: "
            f"size {actual_size} != expected {expected_size}"
        )
    return actual_size


def _install_file(
    host: str, file_entry: FileEntry, models_base: str
) -> FileResult:
    """Ensure ``file_entry`` is present on ``host``, downloading if absent.

    Decision tree:

    1. If file exists and size matches → :class:`FileResult` with
       ``status="skipped"`` (DA-011 additive: no-op).
    2. If file exists and size mismatches → :class:`FileResult` with
       ``status="size_mismatch"``, *not* overwritten. Caller decides
       (e.g. ``--strict``) whether to halt; Nível 3 human resolves the
       underlying issue.
    3. If file does not exist → :func:`_download_remote` is invoked
       and the result is wrapped in :class:`FileResult` with
       ``status="downloaded"``.

    Args:
        host: SSH host alias.
        file_entry: One :class:`FileEntry` from the manifest.
        models_base: Absolute path to the ComfyUI ``models/`` tree on
            the remote (typically :data:`REMOTE_MODELS_BASE`).

    Returns:
        :class:`FileResult` describing the per-file outcome.

    Raises:
        InstallerSSHError, InstallerSizeMismatchError: Propagated from
            :func:`_check_remote_file` and :func:`_download_remote`.
            ``size_mismatch`` on a *pre-existing* file is captured in
            the result, not raised.
    """
    remote_path = f"{models_base}/{file_entry.path}"
    exists, actual_size = _check_remote_file(host, remote_path)

    if exists and actual_size == file_entry.size_bytes:
        logger.info(
            "[%s] %s: skipped (size match: %d bytes)",
            host, file_entry.path, actual_size,
        )
        return FileResult(
            path=file_entry.path,
            status="skipped",
            size_bytes_actual=actual_size,
            error_message=None,
        )

    if exists and actual_size != file_entry.size_bytes:
        msg = (
            f"existing file size {actual_size} != expected "
            f"{file_entry.size_bytes}; not overwritten (DA-011 additive)"
        )
        logger.warning(
            "[%s] %s: size_mismatch — %s",
            host, file_entry.path, msg,
        )
        return FileResult(
            path=file_entry.path,
            status="size_mismatch",
            size_bytes_actual=actual_size,
            error_message=msg,
        )

    # File absent on remote — download.
    downloaded_size = _download_remote(
        host, file_entry.url, remote_path, file_entry.size_bytes,
    )
    logger.info(
        "[%s] %s: downloaded (%d bytes)",
        host, file_entry.path, downloaded_size,
    )
    return FileResult(
        path=file_entry.path,
        status="downloaded",
        size_bytes_actual=downloaded_size,
        error_message=None,
    )


def _save_summary(summary: InstallerSummary, output_path: Path) -> None:
    """Write the :class:`InstallerSummary` as pretty JSON (indent=2, UTF-8).

    Mirrors :func:`installer.benchmark.runner._save_summary`. Trailing
    newline appended for POSIX-friendliness.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, ensure_ascii=False)
        f.write("\n")


def _build_argparser() -> argparse.ArgumentParser:
    """Build the CLI parser for :func:`main`."""
    parser = argparse.ArgumentParser(
        description=(
            "Idempotent model installer for the Comfy Workflow Benchmark. "
            "Reads a manifest YAML and ensures each file is present on "
            "every --ssh-hosts target, downloading via the executor's "
            "local curl.exe (DA-011 additive: never overwrites)."
        ),
    )
    parser.add_argument(
        "--manifest",
        default="installer/benchmark/models_manifest.yaml",
        help="Path to manifest YAML (default: installer/benchmark/models_manifest.yaml).",
    )
    parser.add_argument(
        "--ssh-hosts",
        nargs="+",
        required=True,
        help="One or more SSH host aliases, space-separated.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: ./installer_outputs/<UTC-timestamp>).",
    )
    parser.add_argument(
        "--remote-models-base",
        default=REMOTE_MODELS_BASE,
        help=(
            "Absolute path of ComfyUI's models/ directory on the remote "
            f"(default: {REMOTE_MODELS_BASE.replace('%', '%%')})."
        ),
    )
    parser.add_argument(
        "--filter-tier",
        default=None,
        help="Optional tier filter (e.g. 'basic'). Default: install all tiers.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Halt on first size_mismatch (default: warn and continue).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report status of each file without downloading anything.",
    )
    return parser


def main() -> None:
    """End-to-end installer orchestrator (Bloco 17 entrypoint).

    Sequence:
        1. Load + validate manifest; optionally filter by tier.
        2. For each host: git pull (best-effort), then iterate models /
           files invoking :func:`_install_file` (or, in ``--dry-run``,
           :func:`_check_remote_file`).
        3. Aggregate per-host outcomes into :class:`InstallerSummary`.
        4. Save as pretty JSON; print to stdout.

    The installer is idempotent: re-running against the same manifest
    issues no downloads when all files are present and size-matched
    (DA-011 additive).

    Raises:
        InstallerError, InstallerSSHError, InstallerManifestError,
        InstallerSizeMismatchError: On any step failure; the script
        exits non-zero via the uncaught exception. ``--strict`` halts
        on the first ``size_mismatch`` rather than continuing.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _build_argparser().parse_args()
    manifest_path = Path(cast(str, args.manifest))
    ssh_hosts = cast("list[str]", args.ssh_hosts)
    output_dir_arg = cast("str | None", args.output_dir)
    remote_models_base = cast(str, args.remote_models_base)
    filter_tier = cast("str | None", args.filter_tier)
    strict = cast(bool, args.strict)
    dry_run = cast(bool, args.dry_run)

    if output_dir_arg is None:
        ts_dir = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_dir = Path("installer_outputs") / ts_dir
    else:
        output_dir = Path(output_dir_arg)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load + filter manifest.
    models = _load_manifest(manifest_path)
    if filter_tier is not None:
        models = [m for m in models if m.tier == filter_tier]
        logger.info(
            "filtered to tier=%s: %d models remaining",
            filter_tier, len(models),
        )
    if not models:
        logger.warning("no models to install (filter-tier or empty manifest)")
    total_files = sum(len(m.files) for m in models)
    logger.info(
        "manifest %s loaded: %d models, %d files; targets: %s; dry_run=%s",
        manifest_path, len(models), total_files, ssh_hosts, dry_run,
    )

    host_results: list[HostResult] = []
    skipped_total = 0
    downloaded_total = 0
    mismatch_total = 0
    would_download_total = 0

    for host in ssh_hosts:
        logger.info("=== host %s ===", host)
        try:
            pull_out = _ssh_pull(host)
            first_line = (
                pull_out.strip().splitlines()[0]
                if pull_out.strip()
                else "(empty)"
            )
            logger.info("[%s] git pull: %s", host, first_line)
        except InstallerSSHError as exc:
            logger.warning(
                "[%s] git pull failed (proceeding anyway): %s", host, exc,
            )

        file_results: list[FileResult] = []
        for model in models:
            for file_entry in model.files:
                if dry_run:
                    remote_path = f"{remote_models_base}/{file_entry.path}"
                    exists, actual_size = _check_remote_file(host, remote_path)
                    if exists and actual_size == file_entry.size_bytes:
                        status = "skipped"
                        skipped_total += 1
                        msg: str | None = None
                    elif exists:
                        status = "size_mismatch"
                        mismatch_total += 1
                        msg = (
                            f"existing file size {actual_size} != expected "
                            f"{file_entry.size_bytes}; would not overwrite "
                            "(DA-011)"
                        )
                    else:
                        status = "would_download"
                        would_download_total += 1
                        msg = None
                        actual_size = 0
                    logger.info(
                        "[%s] %s: dry-run %s",
                        host, file_entry.path, status,
                    )
                    file_results.append(FileResult(
                        path=file_entry.path,
                        status=status,
                        size_bytes_actual=actual_size,
                        error_message=msg,
                    ))
                else:
                    result = _install_file(
                        host, file_entry, remote_models_base,
                    )
                    file_results.append(result)
                    if result.status == "skipped":
                        skipped_total += 1
                    elif result.status == "downloaded":
                        downloaded_total += 1
                    elif result.status == "size_mismatch":
                        mismatch_total += 1

        host_results.append(HostResult(host=host, files=file_results))

        if strict and any(r.status == "size_mismatch" for r in file_results):
            raise InstallerError(
                f"--strict halt: host {host} has size_mismatch entries; "
                "operator action required (Nível 3)"
            )

    # Build & save summary.
    timestamp_utc = (
        datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    summary = InstallerSummary(
        schema_version=1,
        timestamp_utc=timestamp_utc,
        manifest_path=str(manifest_path),
        hosts=host_results,
    )
    summary_path = output_dir / "summary.json"
    _save_summary(summary, summary_path)
    logger.info("saved summary to %s", summary_path)
    logger.info(
        "totals: skipped=%d downloaded=%d size_mismatch=%d would_download=%d",
        skipped_total, downloaded_total, mismatch_total, would_download_total,
    )

    # Print pretty JSON to stdout.
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
