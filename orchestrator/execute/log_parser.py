"""
Log parser — Execute component.
Reads .nextflow.log and classifies the cause of the failure
to decide the retry strategy.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path


class FailureCause(Enum):
    OOM_VRAM          = "oom_vram"
    OOM_RAM           = "oom_ram"
    MISSING_PACKAGE   = "missing_package"
    DSL_MISMATCH      = "dsl_mismatch"
    CONTAINER_ERROR   = "container_error"
    GPU_NOT_AVAILABLE = "gpu_not_available"
    UNKNOWN           = "unknown"


# Text patterns to match in the log file for each failure cause.
_PATTERNS: list[tuple[FailureCause, list[str]]] = [
    (FailureCause.OOM_VRAM, [
        "CUDA out of memory",
        "CUDA error: out of memory",
        "RuntimeError: CUDA",
        "out of memory on device",
    ]),
    (FailureCause.OOM_RAM, [
        "Killed",
        "std::bad_alloc",
        "Cannot allocate memory",
        "MemoryError",
    ]),
    (FailureCause.MISSING_PACKAGE, [
        "there is no package called",
        "ModuleNotFoundError",
        "No module named",
        "ImportError",
        "library(",
    ]),
    (FailureCause.DSL_MISMATCH, [
        "DSL2 is not supported",
        "Nextflow DSL1",
        "process is not a valid",
        "Unexpected token",
    ]),
    (FailureCause.CONTAINER_ERROR, [
        "Unable to find image",
        "docker: Error response",
        "container failed",
        "OCI runtime",
    ]),
    (FailureCause.GPU_NOT_AVAILABLE, [
        "CUDA driver version is insufficient",
        "no kernel image is available",
        "could not select device driver",
        "GPU access is not available",
    ]),
]


def _match_patterns(content: str) -> FailureCause | None:
    """Returns the first FailureCause whose pattern appears in content, else None."""
    for cause, patterns in _PATTERNS:
        for pattern in patterns:
            if pattern in content:
                return cause
    return None


def _classify_from_command_err(work_dir: str | Path) -> FailureCause:
    """
    Fallback scan of the Nextflow work directory.

    Nextflow writes per-task stderr to work/<2>/<30>/.command.err. OOM messages
    emitted by the tool (e.g. "CUDA out of memory", the kernel OOM-killer
    "Killed") land there, not in .nextflow.log, which only records the generic
    exit status. Scanning these files recovers the real cause.
    """
    work_path = Path(work_dir)
    if not work_path.is_dir():
        return FailureCause.UNKNOWN

    for err_file in sorted(work_path.rglob(".command.err")):
        try:
            content = err_file.read_text(errors="replace")
        except OSError:
            continue
        cause = _match_patterns(content)
        if cause is not None:
            return cause

    return FailureCause.UNKNOWN


def classify_failure(
    log_path: str = ".nextflow.log",
    work_dir: str | Path | None = None,
) -> FailureCause:
    """
    Reads the Nextflow log and returns the cause of the failure.

    Looks first in .nextflow.log. If no pattern matches there, falls back to
    scanning the per-task .command.err files under the Nextflow work directory,
    where tool-level OOM messages are actually written. If nothing matches
    anywhere (or the log does not exist), returns UNKNOWN.

    Parameters
    ----------
    log_path : str
        Path to .nextflow.log.
    work_dir : str | Path | None
        Nextflow work directory to scan as fallback. When None, defaults to a
        'work' folder sibling of the log file (i.e. <repo_root>/work).
    """
    path = Path(log_path)
    if not path.exists():
        return FailureCause.UNKNOWN

    try:
        content = path.read_text(errors="replace")
    except OSError:
        return FailureCause.UNKNOWN

    cause = _match_patterns(content)
    if cause is not None:
        return cause

    # Fallback: the log had no recognizable pattern (typically it only holds the
    # generic "terminated with an error exit status" line) — scan .command.err.
    if work_dir is None:
        work_dir = path.parent / "work"
    return _classify_from_command_err(work_dir)
