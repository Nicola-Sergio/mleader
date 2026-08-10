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


def classify_failure(log_path: str = ".nextflow.log") -> FailureCause:
    """
    Reads the Nextflow log and returns the cause of the failure.
    If the log does not exist or does not match any pattern, returns UNKNOWN.
    """
    path = Path(log_path)
    if not path.exists():
        return FailureCause.UNKNOWN

    try:
        content = path.read_text(errors="replace")
    except OSError:
        return FailureCause.UNKNOWN

    for cause, patterns in _PATTERNS:
        for pattern in patterns:
            if pattern in content:
                return cause

    return FailureCause.UNKNOWN
