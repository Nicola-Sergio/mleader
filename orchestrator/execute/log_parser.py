"""
Log parser — Execute component.
Legge .nextflow.log e classifica la causa del fallimento
per decidere la strategia di retry.
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


# Pattern di testo da cercare nel log per classificare il fallimento
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
    Legge il log di Nextflow e restituisce la causa del fallimento.
    Se il log non esiste o non matcha nessun pattern, restituisce UNKNOWN.
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
