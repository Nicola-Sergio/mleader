"""
Execute — fourth phase of the MAPE-K loop.
Supervises the lifecycle of the pipeline with adaptive retry.
"""

from .supervisor import supervise, RunResult
from .log_parser import classify_failure, FailureCause


def run_execute(
    pipeline: str,
    config_path: str,
    auto: bool = False,
    repo_root: str = "."
) -> RunResult:
    """
    Executes the Execute phase: launches Nextflow and supervises it.
    """
    print(f"[Execute] Pipeline: {pipeline}")
    print(f"[Execute] Config:   {config_path}")
    return supervise(pipeline, config_path, auto=auto)
