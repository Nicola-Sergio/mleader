"""
Execute — quarta fase del loop MAPE-K.
Supervisiona il lifecycle della pipeline con retry adattivo.
"""

from .supervisor import supervise, RunResult
from .log_parser import classify_failure, FailureCause


def run_execute(
    pipeline: str,
    config_path: str,
    auto: bool = False,
) -> RunResult:
    """
    Esegue la fase Execute: lancia Nextflow e supervisiona.
    """
    print(f"[Execute] Pipeline: {pipeline}")
    print(f"[Execute] Config:   {config_path}")
    return supervise(pipeline, config_path, auto=auto)
