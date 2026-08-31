"""
Monitor — first step of the MAPE-K loop.
Discovers hardware, environment, and verifies preflight checks.
Output: HardwareProfile populated.
"""

from .hardware import HardwareProfile, probe_hardware
from .environment import run_environment_checks
from .preflight import run_preflight_checks
from typing import Optional
from .pipeline_config import parse_pipeline_dsl


def run_monitor(repo_root: str = ".",
                compose_file: Optional[str] = None) -> HardwareProfile:
    """
    Executes the entire Monitor phase:
    1. Hardware profiling (GPU, RAM, CPU, disk)
    2. Environment check (nextflow, docker, licenses, container)
    3. Preflight checks (critical constraints and fallbacks)

    Returns a completely populated HardwareProfile.
    """
    print("[Monitor] Discovering hardware...")
    profile = probe_hardware(work_dir=repo_root)

    #relieve DSL version from pipeline file
    profile.pipeline_dsl = parse_pipeline_dsl(repo_root)

    print("[Monitor] Checking environment...")
    run_environment_checks(profile, repo_root=repo_root, compose_file=compose_file)

    print("[Monitor] Running preflight checks...")
    run_preflight_checks(profile)

    return profile
