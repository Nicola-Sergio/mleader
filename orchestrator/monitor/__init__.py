"""
Monitor — prima fase del loop MAPE-K.
Rileva hardware, ambiente e verifica i preflight check.
Output: HardwareProfile popolato.
"""

from .hardware import HardwareProfile, probe_hardware
from .environment import run_environment_checks
from .preflight import run_preflight_checks


def run_monitor(repo_root: str = ".",
                compose_file: Optional[str] = None) -> HardwareProfile:
    """
    Esegue l'intera fase Monitor:
    1. Hardware profiling (GPU, RAM, CPU, disco)
    2. Environment check (nextflow, docker, licenze, container)
    3. Preflight checks (vincoli critici e fallback)

    Restituisce un HardwareProfile completamente popolato.
    """
    print("[Monitor] Rilevamento hardware...")
    profile = probe_hardware(work_dir=repo_root)

    print("[Monitor] Verifica ambiente...")
    run_environment_checks(profile, repo_root=repo_root, compose_file=compose_file)

    print("[Monitor] Preflight checks...")
    run_preflight_checks(profile)

    return profile
