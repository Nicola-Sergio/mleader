"""
Analyze — seconda fase del loop MAPE-K.
Valuta compatibilità e stima i parametri ottimali.
Output: ExecutionPlan.
"""

from .estimator import ExecutionPlan, estimate_params
from .dry_run import profile_fastsurfer_vram


def run_analyze(
    profile,
    dry_run: bool = False,
    sample_nii: str = None,
    license_path: str = "license.txt",
) -> ExecutionPlan:
    """
    Esegue la fase Analyze:
    1. (opzionale) dry-run su soggetto campione per misurare VRAM
    2. Stima parametri ottimali in base alle risorse disponibili

    Se dry_run=True e sample_nii è fornito, misura empiricamente
    il costo VRAM per soggetto (principio Lotaru).
    Altrimenti usa i valori della knowledge base.
    """
    vram_per_subject = None

    if dry_run and sample_nii and profile.gpu and profile.docker_gpu_runtime:
        print("[Analyze] Avvio dry-run per profiling VRAM...")
        vram_per_subject = profile_fastsurfer_vram(
            sample_nii=sample_nii,
            license_path=license_path,
        )
        if vram_per_subject:
            print(f"[Analyze] Costo VRAM misurato: {vram_per_subject:.2f}GB per soggetto")
        else:
            print("[Analyze] Dry-run fallito, uso knowledge base")

    print("[Analyze] Stima parametri...")
    plan = estimate_params(profile, vram_per_subject_gb=vram_per_subject)

    if vram_per_subject:
        plan.source = "dry_run"

    return plan
