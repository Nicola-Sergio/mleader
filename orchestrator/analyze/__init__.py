"""
Analyze — second phase of the MAPE-K loop.
Evaluates hardware compatibility and estimates optimal pipeline parameters.
Output: ExecutionPlan.

Parameter resolution order:
1. dry_run measurement (most accurate — measures actual VRAM/RAM on this host)
2. empirical data from trace TSV files (host-agnostic, based on tool behavior)
3. hardware-conservative fallback (cold start — no historical data available)
"""

from typing import Optional

from .estimator import ExecutionPlan, estimate_params
from .dry_run import profile_fastsurfer_vram
from .trace_reader import get_peak_rss_for_process, summarize_traces


def run_analyze(
    profile,
    repo_root: str = ".",
    pipeline: str = "preprocessing",
    dry_run: bool = False,
    sample_nii: Optional[str] = None,
    license_path: str = "license.txt",
    custom_traces_dir: Optional[str] = None,
) -> ExecutionPlan:
    """
    Runs the Analyze phase.

    Steps:
    1. Read trace TSV files to get empirical peak_rss per process
    2. (optional) dry-run on a sample subject to measure VRAM
    3. Estimate optimal parameters based on available data

    Parameters
    ----------
    profile : HardwareProfile
        Populated by the Monitor phase.
    repo_root : str
        Root of the FTD pipeline repository. Used to locate trace files.
    pipeline : str
        "preprocessing" or "training". Determines which trace folder to read.
    dry_run : bool
        If True and GPU is available, runs fastsurfer on sample_nii to
        measure VRAM empirically (takes 15-30 minutes).
    sample_nii : str, optional
        Path to a sample .nii file for the dry-run.
    license_path : str
        Path to FreeSurfer license file (needed for dry-run).
    custom_traces_dir : str, optional
        Override the default traces directory. If None, uses
        <repo_root>/reports/traces/<pipeline>/
    """
    vram_per_subject      = None
    ram_per_subject_fs    = None

    # ── Step 1: read empirical data from trace files ──────────────────
    print("[Analyze] Reading trace files for empirical resource data...")

    # FreeSurfer RAM
    ram_per_subject_fs, n_fs = get_peak_rss_for_process(
        process_name="freesurfer",
        repo_root=repo_root,
        pipeline=pipeline,
        custom_traces_dir=custom_traces_dir,
    )
    if ram_per_subject_fs:
        print(f"[Analyze] FreeSurfer: peak_rss_max = {ram_per_subject_fs:.2f} GB "
              f"(from {n_fs} observations)")
    else:
        print("[Analyze] FreeSurfer: no trace data available — will use hardware-conservative fallback")

    # FastSurfer GPU RAM (host-side, not VRAM — used as proxy)
    ram_per_subject_fas_gpu, n_fas_gpu = get_peak_rss_for_process(
        process_name="fastsurfer",
        repo_root=repo_root,
        pipeline=pipeline,
        custom_traces_dir=custom_traces_dir,
        device_filter="cuda",
    )
    if ram_per_subject_fas_gpu:
        print(f"[Analyze] FastSurfer (cuda): peak_rss_max = {ram_per_subject_fas_gpu:.2f} GB RAM host "
              f"(from {n_fas_gpu} observations) — note: VRAM not available from trace")

    # FastSurfer CPU RAM
    ram_per_subject_fas_cpu, n_fas_cpu = get_peak_rss_for_process(
        process_name="fastsurfer",
        repo_root=repo_root,
        pipeline=pipeline,
        custom_traces_dir=custom_traces_dir,
        device_filter="cpu",
    )
    if ram_per_subject_fas_cpu:
        print(f"[Analyze] FastSurfer (cpu):  peak_rss_max = {ram_per_subject_fas_cpu:.2f} GB "
              f"(from {n_fas_cpu} observations)")

    # ── Step 2: dry-run for VRAM (optional) ──────────────────────────
    if dry_run and sample_nii and profile.gpu and profile.docker_gpu_runtime:
        print("[Analyze] Starting FastSurfer dry-run for VRAM profiling...")
        print("[Analyze] This will take 15-30 minutes...")
        vram_per_subject = profile_fastsurfer_vram(
            sample_nii=sample_nii,
            license_path=license_path,
        )
        if vram_per_subject:
            print(f"[Analyze] Measured VRAM: {vram_per_subject:.2f} GB per subject")
        else:
            print("[Analyze] Dry-run failed — VRAM not measured")

    # ── Step 3: estimate parameters ───────────────────────────────────
    print("[Analyze] Estimating optimal parameters...")

    plan = estimate_params(
        profile=profile,
        vram_per_subject_gb=vram_per_subject,
        ram_per_subject_gb_freesurfer=ram_per_subject_fs,
        ram_per_subject_gb_fastsurfer_gpu=ram_per_subject_fas_gpu,
        ram_per_subject_gb_fastsurfer_cpu=ram_per_subject_fas_cpu,
    )

    print(f"[Analyze] Parameters estimated — source: {plan.source}")
    return plan