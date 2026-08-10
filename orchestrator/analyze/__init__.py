"""
Analyze — second phase of the MAPE-K loop.
Evaluates hardware compatibility and estimates optimal pipeline parameters.
Output: ExecutionPlan.

Parameter resolution order:
1. dry_run measurement (most accurate — measures actual VRAM/RAM on this host)
2. empirical data from trace TSV files (peak_rss_max and duration_mean)
3. hardware-conservative fallback (cold start — no historical data available)

Segmenter selection:
  If GPU is available and trace data exists for both segmenters,
  the module selects the segmenter with the highest estimated throughput:
      throughput = maxForks / (duration_mean_min / 60)  [subjects/hour]
  Otherwise falls back to GPU availability rule (fastsurfer if GPU present).
"""

from typing import Optional

from .estimator import ExecutionPlan, estimate_params
from .dry_run import profile_fastsurfer_vram
from .trace_reader import get_peak_rss_for_process, extract_process_stats, find_trace_files


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
    1. Read trace TSV files to get empirical peak_rss_max and duration_mean
       per process (freesurfer and fastsurfer)
    2. (optional) dry-run on a sample subject to measure VRAM
    3. Estimate optimal parameters and select segmenter via throughput comparison

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
    vram_per_subject         = None
    ram_per_subject_fs       = None
    ram_per_subject_fas_gpu  = None
    duration_mean_fs         = None
    duration_mean_fas        = None

    # ── Step 1: read empirical data from trace files ──────────────────
    print("[Analyze] Reading trace files for empirical resource data...")

    tsv_files = find_trace_files(repo_root, pipeline, custom_traces_dir)

    if tsv_files:
        # FreeSurfer — peak_rss_max and duration_mean
        stats_fs = extract_process_stats(tsv_files, "freesurfer")
        if stats_fs:
            ram_per_subject_fs = stats_fs.get("peak_rss_max_gb")
            duration_mean_fs   = stats_fs.get("duration_mean_min")
            print(f"[Analyze] FreeSurfer: peak_rss_max = {ram_per_subject_fs:.2f} GB, "
                  f"duration_mean = {duration_mean_fs:.1f} min "
                  f"(from {stats_fs['count']} observations)")
        else:
            print("[Analyze] FreeSurfer: no trace data — will use hardware-conservative fallback")

        # FastSurfer GPU — peak_rss_max and duration_mean
        stats_fas = extract_process_stats(tsv_files, "fastsurfer", device_filter="cuda")
        if stats_fas:
            ram_per_subject_fas_gpu = stats_fas.get("peak_rss_max_gb")
            duration_mean_fas       = stats_fas.get("duration_mean_min")
            print(f"[Analyze] FastSurfer (cuda): peak_rss_max = {ram_per_subject_fas_gpu:.2f} GB RAM host, "
                  f"duration_mean = {duration_mean_fas:.1f} min "
                  f"(from {stats_fas['count']} observations) "
                  f"— note: VRAM not available from trace, see github.com/nextflow-io/nextflow/issues/4286")
        else:
            print("[Analyze] FastSurfer (cuda): no trace data available")
    else:
        print("[Analyze] No trace files found — will use hardware-conservative fallback")

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
        duration_mean_min_freesurfer=duration_mean_fs,
        duration_mean_min_fastsurfer=duration_mean_fas,
    )

    print(f"[Analyze] Parameters estimated — source: {plan.source}")
    return plan