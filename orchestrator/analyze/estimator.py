"""
Capacity estimator — Analyze component.

Estimates optimal parameters for the FTD pipeline based on:
1. Empirical VRAM measurement from dry_run (fastsurfer GPU only)
2. Empirical peak_rss from trace TSV files (all processes)
3. Hardware-conservative fallback (cold start — no data available)

Parameter resolution per segmenter:

  fastsurfer (cuda):
    - VRAM from dry_run        -> maxForks = floor(vram_free / vram_cost_per_subject)
    - RAM proxy from trace GPU -> maxForks = floor(ram_available / ram_cost_per_subject)
    - Cold start               -> maxForks = 1 (safe: unknown VRAM consumption)

  freesurfer:
    - RAM from trace           -> maxForks = min(floor(ram_available / peak_rss_max), cpu_cores_free)
    - Cold start               -> maxForks = cpu_cores_free
    (FreeSurfer recon-all without -openmp uses 1 core per subject,
     confirmed by %cpu ~99% in trace files)

  pyradiomics:
    Always hardware-based: pyradiomics_jobs = cpu_threads - 1
    (PyRadiomics is single-instance per run, not per-subject)

No safety margin is applied to peak_rss because the maximum observed
value across all completed subjects is already a conservative worst-case
estimate. Adding a further margin would be doubly conservative without
statistical justification.

If a future subject exceeds the observed peak_rss_max, the OOM retry
mechanism in the Execute phase will reduce maxForks and resume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..monitor.hardware import HardwareProfile


@dataclass
class ExecutionPlan:
    """
    Output of the Analyze phase.
    Contains all parameter decisions to be passed to Plan.
    """
    brain_segmenter:    str
    fastsurfer_device:  Optional[str]
    maxforks_segmenter: int
    fastsurfer_threads: Optional[int]
    pyradiomics_jobs:   int

    # metadata for reporting
    vram_free_gb:           Optional[float]
    ram_available_gb:       float
    cpu_threads:            int
    cpu_cores_free:         int
    vram_per_subject_gb:    Optional[float]
    ram_per_subject_gb:     Optional[float]
    source:                 str


def estimate_params(
    profile: HardwareProfile,
    vram_per_subject_gb: Optional[float] = None,
    ram_per_subject_gb_freesurfer: Optional[float] = None,
    ram_per_subject_gb_fastsurfer_gpu: Optional[float] = None,
    duration_mean_min_freesurfer: Optional[float] = None,
    duration_mean_min_fastsurfer: Optional[float] = None,
) -> ExecutionPlan:
    """
    Calculates optimal parameters for the FTD pipeline.

    maxForks is computed as:
        floor(available_resource / cost_per_subject)

    where cost_per_subject is the maximum peak_rss observed across all
    completed subjects in the trace files. No safety margin is applied —
    the observed maximum is already a conservative worst-case estimate.

    Parameters
    ----------
    profile : HardwareProfile
        Populated by the Monitor phase.
    vram_per_subject_gb : float, optional
        VRAM cost per subject measured by dry_run (fastsurfer GPU).
    ram_per_subject_gb_freesurfer : float, optional
        RAM cost per subject from trace files (freesurfer peak_rss_max).
    ram_per_subject_gb_fastsurfer_gpu : float, optional
        RAM host cost per subject from trace files (fastsurfer GPU mode peak_rss_max).
        Used as RAM proxy when VRAM is not available from trace.
    duration_mean_min_freesurfer : float, optional
        Mean duration per subject in minutes (freesurfer), used for throughput comparison.
    duration_mean_min_fastsurfer : float, optional
        Mean duration per subject in minutes (fastsurfer GPU), used for throughput comparison.
    """

    # ── Free CPU cores estimate ───────────────────────────────────────
    load_1min      = getattr(profile, 'cpu_load_1min', 0.0)
    cores_busy     = min(int(load_1min), profile.cpu_cores)
    cpu_cores_free = max(1, profile.cpu_cores - cores_busy)

    # ── GPU availability ──────────────────────────────────────────────
    gpu_available = (
        profile.gpu is not None
        and profile.docker_gpu_runtime
        and "brain_segmenter" not in profile.fallbacks
    )

    # ── Compute maxForks for both segmenters ──────────────────────────
    # FreeSurfer
    if ram_per_subject_gb_freesurfer is not None:
        maxforks_fs = min(
            math.floor(profile.ram_available_gb / ram_per_subject_gb_freesurfer),
            cpu_cores_free
        )
        source_fs = "trace_empirical"
    else:
        maxforks_fs = cpu_cores_free
        source_fs   = "hardware_conservative"

    # FastSurfer GPU
    if gpu_available:
        if vram_per_subject_gb is not None:
            maxforks_fas = min(
                math.floor(profile.gpu.vram_free_gb / vram_per_subject_gb),
                cpu_cores_free
            )
            source_fas = "dry_run"
        elif ram_per_subject_gb_fastsurfer_gpu is not None:
            maxforks_fas = min(
                math.floor(profile.ram_available_gb / ram_per_subject_gb_fastsurfer_gpu),
                cpu_cores_free
            )
            source_fas = "trace_empirical_ram_proxy"
        else:
            maxforks_fas = 1
            source_fas   = "hardware_conservative"
    else:
        maxforks_fas = 0  # fastsurfer not available
        source_fas   = "unavailable"

    # ── Throughput comparison ─────────────────────────────────────────
    # throughput = maxForks / (duration_mean_min / 60)  [subjects/hour]
    # Requires duration_mean from trace files for both segmenters.
    # If data is missing for either, fall back to GPU availability rule.
    if (
        gpu_available
        and duration_mean_min_freesurfer is not None
        and duration_mean_min_fastsurfer is not None
        and maxforks_fas > 0
    ):
        tp_fs  = maxforks_fs  / (duration_mean_min_freesurfer  / 60)
        tp_fas = maxforks_fas / (duration_mean_min_fastsurfer / 60)

        use_fastsurfer = tp_fas > tp_fs
        source = f"throughput_comparison (fs={tp_fs:.2f} vs fas={tp_fas:.2f} subj/h)"

        print(f"[Analyze] Throughput FreeSurfer:  {tp_fs:.2f} subj/h "
              f"(maxForks={maxforks_fs}, duration={duration_mean_min_freesurfer:.0f}min)")
        print(f"[Analyze] Throughput FastSurfer:  {tp_fas:.2f} subj/h "
              f"(maxForks={maxforks_fas}, duration={duration_mean_min_fastsurfer:.0f}min)")
        print(f"[Analyze] Winner: {'fastsurfer' if use_fastsurfer else 'freesurfer'}")

    elif gpu_available:
        # no duration data for comparison — use GPU if available
        use_fastsurfer = maxforks_fas > 0
        source = source_fas if use_fastsurfer else source_fs
    else:
        use_fastsurfer = False
        source = source_fs

    # ── Final parameter assignment ────────────────────────────────────
    if use_fastsurfer:
        brain_segmenter    = "fastsurfer"
        fastsurfer_device  = "cuda"
        maxforks           = maxforks_fas
        fastsurfer_threads = max(2, profile.cpu_threads - 1)
        ram_used           = ram_per_subject_gb_fastsurfer_gpu
        vram_reported      = vram_per_subject_gb
    else:
        brain_segmenter    = profile.fallbacks.get("brain_segmenter", "freesurfer")
        fastsurfer_device  = None
        maxforks           = maxforks_fs
        fastsurfer_threads = None
        ram_used           = ram_per_subject_gb_freesurfer
        vram_reported      = None

    # ── pyradiomics_jobs ──────────────────────────────────────────────
    pyradiomics_jobs = max(1, cpu_cores_free - 1)

    return ExecutionPlan(
        brain_segmenter    = brain_segmenter,
        fastsurfer_device  = fastsurfer_device,
        maxforks_segmenter = maxforks,
        fastsurfer_threads = fastsurfer_threads,
        pyradiomics_jobs   = pyradiomics_jobs,
        vram_free_gb       = profile.gpu.vram_free_gb if profile.gpu else None,
        ram_available_gb   = profile.ram_available_gb,
        cpu_threads        = profile.cpu_threads,
        cpu_cores_free     = cpu_cores_free,
        vram_per_subject_gb= vram_reported,
        ram_per_subject_gb = ram_used,
        source             = source,
    )