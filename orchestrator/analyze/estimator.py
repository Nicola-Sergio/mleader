"""
Capacity estimator — Analyze component.

Estimates optimal parameters for the FTD pipeline based on:
1. Empirical VRAM measurement from dry_run (fastsurfer GPU only)
2. Empirical peak_rss from trace TSV files (all processes)
3. Hardware-conservative fallback (cold start — no data available)

Parameter resolution per segmenter:

  fastsurfer (cuda):
    - VRAM from dry_run          -> maxForks = floor(vram_free * 0.80 / vram_cost)
    - RAM proxy from trace GPU   -> maxForks = floor(ram_available * 0.80 / ram_cost)
    - Cold start                 -> maxForks = 1 (safe: unknown VRAM consumption)

  fastsurfer (cpu):
    - RAM from trace CPU         -> maxForks = min(floor(ram * 0.80 / cost), cpu_cores_free)
    - Cold start                 -> maxForks = cpu_cores_free

  freesurfer:
    - RAM from trace             -> maxForks = min(floor(ram * 0.80 / cost), cpu_cores_free)
    - Cold start                 -> maxForks = cpu_cores_free
    (FreeSurfer is single-threaded: 1 core per subject, so cpu_cores_free
     is always the natural upper bound regardless of RAM data availability)

  pyradiomics:
    Always hardware-based: pyradiomics_jobs = cpu_threads - 1
    (PyRadiomics is single-instance per run, not per-subject)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..monitor.hardware import HardwareProfile

SAFETY_VRAM = 0.80
SAFETY_RAM  = 0.80


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
    ram_per_subject_gb_fastsurfer_cpu: Optional[float] = None,
) -> ExecutionPlan:
    """
    Calculates optimal parameters for the FTD pipeline.

    Parameters
    ----------
    profile : HardwareProfile
        Populated by the Monitor phase.
    vram_per_subject_gb : float, optional
        VRAM cost per subject measured by dry_run (fastsurfer GPU).
    ram_per_subject_gb_freesurfer : float, optional
        RAM cost per subject from trace files (freesurfer).
    ram_per_subject_gb_fastsurfer_gpu : float, optional
        RAM host cost per subject from trace files (fastsurfer GPU mode).
        Used as RAM proxy when VRAM is not available.
    ram_per_subject_gb_fastsurfer_cpu : float, optional
        RAM cost per subject from trace files (fastsurfer CPU mode).
    """

    # ── Free CPU cores estimate ───────────────────────────────────────
    load_1min   = getattr(profile, 'cpu_load_1min', 0.0)
    cores_busy  = min(int(load_1min), profile.cpu_cores)
    cpu_cores_free = max(1, profile.cpu_cores - cores_busy)

    # ── Segmenter decision ────────────────────────────────────────────
    use_fastsurfer = (
        profile.gpu is not None
        and profile.docker_gpu_runtime
        and "brain_segmenter" not in profile.fallbacks
    )

    if use_fastsurfer:
        brain_segmenter   = "fastsurfer"
        fastsurfer_device = "cuda"
    else:
        brain_segmenter   = profile.fallbacks.get("brain_segmenter", "freesurfer")
        fastsurfer_device = None

    # ── maxForks calculation ──────────────────────────────────────────

    if brain_segmenter == "fastsurfer":

        if vram_per_subject_gb is not None:
            # Best case: empirical VRAM from dry_run
            vram_free = profile.gpu.vram_free_gb
            maxforks  = max(1, math.floor(vram_free * SAFETY_VRAM / vram_per_subject_gb))
            fastsurfer_threads = max(2, profile.cpu_threads - 1)
            ram_used   = vram_per_subject_gb
            source     = "dry_run"

        elif ram_per_subject_gb_fastsurfer_gpu is not None:
            # Fallback: use host RAM as proxy (VRAM not available from trace)
            # Conservative but safe: if RAM is the bottleneck, respect it
            maxforks_ram = max(1, math.floor(
                profile.ram_available_gb * SAFETY_RAM / ram_per_subject_gb_fastsurfer_gpu
            ))
            maxforks  = min(maxforks_ram, cpu_cores_free)
            fastsurfer_threads = max(2, profile.cpu_threads - 1)
            ram_used   = ram_per_subject_gb_fastsurfer_gpu
            source     = "trace_empirical_ram_proxy"

        else:
            # Cold start: no data — run 1 subject at a time (always safe)
            maxforks   = 1
            fastsurfer_threads = max(1, profile.cpu_threads - 1)
            ram_used   = None
            source     = "hardware_conservative"

        vram_reported = vram_per_subject_gb

    elif brain_segmenter == "fastsurfer" and fastsurfer_device == "cpu":
        # FastSurfer on CPU (unusual, but handle it)
        if ram_per_subject_gb_fastsurfer_cpu is not None:
            maxforks_ram = max(1, math.floor(
                profile.ram_available_gb * SAFETY_RAM / ram_per_subject_gb_fastsurfer_cpu
            ))
            maxforks   = min(maxforks_ram, cpu_cores_free)
            ram_used   = ram_per_subject_gb_fastsurfer_cpu
            source     = "trace_empirical"
        else:
            maxforks   = cpu_cores_free
            ram_used   = None
            source     = "hardware_conservative"

        fastsurfer_threads = max(2, profile.cpu_threads - 1)
        vram_reported = None

    else:
        # FreeSurfer (always CPU, single-threaded)
        if ram_per_subject_gb_freesurfer is not None:
            # Empirical: derive maxForks from measured RAM cost
            # Cap at cpu_cores_free: FreeSurfer uses 1 core per subject
            maxforks_ram = max(1, math.floor(
                profile.ram_available_gb * SAFETY_RAM / ram_per_subject_gb_freesurfer
            ))
            maxforks = min(maxforks_ram, cpu_cores_free)
            ram_used = ram_per_subject_gb_freesurfer
            source   = "trace_empirical"
        else:
            # Cold start: FreeSurfer is single-threaded -> 1 core per subject
            # -> maxForks = free cores (always safe regardless of RAM)
            maxforks = cpu_cores_free
            ram_used = None
            source   = "hardware_conservative"

        fastsurfer_threads = None
        vram_reported      = None

    # ── pyradiomics_jobs ──────────────────────────────────────────────
    # Single-instance process: use all available threads minus 1 for OS
    pyradiomics_jobs = max(1, profile.cpu_threads - 1)

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