"""
Capacity estimator — Analyze component.
Stima i parametri ottimali per ogni tool della pipeline
in base alle risorse hardware disponibili.
Ispirato al principio di microbenchmarking di Lotaru:
misura empiricamente invece di assumere valori fissi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from ..monitor.hardware import HardwareProfile


KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge_base" / "tool_costs.yaml"


def _load_costs() -> dict:
    with open(KNOWLEDGE_BASE_PATH) as f:
        return yaml.safe_load(f)


@dataclass
class ExecutionPlan:
    """
    Output della fase Analyze.
    Contiene tutte le decisioni sui parametri da passare al Plan.
    """
    # Segmentatore scelto
    brain_segmenter: str           # "freesurfer" o "fastsurfer"
    fastsurfer_device: Optional[str]  # "cuda", "cpu", o None se freesurfer

    # Parametri calcolati
    maxforks_segmenter: int        # maxForks per freesurfer o fastsurfer
    fastsurfer_threads: Optional[int]  # thread per istanza fastsurfer (None se freesurfer)
    pyradiomics_jobs: int          # --jobs per pyradiomics

    # Metadati utili per il report
    vram_free_gb: Optional[float]
    ram_available_gb: float
    cpu_threads: int
    vram_per_subject_gb: Optional[float]  # misurato o da knowledge base
    ram_per_subject_gb: float
    source: str  # "dry_run" o "knowledge_base"


def estimate_params(
    profile: HardwareProfile,
    vram_per_subject_gb: Optional[float] = None,
) -> ExecutionPlan:
    """
    Calcola i parametri ottimali per la pipeline FTD in base
    alle risorse rilevate dal Monitor.

    Se vram_per_subject_gb è fornito (da dry_run), viene usato quello.
    Altrimenti si usa il valore dalla knowledge base.
    """
    costs = _load_costs()
    margins = costs["safety_margins"]

    # ── Decisione segmentatore ────────────────────────────────────────

    # Usa fastsurfer solo se:
    # - GPU presente
    # - docker GPU runtime funzionante
    # - VRAM sufficiente per almeno 1 soggetto
    # Altrimenti fallback a freesurfer (già registrato nei fallbacks del Monitor)

    use_fastsurfer = (
        profile.gpu is not None
        and profile.docker_gpu_runtime
        and "brain_segmenter" not in profile.fallbacks
    )

    if use_fastsurfer:
        brain_segmenter = "fastsurfer"
        fastsurfer_device = "cuda"
    else:
        brain_segmenter = profile.fallbacks.get("brain_segmenter", "freesurfer")
        fastsurfer_device = None

    # ── maxForks segmentatore ─────────────────────────────────────────

    if brain_segmenter == "fastsurfer":
        # Bottleneck: VRAM
        vram_cost = vram_per_subject_gb or costs["fastsurfer"]["vram_gb_per_subject"]
        vram_free = profile.gpu.vram_free_gb
        maxforks = max(1, math.floor(vram_free * margins["vram"] / vram_cost))

        # fastsurfer_threads: divide i thread CPU tra le istanze parallele
        fastsurfer_threads = max(1, profile.cpu_threads // maxforks)
        ram_per_subject = costs["fastsurfer"]["ram_gb_per_subject"]

    else:
        # Bottleneck: RAM + CPU cores
        ram_per_subject = costs["freesurfer"]["ram_gb_per_subject"]
        maxforks_ram = max(1, math.floor(
            profile.ram_available_gb * margins["ram"] / ram_per_subject
        ))
        maxforks_cpu = profile.cpu_cores
        maxforks = min(maxforks_ram, maxforks_cpu)
        fastsurfer_threads = None
        vram_per_subject_gb = None
        vram_free = None

    # ── pyradiomics_jobs ─────────────────────────────────────────────
    # PyRadiomics è CPU-bound, usa thread disponibili
    # Lascia 2 thread liberi per il sistema e per Nextflow stesso
    ram_per_pyrad_job = costs["pyradiomics"]["ram_gb_per_job"]
    jobs_by_cpu = max(1, math.floor(profile.cpu_threads * margins["cpu"]) - 2)
    jobs_by_ram = max(1, math.floor(
        profile.ram_available_gb * margins["ram"] / ram_per_pyrad_job
    ))
    pyradiomics_jobs = min(jobs_by_cpu, jobs_by_ram)

    return ExecutionPlan(
        brain_segmenter=brain_segmenter,
        fastsurfer_device=fastsurfer_device,
        maxforks_segmenter=maxforks,
        fastsurfer_threads=fastsurfer_threads,
        pyradiomics_jobs=pyradiomics_jobs,
        vram_free_gb=profile.gpu.vram_free_gb if profile.gpu else None,
        ram_available_gb=profile.ram_available_gb,
        cpu_threads=profile.cpu_threads,
        vram_per_subject_gb=vram_per_subject_gb,
        ram_per_subject_gb=ram_per_subject,
        source="knowledge_base",
    )
