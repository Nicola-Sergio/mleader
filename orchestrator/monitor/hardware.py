"""
Hardware profiler — Monitor component.
Wraps nvidia-smi, psutil, /proc/meminfo, df.
"""

from __future__ import annotations
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import psutil


@dataclass
class GpuInfo:
    name: str
    vram_total_gb: float
    vram_free_gb: float
    driver_version: str
    compute_capability: str
    is_vgpu: bool = False
    vgpu_license_status: Optional[str] = None


@dataclass
class HardwareProfile:
    # CPU
    cpu_cores: int
    cpu_threads: int
    cpu_load_percent: float

    # RAM
    ram_total_gb: float
    ram_available_gb: float

    # Disco
    disk_free_gb: float

    # GPU (None se assente o non rilevabile)
    gpu: Optional[GpuInfo] = None
    cpu_load_1min: float = 0.0

    # Ambiente (popolato da environment.py)
    nextflow_version: Optional[str] = None
    pipeline_dsl: Optional[str] = None
    docker_gpu_runtime: bool = False
    fs_license_present: bool = False
    containers_built: list[str] = field(default_factory=list)
    containers_missing: list[str] = field(default_factory=list)

    # Preflight check results (popolato da preflight.py)
    preflight_passed: bool = False
    preflight_errors: list[str] = field(default_factory=list)
    preflight_warnings: list[str] = field(default_factory=list)
    fallbacks: dict = field(default_factory=dict)


def probe_gpu() -> Optional[GpuInfo]:
    """
    Interroga nvidia-smi per rilevare la GPU e la VRAM disponibile.
    Restituisce None se la GPU non è presente o nvidia-smi non è installato.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode().strip()

        # Prende solo la prima GPU (singolo host)
        line = out.split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            return None

        name, vram_total, vram_free, driver, compute_cap = parts
        return GpuInfo(
            name=name,
            vram_total_gb=float(vram_total) / 1024,
            vram_free_gb=float(vram_free) / 1024,
            driver_version=driver,
            compute_capability=compute_cap,
        )
    except (subprocess.SubprocessError, FileNotFoundError, ValueError, OSError):
        return None

def check_vgpu_license(gpu: GpuInfo) -> None:
    """
    Verifica lo stato della licenza vGPU interrogando `nvidia-smi -q`.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-q"],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode(errors="replace")
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return

    if "vGPU Software Licensed Product" not in out:
        gpu.is_vgpu = False
        return

    gpu.is_vgpu = True
    match = re.search(r"License Status\s*:\s*(.+)", out)
    if match:
        gpu.vgpu_license_status = match.group(1).strip()


def probe_hardware(work_dir: str = ".") -> HardwareProfile:
    """
    Relieve hardware resources of the current host.
    """
    # CPU
    load_1min = psutil.getloadavg()[0]
    cpu_cores = psutil.cpu_count(logical=False) or 1
    cpu_threads = psutil.cpu_count(logical=True) or 1
    cpu_load = psutil.cpu_percent(interval=1)

    # RAM
    mem = psutil.virtual_memory()
    ram_total_gb = mem.total / 1e9
    ram_available_gb = mem.available / 1e9

    # Disco sul workDir
    try:
        disk = psutil.disk_usage(work_dir)
        disk_free_gb = disk.free / 1e9
    except OSError:
        disk_free_gb = 0.0

    # GPU
    gpu = probe_gpu()
    if gpu is not None:
      check_vgpu_license(gpu)

    return HardwareProfile(
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        cpu_load_percent=cpu_load,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        disk_free_gb=disk_free_gb,
        gpu=gpu,
        cpu_load_1min = load_1min,
    )
