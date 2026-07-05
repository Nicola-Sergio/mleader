"""
Dry-run profiler — Analyze component.
Lancia fastsurfer su un singolo soggetto campione e misura
il consumo di VRAM durante l'esecuzione.
Ispirato al principio di Lotaru: misura empiricamente invece
di assumere valori a priori dalla knowledge base.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Optional


# Margine di sicurezza aggiuntivo sulla misura empirica
EMPIRICAL_SAFETY_MARGIN = 1.15


def _sample_vram_usage_mb() -> Optional[float]:
    """Campiona la VRAM usata dalla GPU in questo momento (MB)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        return float(out.split("\n")[0])
    except (subprocess.SubprocessError, ValueError):
        return None


def _monitor_vram(
    stop_event: threading.Event,
    interval_s: float,
    samples: list,
) -> None:
    """Thread che campiona la VRAM ogni interval_s secondi."""
    while not stop_event.is_set():
        usage = _sample_vram_usage_mb()
        if usage is not None:
            samples.append(usage)
        time.sleep(interval_s)


def profile_fastsurfer_vram(
    sample_nii: str,
    license_path: str,
    fastsurfer_image: str = "deepmi/fastsurfer:cuda-v2.4.2",
    sample_interval_s: float = 5.0,
) -> Optional[float]:
    """
    Lancia fastsurfer su un singolo soggetto campione e misura
    il picco di VRAM consumata durante l'esecuzione.

    Restituisce il costo stimato in GB per soggetto (con safety margin),
    oppure None se il dry-run fallisce.

    Parametri:
        sample_nii: path al file .nii del soggetto campione
        license_path: path al file license.txt di FreeSurfer
        fastsurfer_image: tag dell'immagine Docker FastSurfer
        sample_interval_s: intervallo di campionamento VRAM in secondi
    """
    nii_path = Path(sample_nii).resolve()
    lic_path = Path(license_path).resolve()

    if not nii_path.exists():
        print(f"[DryRun] File campione non trovato: {nii_path}")
        return None

    if not lic_path.exists():
        print(f"[DryRun] Licenza non trovata: {lic_path}")
        return None

    print(f"[DryRun] Profiling VRAM su soggetto campione: {nii_path.name}")
    print("[DryRun] Questo richiederà 15-30 minuti...")

    # Misura VRAM baseline prima del lancio
    baseline = _sample_vram_usage_mb() or 0.0

    # Avvia il thread di monitoraggio VRAM
    samples: list[float] = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=_monitor_vram,
        args=(stop_event, sample_interval_s, samples),
        daemon=True,
    )
    monitor_thread.start()

    # Lancia fastsurfer su 1 soggetto
    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "--entrypoint", "",
        "-v", f"{nii_path.parent}:/input:ro",
        "-v", f"{lic_path.parent}:/license:ro",
        "-v", "/tmp/dry_run_output:/output",
        fastsurfer_image,
        "run_fastsurfer.sh",
        "--t1", f"/input/{nii_path.name}",
        "--sid", "dry_run_subject",
        "--sd", "/output",
        "--fs_license", f"/license/{lic_path.name}",
        "--device", "cuda",
        "--threads", "1",
        "--allow_root",
        "--seg_only",   # solo segmentazione DNN, più veloce per il profiling
    ]

    try:
        subprocess.run(cmd, check=True, timeout=3600)
    except subprocess.CalledProcessError as e:
        print(f"[DryRun] FastSurfer fallito durante il dry-run: {e}")
        stop_event.set()
        return None
    except subprocess.TimeoutExpired:
        print("[DryRun] Timeout durante il dry-run (>60min)")
        stop_event.set()
        return None
    finally:
        stop_event.set()
        monitor_thread.join(timeout=10)

    if not samples:
        print("[DryRun] Nessun campione VRAM raccolto")
        return None

    # Calcola il delta rispetto alla baseline
    peak_mb = max(samples)
    delta_mb = max(0.0, peak_mb - baseline)
    cost_gb = (delta_mb / 1024) * EMPIRICAL_SAFETY_MARGIN

    print(f"[DryRun] Peak VRAM: {peak_mb:.0f}MB | Baseline: {baseline:.0f}MB")
    print(f"[DryRun] Costo stimato per soggetto: {cost_gb:.2f}GB (con safety margin {EMPIRICAL_SAFETY_MARGIN}x)")

    return cost_gb
