"""
Dry-run profiler — Analyze component.
Run FastSurfer on a single sample subject and measure
VRAM consumption during execution.
Inspired by Lotaru's principle: measure empirically instead
of assuming a priori values ​​from the knowledge base.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Optional


# Additional safety margin on the empirical measurement
EMPIRICAL_SAFETY_MARGIN = 1.15


def _sample_vram_usage_mb() -> Optional[float]:
    """Sample the VRAM usage by the GPU at the current moment (MB)."""
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
    """Thread that samples the VRAM every interval_s seconds."""
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
    Run FastSurfer on a single sample subject and measure
    the peak VRAM consumption during execution.

    Returns the estimated cost in GB per subject (including a safety margin),
    or None if the dry-run fails.

    Parameters:
        sample_nii: path to the .nii file of the sample subject
        license_path: path to the license.txt file of FreeSurfer
        fastsurfer_image: tag of the Docker image for FastSurfer
        sample_interval_s: sampling interval for VRAM in seconds
    """
    nii_path = Path(sample_nii).resolve()
    lic_path = Path(license_path).resolve()

    if not nii_path.exists():
        print(f"[DryRun] Sample file not found: {nii_path}")
        return None

    if not lic_path.exists():
        print(f"[DryRun] License not found: {lic_path}")
        return None

    print(f"[DryRun] Profiling VRAM on sample subject: {nii_path.name}")
    print("[DryRun] This will take 15-30 minutes...")

    # Measure VRAM baseline before launching
    baseline = _sample_vram_usage_mb() or 0.0

    # Start the VRAM monitoring thread
    samples: list[float] = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=_monitor_vram,
        args=(stop_event, sample_interval_s, samples),
        daemon=True,
    )
    monitor_thread.start()

    # Run FastSurfer on 1 subject
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
        "--seg_only",   # DNN segmentation only, faster for profiling
    ]

    try:
        subprocess.run(cmd, check=True, timeout=3600)
    except subprocess.CalledProcessError as e:
        print(f"[DryRun] FastSurfer failed during the dry-run: {e}")
        stop_event.set()
        return None
    except subprocess.TimeoutExpired:
        print("[DryRun] Timeout during the dry-run (>60min)")
        stop_event.set()
        return None
    finally:
        stop_event.set()
        monitor_thread.join(timeout=10)

    if not samples:
        print("[DryRun] No VRAM samples collected")
        return None

    # Calcola il delta rispetto alla baseline
    peak_mb = max(samples)
    delta_mb = max(0.0, peak_mb - baseline)
    cost_gb = (delta_mb / 1024) * EMPIRICAL_SAFETY_MARGIN

    print(f"[DryRun] Peak VRAM: {peak_mb:.0f}MB | Baseline: {baseline:.0f}MB")
    print(f"[DryRun] Estimated cost per subject: {cost_gb:.2f}GB (with safety margin {EMPIRICAL_SAFETY_MARGIN}x)")

    return cost_gb
