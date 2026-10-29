"""
Pilot-run profiler - Analyze component.
Run FastSurfer on a sample of subjects and measure peak VRAM
consumption during execution.
Inspired by Lotaru's principle: measure empirically instead
of assuming a priori values from the knowledge base.

The sample is drawn with a fixed-seed random selection over the dataset
folders declared in nextflow.config (params.dataset), so the pilot run is
reproducible: the same seed on the same dataset selects the same subjects.
The per-subject VRAM cost reported is the MAXIMUM peak observed across the
sampled subjects (conservative, consistent with the trace-based estimator),
times a safety margin.
"""

from __future__ import annotations

import os
import random
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Sequence, Union


# Additional safety margin on the empirical measurement
EMPIRICAL_SAFETY_MARGIN = 1.15

# Defaults for the pilot sample
DEFAULT_PILOT_N = 10
DEFAULT_PILOT_SEED = 42

# Host folder mounted as FastSurfer output during the pilot run
PILOT_OUTPUT_DIR = Path("/tmp/pilot_run_output")

# NIfTI extensions recognised when building the sampling pool
_NII_GLOBS = ("*.nii", "*.nii.gz")


# Where params.dataset may be declared, relative to the repo root
_CONFIG_CANDIDATES = ("nextflow.config", "nextflow/nextflow.config")


def read_dataset_dirs(repo_root: Union[str, Path] = ".") -> list[str]:
    """
    Read ``params.dataset`` (a Groovy list of folders) from the repo's
    nextflow.config, looked up at <repo_root>/nextflow.config and then at
    <repo_root>/nextflow/nextflow.config.

    Returns the folders as written in the file (relative paths stay relative),
    or an empty list when no file declares the parameter.
    """
    for rel in _CONFIG_CANDIDATES:
        try:
            text = (Path(repo_root) / rel).read_text(errors="replace")
        except OSError:
            continue
        m = re.search(r"params\.dataset\s*=\s*\[(.*?)\]", text, re.DOTALL)
        if m:
            return re.findall(r"""["']([^"']+)["']""", m.group(1))
    return []


def sample_subjects(
    dataset_dirs: Sequence[Union[str, Path]],
    n: int = DEFAULT_PILOT_N,
    seed: int = DEFAULT_PILOT_SEED,
    repo_root: Union[str, Path] = ".",
) -> list[Path]:
    """
    Select ``n`` subject images at random (fixed seed) from the dataset folders.

    The pool is the union of all NIfTI files under ``dataset_dirs`` (each
    resolved against ``repo_root`` when relative). The pool is sorted before
    sampling so that, for a given seed and dataset, the selection is fully
    deterministic and reproducible. If fewer than ``n`` images exist, all of
    them are returned.
    """
    root = Path(repo_root)
    pool: list[Path] = []
    for d in dataset_dirs:
        d_path = Path(d)
        if not d_path.is_absolute():
            d_path = root / d_path
        if not d_path.is_dir():
            continue
        for pattern in _NII_GLOBS:
            pool.extend(d_path.glob(pattern))

    # Deterministic pool: unique, sorted by resolved path.
    pool = sorted({p.resolve() for p in pool})

    if n <= 0 or not pool:
        return []
    if len(pool) <= n:
        return pool

    rng = random.Random(seed)
    selected = rng.sample(pool, n)
    return sorted(selected)


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


def _profile_one(
    nii_path: Path,
    lic_path: Path,
    fastsurfer_image: str,
    sample_interval_s: float,
) -> Optional[float]:
    """
    Run FastSurfer --seg_only on a single subject and return the peak VRAM
    delta over baseline in GB (WITHOUT the safety margin), or None on failure.
    """
    print(f"[PilotRun] Profiling VRAM on subject: {nii_path.name}")

    baseline = _sample_vram_usage_mb() or 0.0

    samples: list[float] = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=_monitor_vram,
        args=(stop_event, sample_interval_s, samples),
        daemon=True,
    )
    monitor_thread.start()

    # The output folder must exist and belong to the user running the container;
    # otherwise Docker creates it as root and a non-root --user cannot write to it.
    PILOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        # Same as the Nextflow process: run as the calling user. Without it the
        # image falls back to its built-in "nonroot" user, which is missing from
        # the image's passwd file ("unable to find user nonroot").
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--entrypoint", "",
        "-v", f"{nii_path.parent}:/input:ro",
        "-v", f"{lic_path.parent}:/license:ro",
        "-v", f"{PILOT_OUTPUT_DIR}:/output",
        fastsurfer_image,
        "run_fastsurfer.sh",
        "--t1", f"/input/{nii_path.name}",
        "--sid", f"pilot_{nii_path.stem}",
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
        print(f"[PilotRun] FastSurfer failed on {nii_path.name}: {e}")
        return None
    except subprocess.TimeoutExpired:
        print(f"[PilotRun] Timeout (>60min) on {nii_path.name}")
        return None
    finally:
        stop_event.set()
        monitor_thread.join(timeout=10)

    if not samples:
        print(f"[PilotRun] No VRAM samples collected for {nii_path.name}")
        return None

    peak_mb = max(samples)
    delta_mb = max(0.0, peak_mb - baseline)
    delta_gb = delta_mb / 1024
    print(f"[PilotRun] {nii_path.name}: peak {peak_mb:.0f}MB, baseline {baseline:.0f}MB "
          f"-> {delta_gb:.2f}GB")
    return delta_gb


def profile_fastsurfer_vram(
    sample_nii: Union[str, Path, Sequence[Union[str, Path]]],
    license_path: str,
    fastsurfer_image: str = "deepmi/fastsurfer:cuda-v2.4.2",
    sample_interval_s: float = 5.0,
) -> Optional[float]:
    """
    Run FastSurfer on one or more sample subjects and return the estimated
    VRAM cost per subject in GB (with safety margin), or None if every pilot
    run failed.

    The reported cost is ``max(per-subject peak delta) * EMPIRICAL_SAFETY_MARGIN``.
    A single path is accepted for backward compatibility.
    """
    if isinstance(sample_nii, (str, Path)):
        sample_nii = [sample_nii]
    niis = [Path(p).resolve() for p in sample_nii]

    lic_path = Path(license_path).resolve()
    if not lic_path.exists():
        print(f"[PilotRun] License not found: {lic_path}")
        return None

    niis = [p for p in niis if p.exists()]
    if not niis:
        print("[PilotRun] No valid sample subjects found")
        return None

    print(f"[PilotRun] Profiling VRAM on {len(niis)} subject(s) with --seg_only.")

    costs: list[float] = []
    for nii_path in niis:
        delta_gb = _profile_one(nii_path, lic_path, fastsurfer_image, sample_interval_s)
        if delta_gb is not None:
            costs.append(delta_gb)

    if not costs:
        print("[PilotRun] No VRAM samples collected on any subject")
        return None

    cost_gb = max(costs) * EMPIRICAL_SAFETY_MARGIN
    print(f"[PilotRun] Per-subject VRAM (max of {len(costs)} subjects): "
          f"{max(costs):.2f}GB -> {cost_gb:.2f}GB with margin {EMPIRICAL_SAFETY_MARGIN}x")
    return cost_gb


def run_pilot(
    dataset_dirs: Sequence[Union[str, Path]],
    license_path: str,
    repo_root: Union[str, Path] = ".",
    n: int = DEFAULT_PILOT_N,
    seed: int = DEFAULT_PILOT_SEED,
    fastsurfer_image: str = "deepmi/fastsurfer:cuda-v2.4.2",
    sample_interval_s: float = 5.0,
) -> Optional[float]:
    """
    Entry point for the Analyze phase: sample ``n`` subjects from
    ``dataset_dirs`` (fixed seed) and return the estimated per-subject VRAM
    cost in GB, or None if the pilot could not run.
    """
    subjects = sample_subjects(dataset_dirs, n=n, seed=seed, repo_root=repo_root)
    if not subjects:
        print(f"[PilotRun] No subjects found under: {list(dataset_dirs)}")
        return None
    print(f"[PilotRun] Sampled {len(subjects)} subject(s) with seed={seed}")
    return profile_fastsurfer_vram(
        subjects,
        license_path=license_path,
        fastsurfer_image=fastsurfer_image,
        sample_interval_s=sample_interval_s,
    )