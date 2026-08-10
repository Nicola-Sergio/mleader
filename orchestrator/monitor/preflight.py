"""
Preflight checks — Monitor component.
Checks critical and non-critical constraints before launching the pipeline.
Critical constraints: block the launch.
Non-critical constraints: generate warnings and activate fallbacks.
"""

from __future__ import annotations

from pathlib import Path

from .hardware import HardwareProfile


# ── FastSurfer official system requirements ───────────────────────────────────
# Source: https://github.com/Deep-MI/FastSurfer
# Intel or AMD CPU (6 or more cores)
# 16 GB system memory
# NVIDIA GPU (2016 or newer = compute capability >= 6.0 / Pascal architecture)
# 12 GB graphics memory
FASTSURFER_MIN_CORES        = 6
FASTSURFER_MIN_RAM_GB       = 16.0
FASTSURFER_MIN_VRAM_GB      = 12.0
FASTSURFER_MIN_COMPUTE_CAP  = 6.0   # Pascal (2016) and newer

# Minimum free disk space recommended for preprocessing pipeline
DISK_MIN_FREE_GB = 50.0

# Known output directories produced by previous pipeline runs
# Used to provide actionable hints when disk space is low
KNOWN_OUTPUT_DIRS = [
    "data/interim/freesurfer_segmentation",
    "data/interim/fastsurfer_segmentation",
    "data/interim/features-freesurfer",
    "data/interim/features-fastsurfer",
    ".nextflow/runningfiles",
]


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Converts a version string to a tuple of integers for comparison."""
    try:
        parts = version_str.split(".")
        return tuple(int(p) for p in parts[:3])
    except (ValueError, AttributeError):
        return (0, 0, 0)


def run_preflight_checks(profile: HardwareProfile) -> None:
    """
    Runs all preflight checks and populates:
    - profile.preflight_errors   (critical — block the launch)
    - profile.preflight_warnings (non-critical — activate fallbacks)
    - profile.fallbacks          (fallback decisions applied)
    - profile.preflight_passed   (True if no critical errors)
    """
    errors = []
    warnings = []
    fallbacks = {}

    # ── CRITICAL CONSTRAINTS ─────────────────────────────────────────

    # 1. Nextflow installed
    if profile.nextflow_version is None:
        errors.append(
            "Nextflow not found. "
            "Please, install Nextflow"
        )

    # 2. FreeSurfer license present
    if not profile.fs_license_present:
        errors.append(
            "license.txt not found in repo root. "
            "Obtain the FreeSurfer license and place it as license.txt in the project root."
        )

    # 3. Required Docker containers built
    if profile.containers_missing:
        missing_str = ", ".join(profile.containers_missing)
        errors.append(
            f"Missing Docker containers: {missing_str}. "
            "Run 'docker compose build' in the FTD repo root."
        )

    # ── NON-CRITICAL CONSTRAINTS (fallbacks) ─────────────────────────

    # 5. GPU present but nvidia-container-toolkit not installed
    if profile.gpu is not None and not profile.docker_gpu_runtime:
        warnings.append(
            "GPU detected but nvidia-container-toolkit is not installed. "
            "Please, install it."
            "Fallback: brain_segmenter = freesurfer (CPU)."
        )
        fallbacks["brain_segmenter"] = "freesurfer"
        fallbacks["fastsurfer_device"] = None

    # 5b. GPU present, toolkit installed, but vGPU license missing/expired
    if (
        profile.gpu is not None
        and profile.docker_gpu_runtime
        and profile.gpu.is_vgpu
        and profile.gpu.vgpu_license_status
        and "unlicensed" in profile.gpu.vgpu_license_status.lower()
    ):
        warnings.append(
            f"Virtualized GPU (vGPU) detected but unlicensed "
            f"(License Status: {profile.gpu.vgpu_license_status}). "
            "Without a valid license, GPU-enabled containers will fail at runtime "
            "with opaque errors. Check license server configuration or consider "
            "switching to MIG passthrough. "
            "Fallback: brain_segmenter = freesurfer (CPU)."
        )
        fallbacks["brain_segmenter"] = "freesurfer"
        fallbacks["fastsurfer_device"] = None

    # 6. GPU absent — silent fallback, no warning needed
    if profile.gpu is None:
        fallbacks["brain_segmenter"] = "freesurfer"
        fallbacks["fastsurfer_device"] = None

    # 7. FastSurfer official system requirements check
    # Only evaluated if GPU is present, toolkit is available,
    # and no fallback has already been registered (checks 5, 5b, 6).
    if (
        profile.gpu is not None
        and profile.docker_gpu_runtime
        and "brain_segmenter" not in fallbacks
    ):
        fastsurfer_issues = []

        # 7a. VRAM >= 12 GB
        if profile.gpu.vram_free_gb < FASTSURFER_MIN_VRAM_GB:
            fastsurfer_issues.append(
                f"VRAM available ({profile.gpu.vram_free_gb:.1f} GB) "
                f"below FastSurfer minimum ({FASTSURFER_MIN_VRAM_GB} GB)"
            )

        # 7b. System RAM >= 16 GB
        if profile.ram_available_gb < FASTSURFER_MIN_RAM_GB:
            fastsurfer_issues.append(
                f"System RAM available ({profile.ram_available_gb:.1f} GB) "
                f"below FastSurfer minimum ({FASTSURFER_MIN_RAM_GB} GB)"
            )

        # 7c. CPU cores >= 6
        if profile.cpu_cores < FASTSURFER_MIN_CORES:
            fastsurfer_issues.append(
                f"CPU cores ({profile.cpu_cores}) "
                f"below FastSurfer minimum ({FASTSURFER_MIN_CORES})"
            )

        # 7d. GPU compute capability >= 6.0 (NVIDIA Pascal 2016+)
        try:
            cc = float(profile.gpu.compute_capability)
            if cc < FASTSURFER_MIN_COMPUTE_CAP:
                fastsurfer_issues.append(
                    f"GPU compute capability ({cc}) "
                    f"below FastSurfer minimum ({FASTSURFER_MIN_COMPUTE_CAP} — NVIDIA 2016+)"
                )
        except (ValueError, TypeError):
            pass  # compute_capability not available — skip check

        if fastsurfer_issues:
            issues_str = "; ".join(fastsurfer_issues)
            warnings.append(
                f"FastSurfer official requirements not met: {issues_str}. "
                f"Official requirements: {FASTSURFER_MIN_CORES}+ CPU cores, "
                f"{FASTSURFER_MIN_RAM_GB} GB RAM, "
                f"{FASTSURFER_MIN_VRAM_GB} GB VRAM, "
                f"NVIDIA GPU 2016+ (compute capability >= {FASTSURFER_MIN_COMPUTE_CAP}). "
                "Fallback: brain_segmenter = freesurfer (CPU)."
            )
            fallbacks["brain_segmenter"] = "freesurfer"
            fallbacks["fastsurfer_device"] = None

    # 8. Low disk space — check if previous pipeline output is consuming space
    # profile.repo_root is set by the Monitor via probe_hardware(work_dir=repo_root)
    if profile.disk_free_gb < DISK_MIN_FREE_GB:
        repo_root = getattr(profile, 'repo_root', '.')
        existing = [
            str(Path(repo_root) / d)
            for d in KNOWN_OUTPUT_DIRS
            if (Path(repo_root) / d).exists()
        ]

        if existing:
            dirs_str = "\n    ".join(existing)
            warnings.append(
                f"Low disk space ({profile.disk_free_gb:.1f} GB available). "
                f"The following pipeline output directories were found and "
                f"may be consuming significant space:\n    {dirs_str}\n"
                "Consider removing them if no longer needed "
                "(preprocessing will need to be re-run if segmentation is deleted)."
            )
        else:
            warnings.append(
                f"Low disk space ({profile.disk_free_gb:.1f} GB available). "
                "The preprocessing pipeline produces heavy volumetric data. "
                f"At least {DISK_MIN_FREE_GB:.0f} GB free is recommended."
            )

    # ── FINAL RESULT ─────────────────────────────────────────────────
    profile.preflight_errors   = errors
    profile.preflight_warnings = warnings
    profile.fallbacks          = fallbacks
    profile.preflight_passed   = len(errors) == 0