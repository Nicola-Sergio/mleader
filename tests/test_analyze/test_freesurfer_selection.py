"""
Unit tests for freesurfer selection logic in the Analyze phase.

Tests that the module correctly selects freesurfer as brain_segmenter
when GPU is not available, and computes maxForks correctly based on
empirical peak_rss data or hardware-conservative fallback.
"""

import math
import pytest

from orchestrator.monitor.hardware import HardwareProfile, GpuInfo
from orchestrator.monitor.preflight import run_preflight_checks
from orchestrator.analyze.estimator import estimate_params, ExecutionPlan


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_profile(
    cpu_cores: int = 32,
    cpu_threads: int = 32,
    ram_available_gb: float = 65.5,
    ram_total_gb: float = 65.8,
    gpu: GpuInfo = None,
    docker_gpu_runtime: bool = False,
    preflight_passed: bool = True,
    fallbacks: dict = None,
) -> HardwareProfile:
    """Creates a HardwareProfile with sensible defaults for testing."""
    profile = HardwareProfile(
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        cpu_load_percent=1.0,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        disk_free_gb=200.0,
        gpu=gpu,
        nextflow_version="24.10.5",
        docker_gpu_runtime=docker_gpu_runtime,
        fs_license_present=True,
        containers_built=["freesurfer", "fsl", "pyradiomics", "ftd-training"],
        containers_missing=[],
        preflight_passed=preflight_passed,
        fallbacks=fallbacks or {},
    )
    return profile


# ── Test: freesurfer selected when no GPU ────────────────────────────────────

class TestFreesurferSelection:
    """Tests that freesurfer is selected when GPU is not available."""

    def test_no_gpu_selects_freesurfer(self):
        profile = _make_profile(gpu=None, docker_gpu_runtime=False)
        run_preflight_checks(profile)
        plan = estimate_params(profile)

        assert plan.brain_segmenter == "freesurfer"
        assert plan.fastsurfer_device is None
        assert plan.fastsurfer_threads is None

    def test_gpu_present_but_toolkit_missing_selects_freesurfer(self):
        gpu = GpuInfo(
            name="NVIDIA H200-35C",
            vram_total_gb=35.0,
            vram_free_gb=34.2,
            driver_version="525.0",
            compute_capability="9.0",
        )
        profile = _make_profile(gpu=gpu, docker_gpu_runtime=False)
        run_preflight_checks(profile)
        plan = estimate_params(profile)

        assert plan.brain_segmenter == "freesurfer"
        assert plan.fastsurfer_device is None

    def test_gpu_present_but_vram_insufficient_selects_freesurfer(self):
        gpu = GpuInfo(
            name="NVIDIA GTX 1060",
            vram_total_gb=6.0,
            vram_free_gb=5.5,
            driver_version="525.0",
            compute_capability="6.0",
        )
        profile = _make_profile(gpu=gpu, docker_gpu_runtime=True)
        run_preflight_checks(profile)
        plan = estimate_params(profile)

        assert plan.brain_segmenter == "freesurfer"
        assert plan.fastsurfer_device is None

    def test_gpu_present_but_compute_capability_too_low_selects_freesurfer(self):
        gpu = GpuInfo(
            name="NVIDIA GTX 980",
            vram_total_gb=4.0,
            vram_free_gb=3.8,
            driver_version="470.0",
            compute_capability="5.2",
        )
        profile = _make_profile(gpu=gpu, docker_gpu_runtime=True)
        run_preflight_checks(profile)
        plan = estimate_params(profile)

        assert plan.brain_segmenter == "freesurfer"

    def test_no_gpu_fastsurfer_maxforks_is_zero(self):
        """
        When no GPU is available, maxforks_fastsurfer must be 0
        to signal that FastSurfer cannot be used.
        """
        profile = _make_profile(gpu=None, docker_gpu_runtime=False)
        run_preflight_checks(profile)
        plan = estimate_params(profile)

        assert plan.maxforks_fastsurfer == 0


# ── Test: maxForks calculation for freesurfer ─────────────────────────────────

class TestFreesurferMaxForks:
    """Tests maxForks calculation for freesurfer."""

    def test_maxforks_empirical_from_trace(self):
        """
        When empirical peak_rss is available from trace files,
        maxForks = min(floor(ram_available / peak_rss), cpu_cores_free).
        No safety factor is applied — observed max is already worst-case.
        """
        profile = _make_profile(
            gpu=None,
            ram_available_gb=65.5,
            cpu_cores=32,
            fallbacks={"brain_segmenter": "freesurfer"},
        )

        peak_rss = 2.2
        plan = estimate_params(profile, ram_per_subject_gb_freesurfer=peak_rss)

        expected = min(math.floor(65.5 / peak_rss), 32)

        assert plan.brain_segmenter == "freesurfer"
        assert plan.maxforks_freesurfer == expected
        assert plan.source == "trace_empirical"

    def test_maxforks_conservative_fallback_no_trace_data(self):
        """
        When no empirical data is available (cold start),
        maxForks = cpu_cores_free (hardware-conservative fallback).
        """
        profile = _make_profile(
            gpu=None,
            cpu_cores=32,
            ram_available_gb=65.5,
            fallbacks={"brain_segmenter": "freesurfer"},
        )

        plan = estimate_params(profile)

        assert plan.brain_segmenter == "freesurfer"
        assert plan.maxforks_freesurfer == plan.cpu_cores_free
        assert plan.source == "hardware_conservative"

    def test_maxforks_capped_by_cpu_cores(self):
        """
        When RAM would allow more parallelism than available CPU cores,
        maxForks is capped at cpu_cores_free.
        """
        profile = _make_profile(
            gpu=None,
            cpu_cores=4,
            cpu_threads=4,
            ram_available_gb=512.0,
            fallbacks={"brain_segmenter": "freesurfer"},
        )

        plan = estimate_params(profile, ram_per_subject_gb_freesurfer=2.2)

        assert plan.maxforks_freesurfer <= 4
        assert plan.brain_segmenter == "freesurfer"

    def test_maxforks_capped_by_ram(self):
        """
        When RAM is the bottleneck, maxForks is limited by RAM not CPU cores.
        """
        profile = _make_profile(
            gpu=None,
            cpu_cores=64,
            cpu_threads=64,
            ram_available_gb=10.0,
            fallbacks={"brain_segmenter": "freesurfer"},
        )

        plan = estimate_params(profile, ram_per_subject_gb_freesurfer=2.2)

        assert plan.maxforks_freesurfer == math.floor(10.0 / 2.2)
        assert plan.brain_segmenter == "freesurfer"

    def test_pyradiomics_jobs_based_on_cpu_cores_free(self):
        """
        pyradiomics_jobs = max(1, cpu_cores_free - 1).
        With low load and 32 cores, cpu_cores_free = 32, jobs = 31.
        """
        profile = _make_profile(
            gpu=None,
            cpu_cores=32,
            cpu_threads=32,
            fallbacks={"brain_segmenter": "freesurfer"},
        )

        plan = estimate_params(profile)

        assert plan.pyradiomics_jobs == 31

    def test_maxforks_at_least_one(self):
        """
        maxForks is always at least 1, even in extreme resource constraints.
        """
        profile = _make_profile(
            gpu=None,
            cpu_cores=1,
            cpu_threads=1,
            ram_available_gb=1.0,
            fallbacks={"brain_segmenter": "freesurfer"},
        )

        plan = estimate_params(profile, ram_per_subject_gb_freesurfer=2.2)

        assert plan.maxforks_freesurfer >= 1

    def test_both_maxforks_always_computed(self):
        """
        Both maxforks_freesurfer and maxforks_fastsurfer are always present
        in the ExecutionPlan, regardless of which segmenter is selected.
        """
        profile = _make_profile(
            gpu=None,
            fallbacks={"brain_segmenter": "freesurfer"},
        )

        plan = estimate_params(profile)

        assert hasattr(plan, "maxforks_freesurfer")
        assert hasattr(plan, "maxforks_fastsurfer")
        assert isinstance(plan.maxforks_freesurfer, int)
        assert isinstance(plan.maxforks_fastsurfer, int)