"""
Test per il modulo preflight checks del Monitor.
"""

import pytest
from orchestrator.monitor.hardware import HardwareProfile, GpuInfo
from orchestrator.monitor.preflight import run_preflight_checks


def _base_profile(**kwargs) -> HardwareProfile:
    """Crea un profilo base valido, sovrascrivibile con kwargs."""
    defaults = dict(
        cpu_cores=8,
        cpu_threads=16,
        cpu_load_percent=10.0,
        ram_total_gb=64.0,
        ram_available_gb=48.0,
        disk_free_gb=200.0,
        gpu=GpuInfo(
            name="NVIDIA A100",
            vram_total_gb=80.0,
            vram_free_gb=78.0,
            driver_version="525.0",
            compute_capability="8.0",
        ),
        nextflow_version="24.10.5",
        docker_gpu_runtime=True,
        fs_license_present=True,
        containers_built=["freesurfer", "fsl", "pyradiomics", "ftd-training"],
        containers_missing=[],
    )
    defaults.update(kwargs)
    return HardwareProfile(**defaults)


def test_all_checks_pass():
    """Un profilo completamente valido non deve avere errori né warning."""
    profile = _base_profile()
    run_preflight_checks(profile)
    assert profile.preflight_passed is True
    assert len(profile.preflight_errors) == 0


def test_nextflow_missing_is_critical():
    profile = _base_profile(nextflow_version=None)
    run_preflight_checks(profile)
    assert profile.preflight_passed is False
    assert any("Nextflow" in e for e in profile.preflight_errors)


def test_license_missing_is_critical():
    profile = _base_profile(fs_license_present=False)
    run_preflight_checks(profile)
    assert profile.preflight_passed is False
    assert any("license.txt" in e for e in profile.preflight_errors)


def test_containers_missing_is_critical():
    profile = _base_profile(
        containers_built=["freesurfer"],
        containers_missing=["fsl", "pyradiomics", "ftd-training"],
    )
    run_preflight_checks(profile)
    assert profile.preflight_passed is False
    assert any("fsl" in e for e in profile.preflight_errors)


def test_no_gpu_triggers_freesurfer_fallback():
    profile = _base_profile(gpu=None, docker_gpu_runtime=False)
    run_preflight_checks(profile)
    assert profile.preflight_passed is True
    assert profile.fallbacks.get("brain_segmenter") == "freesurfer"


def test_gpu_present_but_docker_runtime_missing_triggers_fallback():
    profile = _base_profile(docker_gpu_runtime=False)
    run_preflight_checks(profile)
    assert profile.preflight_passed is True
    assert profile.fallbacks.get("brain_segmenter") == "freesurfer"
    assert len(profile.preflight_warnings) > 0


def test_insufficient_vram_triggers_freesurfer_fallback():
    gpu = GpuInfo(
        name="NVIDIA GTX 1060",
        vram_total_gb=6.0,
        vram_free_gb=2.0,  # sotto la soglia minima
        driver_version="525.0",
        compute_capability="6.0",
    )
    profile = _base_profile(gpu=gpu)
    run_preflight_checks(profile)
    assert profile.preflight_passed is True
    assert profile.fallbacks.get("brain_segmenter") == "freesurfer"


def test_fastsurfer_ram_requirement_triggers_freesurfer_fallback():
    profile = _base_profile(ram_available_gb=8.0)  # < 16 GB
    run_preflight_checks(profile)
    assert profile.preflight_passed is True
    assert profile.fallbacks.get("brain_segmenter") == "freesurfer"
    assert any("RAM" in w for w in profile.preflight_warnings)


def test_low_disk_generates_warning():
    profile = _base_profile(disk_free_gb=20.0)
    run_preflight_checks(profile)
    assert profile.preflight_passed is True
    assert any("disk" in w.lower() for w in profile.preflight_warnings)


def test_multiple_critical_errors_all_reported():
    profile = _base_profile(
        nextflow_version=None,
        fs_license_present=False,
        containers_missing=["fsl"],
        containers_built=["freesurfer", "pyradiomics", "ftd-training"],
    )
    run_preflight_checks(profile)
    assert profile.preflight_passed is False
    assert len(profile.preflight_errors) >= 3
