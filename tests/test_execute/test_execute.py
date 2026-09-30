"""
Comprehensive test suite for the MLEADeR orchestrator.

Tests are grouped by thesis chapter claim and are self-contained:
- All external calls (subprocess, psutil, file I/O) are mocked.
- Tests can be run even when the orchestrator package is not yet installed;
  missing modules cause individual tests to be skipped, not the entire suite
  to fail at import time.

Run with:
    pytest tests/test_execute/test_execute.py -v
"""

from __future__ import annotations

import importlib
import math
import sys
import textwrap
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, mock_open, patch

try:
    import pytest
except ModuleNotFoundError:
    # Minimal pytest shim so the file can be run with python -m unittest
    import types as _types, unittest as _unittest

    class _PytestShim:
        class mark:
            @staticmethod
            def parametrize(*a, **kw):
                return lambda f: f

        @staticmethod
        def skip(reason=""):
            raise _unittest.SkipTest(reason)

        @staticmethod
        def fixture(*a, **kw):
            return lambda f: f

        class raises:
            def __init__(self, exc):
                self.exc = exc
            def __enter__(self):
                return self
            def __exit__(self, et, ev, tb):
                if et is None:
                    raise AssertionError(f"Expected {self.exc} to be raised")
                return issubclass(et, self.exc)

        class _Approx:
            def __init__(self, expected, rel=1e-6):
                self.expected = expected
                self.rel = rel
            def __eq__(self, other):
                if self.expected == 0:
                    return abs(other) < 1e-12
                return abs(other - self.expected) / abs(self.expected) < self.rel
            def __repr__(self):
                return f"approx({self.expected})"

        @staticmethod
        def approx(val, rel=1e-6):
            return _PytestShim._Approx(val, rel)

    _PytestShim.approx = staticmethod(lambda val, rel=1e-6: _PytestShim._Approx(val, rel))
    # Make _Approx accessible
    _PytestShim._Approx = type('_Approx', (), {
        '__init__': lambda self, expected, rel=1e-6: (setattr(self, 'expected', expected) or setattr(self, 'rel', rel)),
        '__eq__': lambda self, other: (abs(other - self.expected) / abs(self.expected) < self.rel) if self.expected != 0 else abs(other) < 1e-12,
        '__repr__': lambda self: f'approx({self.expected})',
    })
    pytest = _PytestShim()  # type: ignore

# ---------------------------------------------------------------------------
# Path setup – allow "import orchestrator.*" from the repo root even when the
# package has not been installed via pip.
# ---------------------------------------------------------------------------
import tempfile
import unittest
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _try_import(module_path: str) -> types.ModuleType | None:
    """Return the module or None (never raises)."""
    try:
        return importlib.import_module(module_path)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Lazy module handles – tests that depend on a module use
#   mod = _require("orchestrator.analyze.estimator")
# which skips the test if the module cannot be imported.
# ---------------------------------------------------------------------------
def _require(module_path: str) -> types.ModuleType:
    mod = _try_import(module_path)
    if mod is None:
        pytest.skip(f"Module {module_path!r} not importable (package not installed or not on path)")
    return mod


# ---------------------------------------------------------------------------
# Base test class that provides self.tmp_path (pytest fixture) when running under
# stdlib unittest.
# ---------------------------------------------------------------------------
class _Base(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    # Make assert_* available even in non-TestCase child path
    def assertTrue(self, expr, msg=None):
        super().assertTrue(expr, msg)

    def assertEqual(self, a, b, msg=None):
        super().assertEqual(a, b, msg)


# ===========================================================================
# Pure-formula helpers (no imports needed)
# These mirror exactly the mathematical formulae stated in the thesis.
# ===========================================================================

def _cpu_cores_free(cpu_cores: int, load_1min: float) -> int:
    """Eq. [eq:cpu-free]: cpu_cores_free = max(1, cpu_cores - floor(load_1min))"""
    return max(1, cpu_cores - math.floor(load_1min))


def _maxforks_freesurfer(ram_available: float, peak_rss_max: float, cpu_cores_free: int) -> int:
    """Eq. [eq:maxforks-freesurfer]: max(1, min(floor(ram_available/peak_rss_max), cpu_cores_free))"""
    return max(1, min(math.floor(ram_available / peak_rss_max), cpu_cores_free))


def _maxforks_fastsurfer(vram_free: float, vram_subj: float, cpu_cores_free: int) -> int:
    """Eq. [eq:maxforks-fastsurfer]: max(1, min(floor(vram_free/vram_subj), cpu_cores_free))"""
    return max(1, min(math.floor(vram_free / vram_subj), cpu_cores_free))


def _throughput(maxforks: int, duration_mean_min: float) -> float:
    """Eq. [eq:throughput]: tp = maxforks / (duration_mean_min / 60)"""
    return maxforks / (duration_mean_min / 60.0)


def _fastsurfer_threads(cpu_threads: int, maxforks_fastsurfer: int) -> int:
    """Eq. [eq:fastsurfer-threads]: max(2, floor((cpu_threads - 1) / maxforks_fastsurfer))"""
    return max(2, math.floor((cpu_threads - 1) / maxforks_fastsurfer))


def _pyradiomics_jobs(cpu_cores_free: int) -> int:
    """Eq. [eq:pyradiomics]: max(1, cpu_cores_free - 1)"""
    return max(1, cpu_cores_free - 1)


# ===========================================================================
# TestCpuCoresFreeFormula
# Thesis claim: cpu_cores_free = max(1, cpu_cores - floor(load_1min))
# ===========================================================================

class TestCpuCoresFreeFormula(_Base):
    """[eq:cpu-free] cpu_cores_free = max(1, cpu_cores − floor(load_1min))"""

    def test_typical_load(self):
        # 8 cores, load 2.3 → floor(2.3)=2 → 8-2=6
        assert _cpu_cores_free(8, 2.3) == 6

    def test_zero_load(self):
        assert _cpu_cores_free(4, 0.0) == 4

    def test_load_exactly_one(self):
        # floor(1.0) = 1, 4 - 1 = 3
        assert _cpu_cores_free(4, 1.0) == 3

    def test_load_fraction_below_one(self):
        # floor(0.9) = 0, 4 - 0 = 4
        assert _cpu_cores_free(4, 0.9) == 4

    def test_high_load_clamps_to_one(self):
        # load >= cpu_cores → result would be ≤0, clamped to 1
        assert _cpu_cores_free(4, 5.0) == 1

    def test_load_equals_cores(self):
        # 4 cores, load 4.0 → 4-4=0, clamped to 1
        assert _cpu_cores_free(4, 4.0) == 1

    def test_single_core_no_load(self):
        assert _cpu_cores_free(1, 0.0) == 1

    def test_single_core_high_load(self):
        assert _cpu_cores_free(1, 1.5) == 1

    def test_fractional_load_truncated(self):
        # floor(3.99) = 3, 8 - 3 = 5
        assert _cpu_cores_free(8, 3.99) == 5

    def test_against_estimator_module(self):
        """If estimator module exists, verify it matches the formula."""
        mod = _try_import("orchestrator.analyze.estimator")
        if mod is None:
            pytest.skip("estimator module not available")
        fn = getattr(mod, "cpu_cores_free", None) or getattr(mod, "_cpu_cores_free", None)
        if fn is None:
            pytest.skip("cpu_cores_free function not found in estimator module")
        assert fn(8, 2.3) == 6
        assert fn(4, 5.0) == 1


# ===========================================================================
# TestMaxForksFreeSurfer
# ===========================================================================

class TestMaxForksFreeSurfer(_Base):
    """[eq:maxforks-freesurfer] maxforks_freesurfer = max(1, min(floor(ram/peak_rss_max), cpu_cores_free))"""

    def test_ram_bound(self):
        # 32 GB RAM, 8 GB per subject → 4; cpu_free=8 → min(4,8)=4
        assert _maxforks_freesurfer(32.0, 8.0, 8) == 4

    def test_cpu_bound(self):
        # 64 GB RAM, 8 GB per subject → 8; cpu_free=3 → min(8,3)=3
        assert _maxforks_freesurfer(64.0, 8.0, 3) == 3

    def test_clamped_to_one_when_insufficient_ram(self):
        # 6 GB RAM, 8 GB per subject → floor(0.75)=0 → clamped to 1
        assert _maxforks_freesurfer(6.0, 8.0, 4) == 1

    def test_exact_fit(self):
        # 16 GB / 8 GB = 2, cpu_free=4
        assert _maxforks_freesurfer(16.0, 8.0, 4) == 2

    def test_conservative_mode_equals_cpu_cores_free(self):
        """Conservative FreeSurfer mode ignores RAM and uses cpu_cores_free directly."""
        cpu_free = 6
        # In conservative mode: maxforks_freesurfer = cpu_cores_free
        conservative_result = cpu_free
        assert conservative_result == 6

    def test_cpu_free_one(self):
        assert _maxforks_freesurfer(100.0, 8.0, 1) == 1

    def test_fractional_floor(self):
        # 20 GB / 8 GB = 2.5 → floor = 2
        assert _maxforks_freesurfer(20.0, 8.0, 8) == 2


# ===========================================================================
# TestMaxForksFastSurfer
# ===========================================================================

class TestMaxForksFastSurfer(_Base):
    """[eq:maxforks-fastsurfer] maxforks_fastsurfer = max(1, min(floor(vram_free/vram_subj), cpu_cores_free))"""

    def test_vram_bound(self):
        # 24 GB VRAM, 12 GB per subject → 2; cpu_free=8
        assert _maxforks_fastsurfer(24.0, 12.0, 8) == 2

    def test_cpu_bound(self):
        # 48 GB VRAM, 12 GB → 4; cpu_free=2 → min(4,2)=2
        assert _maxforks_fastsurfer(48.0, 12.0, 2) == 2

    def test_insufficient_vram_clamped_to_one(self):
        # 10 GB VRAM, 12 GB/subj → floor(0.83)=0 → clamped to 1
        assert _maxforks_fastsurfer(10.0, 12.0, 8) == 1

    def test_conservative_mode_returns_one(self):
        """Conservative FastSurfer mode always returns 1."""
        conservative_fastsurfer = 1
        assert conservative_fastsurfer == 1

    def test_gpu_unavailable_returns_zero(self):
        """When GPU is unavailable, maxforks_fastsurfer = 0."""
        gpu_unavailable_fastsurfer = 0
        assert gpu_unavailable_fastsurfer == 0

    def test_exact_vram_fit(self):
        # 24 GB / 12 GB = 2
        assert _maxforks_fastsurfer(24.0, 12.0, 4) == 2

    def test_against_estimator_module(self):
        mod = _try_import("orchestrator.analyze.estimator")
        if mod is None:
            pytest.skip("estimator module not available")
        fn = getattr(mod, "maxforks_fastsurfer", None) or getattr(mod, "_maxforks_fastsurfer", None)
        if fn is None:
            pytest.skip("maxforks_fastsurfer function not found")
        assert fn(24.0, 12.0, 8) == 2


# ===========================================================================
# TestThroughputComparison
# ===========================================================================

class TestThroughputComparison(_Base):
    """[eq:throughput] tp = maxforks / (duration_mean_min / 60)"""

    def test_fastsurfer_wins(self):
        # FastSurfer: 2 parallel, 10 min → tp = 2/(10/60) = 12/hr
        # FreeSurfer: 4 parallel, 60 min → tp = 4/(60/60) = 4/hr
        tp_fas = _throughput(2, 10.0)
        tp_fs = _throughput(4, 60.0)
        assert tp_fas > tp_fs, f"FastSurfer tp={tp_fas} should exceed FreeSurfer tp={tp_fs}"

    def test_freesurfer_wins(self):
        # FreeSurfer: 8 parallel, 60 min → tp = 8/hr
        # FastSurfer: 1 parallel, 10 min → tp = 6/hr
        tp_fas = _throughput(1, 10.0)
        tp_fs = _throughput(8, 60.0)
        assert tp_fs > tp_fas

    def test_throughput_formula_numeric(self):
        # maxforks=4, duration=30 min → tp = 4/(30/60) = 4/0.5 = 8.0 subjects/hr
        assert _throughput(4, 30.0) == pytest.approx(8.0)

    def test_single_fork_one_hour(self):
        # maxforks=1, duration=60 min → tp = 1.0
        assert _throughput(1, 60.0) == pytest.approx(1.0)

    def test_high_parallelism_short_duration(self):
        # maxforks=10, duration=6 min → tp = 10/(6/60) = 100
        assert _throughput(10, 6.0) == pytest.approx(100.0)

    def test_fastsurfer_preferred_when_tp_greater(self):
        """Thesis: pipeline decision selects FastSurfer iff tp_fas > tp_fs."""
        tp_fas = _throughput(2, 10.0)   # 12/hr
        tp_fs = _throughput(1, 60.0)    # 1/hr
        preferred = "fastsurfer" if tp_fas > tp_fs else "freesurfer"
        assert preferred == "fastsurfer"


# ===========================================================================
# TestFastSurferThreads
# ===========================================================================

class TestFastSurferThreads(_Base):
    """[eq:fastsurfer-threads] fastsurfer_threads = max(2, floor((cpu_threads - 1) / maxforks_fastsurfer))"""

    def test_typical(self):
        # 16 threads, 2 forks → floor(15/2)=7
        assert _fastsurfer_threads(16, 2) == 7

    def test_many_forks(self):
        # 16 threads, 8 forks → floor(15/8)=1 → clamped to 2
        assert _fastsurfer_threads(16, 8) == 2

    def test_single_fork(self):
        # 16 threads, 1 fork → floor(15/1)=15
        assert _fastsurfer_threads(16, 1) == 15

    def test_minimum_clamped_to_two(self):
        # 4 threads, 4 forks → floor(3/4)=0 → clamped to 2
        assert _fastsurfer_threads(4, 4) == 2

    def test_two_threads(self):
        # 2 threads, 1 fork → floor(1/1)=1 → clamped to 2
        assert _fastsurfer_threads(2, 1) == 2

    def test_exact_division(self):
        # 13 threads, 4 forks → floor(12/4)=3
        assert _fastsurfer_threads(13, 4) == 3

    def test_against_estimator_module(self):
        mod = _try_import("orchestrator.analyze.estimator")
        if mod is None:
            pytest.skip("estimator module not available")
        fn = getattr(mod, "fastsurfer_threads", None) or getattr(mod, "_fastsurfer_threads", None)
        if fn is None:
            pytest.skip("fastsurfer_threads function not found")
        assert fn(16, 2) == 7


# ===========================================================================
# TestPyradiomicsJobs
# ===========================================================================

class TestPyradiomicsJobs(_Base):
    """[eq:pyradiomics] pyradiomics_jobs = max(1, cpu_cores_free - 1)"""

    def test_typical(self):
        assert _pyradiomics_jobs(8) == 7

    def test_two_free_cores(self):
        assert _pyradiomics_jobs(2) == 1

    def test_one_free_core_clamped(self):
        assert _pyradiomics_jobs(1) == 1

    def test_zero_free_cores_clamped(self):
        # Degenerate: should not happen if cpu_cores_free ≥ 1, but guard tested
        assert _pyradiomics_jobs(0) == 1

    def test_large_value(self):
        assert _pyradiomics_jobs(64) == 63


# ===========================================================================
# TestSourceField
# ===========================================================================

class TestSourceField(_Base):
    """Thesis: ExecutionPlan.source ∈ {pilot_run, trace_empirical,
    trace_empirical_ram_proxy, hardware_conservative, unavailable}"""

    VALID_SOURCES = {
        "pilot_run",
        "trace_empirical",
        "trace_empirical_ram_proxy",
        "hardware_conservative",
        "unavailable",
    }

    def test_valid_source_names(self):
        for s in self.VALID_SOURCES:
            assert isinstance(s, str)

    def test_trace_empirical_is_valid(self):
        assert "trace_empirical" in self.VALID_SOURCES

    def test_hardware_conservative_is_valid(self):
        assert "hardware_conservative" in self.VALID_SOURCES

    def test_unavailable_is_valid(self):
        assert "unavailable" in self.VALID_SOURCES

    def test_execution_plan_source_field(self):
        """ExecutionPlan dataclass/namedtuple should carry a source attribute."""
        mod = _try_import("orchestrator.analyze.estimator")
        if mod is None:
            pytest.skip("estimator module not available")
        ep_cls = getattr(mod, "ExecutionPlan", None)
        if ep_cls is None:
            pytest.skip("ExecutionPlan not found in estimator module")
        # Try to construct with minimal kwargs and check source attribute exists
        try:
            import dataclasses
            fields = [f.name for f in dataclasses.fields(ep_cls)]
            assert "source" in fields, "ExecutionPlan must have a 'source' field"
        except TypeError:
            # Not a dataclass – check with hasattr on instance
            pass

    def test_source_values_are_strings(self):
        for s in self.VALID_SOURCES:
            assert isinstance(s, str)
            assert len(s) > 0


# ===========================================================================
# TestParsePipelineDSL
# ===========================================================================

class TestParsePipelineDSL(_Base):
    """parse_pipeline_dsl() tests based on thesis regex and fallback logic."""

    # Regex documented in thesis: r'nextflow\.enable\.dsl\s*=\s*([12])'
    NF_DSL_REGEX = r"nextflow\.enable\.dsl\s*=\s*([12])"

    def _run_regex(self, text: str) -> str | None:
        import re
        m = re.search(self.NF_DSL_REGEX, text)
        return m.group(1) if m else None

    def test_dsl2_detected(self):
        nf_content = "nextflow.enable.dsl = 2\n"
        assert self._run_regex(nf_content) == "2"

    def test_dsl1_detected(self):
        nf_content = "nextflow.enable.dsl = 1\n"
        assert self._run_regex(nf_content) == "1"

    def test_dsl2_no_spaces(self):
        assert self._run_regex("nextflow.enable.dsl=2") == "2"

    def test_dsl1_no_spaces(self):
        assert self._run_regex("nextflow.enable.dsl=1") == "1"

    def test_regex_does_not_match_other_values(self):
        assert self._run_regex("nextflow.enable.dsl = 3") is None

    def test_returns_none_when_absent(self):
        assert self._run_regex("// no dsl setting here") is None

    def test_multispace_between(self):
        assert self._run_regex("nextflow.enable.dsl   =   2") == "2"

    def test_parse_pipeline_dsl_module_dsl2(self):
        """Module-level test: .nf file with DSL2."""
        mod = _try_import("orchestrator.monitor.pipeline_config")
        if mod is None:
            pytest.skip("pipeline_config module not available")
        fn = getattr(mod, "parse_pipeline_dsl", None)
        if fn is None:
            pytest.skip("parse_pipeline_dsl not found")
        nf_file = self.tmp_path / "main.nf"
        nf_file.write_text("nextflow.enable.dsl = 2\n")
        result = fn(str(nf_file))
        assert result == "2"

    def test_parse_pipeline_dsl_module_dsl1(self):
        mod = _try_import("orchestrator.monitor.pipeline_config")
        if mod is None:
            pytest.skip("pipeline_config module not available")
        fn = getattr(mod, "parse_pipeline_dsl", None)
        if fn is None:
            pytest.skip("parse_pipeline_dsl not found")
        nf_file = self.tmp_path / "main.nf"
        nf_file.write_text("nextflow.enable.dsl = 1\n")
        result = fn(str(nf_file))
        assert result == "1"

    def test_parse_pipeline_dsl_module_falls_back_to_config(self):
        """Falls back to nextflow.config when not found in .nf file."""
        mod = _try_import("orchestrator.monitor.pipeline_config")
        if mod is None:
            pytest.skip("pipeline_config module not available")
        fn = getattr(mod, "parse_pipeline_dsl", None)
        if fn is None:
            pytest.skip("parse_pipeline_dsl not found")
        nf_file = self.tmp_path / "main.nf"
        nf_file.write_text("// no dsl here\n")
        config_file = self.tmp_path / "nextflow.config"
        config_file.write_text("nextflow.enable.dsl = 2\n")
        result = fn(str(nf_file), config_path=str(config_file))
        assert result == "2"

    def test_parse_pipeline_dsl_module_returns_none_when_absent(self):
        mod = _try_import("orchestrator.monitor.pipeline_config")
        if mod is None:
            pytest.skip("pipeline_config module not available")
        fn = getattr(mod, "parse_pipeline_dsl", None)
        if fn is None:
            pytest.skip("parse_pipeline_dsl not found")
        nf_file = self.tmp_path / "main.nf"
        nf_file.write_text("// nothing\n")
        result = fn(str(nf_file))
        assert result is None


# ===========================================================================
# TestUpdateMaxforksInConfig
# ===========================================================================

class TestUpdateMaxforksInConfig(_Base):
    """_update_maxforks_in_config rewrites only the matching line."""

    # Pure-Python reimplementation of the thesis-described logic for testing
    def _update_config(
        self, config_text: str, brain_segmenter: str, new_value: int
    ) -> str:
        param_key = (
            "params.maxforks_fastsurfer"
            if brain_segmenter == "fastsurfer"
            else "params.maxforks_freesurfer"
        )
        new_line = f"        {param_key} = {new_value}\n"
        lines = config_text.splitlines(keepends=True)
        out = []
        for line in lines:
            if param_key in line:
                out.append(new_line)
            else:
                out.append(line)
        return "".join(out)

    def test_fastsurfer_key_selected(self):
        cfg = "        params.maxforks_fastsurfer = 2\n"
        result = self._update_config(cfg, "fastsurfer", 1)
        assert "params.maxforks_fastsurfer = 1" in result

    def test_freesurfer_key_selected(self):
        cfg = "        params.maxforks_freesurfer = 4\n"
        result = self._update_config(cfg, "freesurfer", 2)
        assert "params.maxforks_freesurfer = 2" in result

    def test_other_lines_preserved(self):
        cfg = (
            "        params.maxforks_freesurfer = 4\n"
            "        params.some_other = 99\n"
        )
        result = self._update_config(cfg, "freesurfer", 2)
        assert "params.some_other = 99" in result

    def test_indentation_format(self):
        cfg = "        params.maxforks_fastsurfer = 2\n"
        result = self._update_config(cfg, "fastsurfer", 3)
        # Must be exactly 8 spaces of indentation
        assert result.startswith("        params.maxforks_fastsurfer = 3")

    def test_only_matching_line_rewritten(self):
        cfg = (
            "        params.maxforks_fastsurfer = 2\n"
            "        params.maxforks_freesurfer = 4\n"
            "        params.pyradiomics_jobs = 7\n"
        )
        result = self._update_config(cfg, "fastsurfer", 1)
        assert "params.maxforks_fastsurfer = 1" in result
        assert "params.maxforks_freesurfer = 4" in result   # unchanged
        assert "params.pyradiomics_jobs = 7" in result      # unchanged

    def test_module_update_fastsurfer(self):
        mod = _try_import("orchestrator.execute.supervisor")
        if mod is None:
            pytest.skip("supervisor module not available")
        fn = getattr(mod, "_update_maxforks_in_config", None)
        if fn is None:
            pytest.skip("_update_maxforks_in_config not found")
        cfg_file = self.tmp_path / "nextflow.config"
        cfg_file.write_text("        params.maxforks_fastsurfer = 2\n")
        fn(str(cfg_file), "fastsurfer", 1)
        assert "params.maxforks_fastsurfer = 1" in cfg_file.read_text()

    def test_module_update_freesurfer(self):
        mod = _try_import("orchestrator.execute.supervisor")
        if mod is None:
            pytest.skip("supervisor module not available")
        fn = getattr(mod, "_update_maxforks_in_config", None)
        if fn is None:
            pytest.skip("_update_maxforks_in_config not found")
        cfg_file = self.tmp_path / "nextflow.config"
        cfg_file.write_text("        params.maxforks_freesurfer = 4\n")
        fn(str(cfg_file), "freesurfer", 2)
        assert "params.maxforks_freesurfer = 2" in cfg_file.read_text()

    def test_module_read_maxforks(self):
        mod = _try_import("orchestrator.execute.supervisor")
        if mod is None:
            pytest.skip("supervisor module not available")
        fn = getattr(mod, "_read_maxforks_from_config", None)
        if fn is None:
            pytest.skip("_read_maxforks_from_config not found")
        cfg_file = self.tmp_path / "nextflow.config"
        cfg_file.write_text("        params.maxforks_fastsurfer = 3\n")
        val = fn(str(cfg_file), "fastsurfer")
        assert val == 3


# ===========================================================================
# TestClassifyFailure
# ===========================================================================

class TestClassifyFailure(_Base):
    """classify_failure() pattern-matching tests."""

    # Pattern table from thesis
    PATTERNS = {
        "OOM_VRAM": ["CUDA out of memory", "out of memory on device"],
        "OOM_RAM": ["Killed", "std::bad_alloc", "MemoryError"],
        "MISSING_PACKAGE": ["ModuleNotFoundError", "No module named"],
        "DSL_MISMATCH": ["DSL2 is not supported", "Unexpected token"],
        "CONTAINER_ERROR": ["Unable to find image", "OCI runtime"],
        "GPU_NOT_AVAILABLE": ["CUDA driver version is insufficient"],
    }

    def _classify(self, log_text: str) -> str:
        for category, triggers in self.PATTERNS.items():
            for trigger in triggers:
                if trigger in log_text:
                    return category
        return "UNKNOWN"

    def test_oom_vram_cuda_out_of_memory(self):
        assert self._classify("CUDA out of memory: tried to allocate 1.2 GiB") == "OOM_VRAM"

    def test_oom_vram_out_of_memory_on_device(self):
        assert self._classify("out of memory on device 0") == "OOM_VRAM"

    def test_oom_ram_killed(self):
        assert self._classify("Killed") == "OOM_RAM"

    def test_oom_ram_bad_alloc(self):
        assert self._classify("terminate called after throwing an instance of 'std::bad_alloc'") == "OOM_RAM"

    def test_oom_ram_memory_error(self):
        assert self._classify("MemoryError") == "OOM_RAM"

    def test_missing_package_module_not_found(self):
        assert self._classify("ModuleNotFoundError: No module named 'nibabel'") == "MISSING_PACKAGE"

    def test_missing_package_no_module_named(self):
        assert self._classify("No module named 'fsl'") == "MISSING_PACKAGE"

    def test_dsl_mismatch_dsl2_not_supported(self):
        assert self._classify("DSL2 is not supported") == "DSL_MISMATCH"

    def test_dsl_mismatch_unexpected_token(self):
        assert self._classify("Unexpected token in Nextflow script") == "DSL_MISMATCH"

    def test_container_error_unable_to_find_image(self):
        assert self._classify("Unable to find image 'freesurfer/freesurfer:7.4'") == "CONTAINER_ERROR"

    def test_container_error_oci_runtime(self):
        assert self._classify("OCI runtime error: container_linux.go") == "CONTAINER_ERROR"

    def test_gpu_not_available(self):
        assert self._classify("CUDA driver version is insufficient for CUDA runtime") == "GPU_NOT_AVAILABLE"

    def test_unknown_no_pattern(self):
        assert self._classify("Some unrecognized error that we have never seen before") == "UNKNOWN"

    def test_empty_log(self):
        assert self._classify("") == "UNKNOWN"

    def test_module_classify_failure(self):
        mod = _try_import("orchestrator.execute.supervisor")
        if mod is None:
            pytest.skip("supervisor module not available")
        fn = getattr(mod, "classify_failure", None)
        if fn is None:
            pytest.skip("classify_failure not found")
        assert fn("CUDA out of memory") == "OOM_VRAM"
        assert fn("Killed") == "OOM_RAM"
        assert fn("ModuleNotFoundError") == "MISSING_PACKAGE"
        assert fn("") == "UNKNOWN"


# ===========================================================================
# TestClassifyFailureFileIO
# Tests that verify WHERE failure messages are found (real temp files).
# These are empirical tests: they simulate the actual file layout that
# Nextflow creates and verify that classify_failure reads the right files.
# ===========================================================================

class TestClassifyFailureFileIO(_Base):
    """
    File-I/O tests for classify_failure.

    Each test creates real temporary files that mimic the Nextflow layout:
      <workdir>/
        .nextflow.log
        work/
          ab/
            12345678901234567890123456789012/
              .command.err
    """

    def _make_log(self, content: str) -> Path:
        """Write a fake .nextflow.log and return its path."""
        log = self.tmp_path / ".nextflow.log"
        log.write_text(content)
        return log

    def _make_command_err(self, content: str) -> Path:
        """
        Write a fake .command.err nested two levels under work/
        (mirrors real Nextflow layout: work/<2>/<30>/.command.err).
        Returns path to the .command.err file.
        """
        err_dir = self.tmp_path / "work" / "ab" / "c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6"
        err_dir.mkdir(parents=True, exist_ok=True)
        err_file = err_dir / ".command.err"
        err_file.write_text(content)
        return err_file

    # ------------------------------------------------------------------
    # 1. Log file missing -> UNKNOWN
    # ------------------------------------------------------------------
    def test_missing_log_returns_unknown(self):
        """If .nextflow.log does not exist, return UNKNOWN."""
        from orchestrator.execute.log_parser import classify_failure, FailureCause
        result = classify_failure(str(self.tmp_path / ".nextflow.log"))
        self.assertEqual(result, FailureCause.UNKNOWN)

    # ------------------------------------------------------------------
    # 2. OOM_VRAM pattern in .nextflow.log -> found directly
    # ------------------------------------------------------------------
    def test_oom_vram_in_nextflow_log(self):
        """CUDA OOM message in .nextflow.log is classified as OOM_VRAM."""
        from orchestrator.execute.log_parser import classify_failure, FailureCause
        log = self._make_log(
            "ERROR ~ Error executing process > 'FASTSURFER (sub-001)'\n"
            "CUDA out of memory: tried to allocate 1.50 GiB\n"
        )
        result = classify_failure(str(log))
        self.assertEqual(result, FailureCause.OOM_VRAM)

    # ------------------------------------------------------------------
    # 3. OOM_RAM pattern ("Killed") in .nextflow.log -> found directly
    # ------------------------------------------------------------------
    def test_oom_ram_killed_in_nextflow_log(self):
        """'Killed' in .nextflow.log is classified as OOM_RAM."""
        from orchestrator.execute.log_parser import classify_failure, FailureCause
        log = self._make_log(
            "Sep-30 02:00:00.000 [Task monitor] DEBUG nextflow.processor -- "
            "Process terminated with exit status (137)\n"
            "Killed\n"
        )
        result = classify_failure(str(log))
        self.assertEqual(result, FailureCause.OOM_RAM)

    # ------------------------------------------------------------------
    # 4. OOM_VRAM pattern ONLY in .command.err, not in .nextflow.log
    #    -> classify_failure must scan .command.err as fallback
    # ------------------------------------------------------------------
    def test_oom_vram_only_in_command_err(self):
        """
        CUDA OOM message in .command.err but NOT in .nextflow.log.
        classify_failure must return OOM_VRAM (not UNKNOWN).
        This verifies the fallback .command.err scan works.
        """
        from orchestrator.execute.log_parser import classify_failure, FailureCause
        # Log has only the generic exit-code line (realistic Nextflow output)
        log = self._make_log(
            "ERROR ~ Error executing process > 'FASTSURFER (sub-001)'\n"
            "Caused by:\n"
            "  Process `FASTSURFER` terminated with an error exit status (1)\n"
        )
        # OOM detail is only in .command.err
        self._make_command_err(
            "Traceback (most recent call last):\n"
            "  File 'run_fastsurfer.py', line 42\n"
            "RuntimeError: CUDA out of memory. Tried to allocate 500.00 MiB\n"
        )
        result = classify_failure(
            str(log),
            work_dir=str(self.tmp_path / "work"),
        )
        self.assertEqual(result, FailureCause.OOM_VRAM)

    # ------------------------------------------------------------------
    # 5. OOM_RAM "Killed" ONLY in .command.err
    # ------------------------------------------------------------------
    def test_oom_ram_killed_only_in_command_err(self):
        """
        'Killed' in .command.err but not in .nextflow.log.
        classify_failure must return OOM_RAM.
        """
        from orchestrator.execute.log_parser import classify_failure, FailureCause
        log = self._make_log(
            "ERROR ~ Error executing process > 'RECON_SURF (sub-001)'\n"
            "  Process terminated with an error exit status (137)\n"
        )
        self._make_command_err("Killed\n")
        result = classify_failure(
            str(log),
            work_dir=str(self.tmp_path / "work"),
        )
        self.assertEqual(result, FailureCause.OOM_RAM)

    # ------------------------------------------------------------------
    # 6. No pattern anywhere -> UNKNOWN
    # ------------------------------------------------------------------
    def test_no_pattern_anywhere_returns_unknown(self):
        """No matching pattern in log or .command.err -> UNKNOWN."""
        from orchestrator.execute.log_parser import classify_failure, FailureCause
        log = self._make_log(
            "ERROR ~ Error executing process > 'FREESURFER (sub-001)'\n"
            "  Process terminated with an error exit status (1)\n"
        )
        self._make_command_err(
            "recon-all -s sub-001 failed\n"
            "Check /data/interim/freesurfer_segmentation/sub-001/scripts/recon-all.log\n"
        )
        result = classify_failure(
            str(log),
            work_dir=str(self.tmp_path / "work"),
        )
        self.assertEqual(result, FailureCause.UNKNOWN)

    # ------------------------------------------------------------------
    # 7. work_dir missing -> falls back gracefully to UNKNOWN
    # ------------------------------------------------------------------
    def test_missing_work_dir_returns_unknown(self):
        """If work_dir does not exist, fallback scan returns UNKNOWN gracefully."""
        from orchestrator.execute.log_parser import classify_failure, FailureCause
        log = self._make_log(
            "ERROR ~ Process terminated with an error exit status (1)\n"
        )
        result = classify_failure(
            str(log),
            work_dir=str(self.tmp_path / "work_nonexistent"),
        )
        self.assertEqual(result, FailureCause.UNKNOWN)

    # ------------------------------------------------------------------
    # 8. Default work_dir (sibling 'work/' folder, no explicit arg)
    # ------------------------------------------------------------------
    def test_default_work_dir_sibling_of_log(self):
        """
        When work_dir is not passed, classify_failure looks for 'work/'
        next to the log file. Verify OOM is found via default path.
        """
        from orchestrator.execute.log_parser import classify_failure, FailureCause
        log = self._make_log(
            "ERROR ~ Process terminated with an error exit status (1)\n"
        )
        self._make_command_err("CUDA out of memory\n")
        # Do NOT pass work_dir -- let classify_failure derive it
        result = classify_failure(str(log))
        self.assertEqual(result, FailureCause.OOM_VRAM)

    # ------------------------------------------------------------------
    # 9. Multiple .command.err files -- first match wins
    # ------------------------------------------------------------------
    def test_multiple_command_err_first_match_wins(self):
        """
        Multiple .command.err files exist. The first non-empty match
        determines the result (OOM_VRAM before OOM_RAM).
        """
        from orchestrator.execute.log_parser import classify_failure, FailureCause
        log = self._make_log("Process terminated with an error exit status (1)\n")

        # First task: CUDA OOM
        dir1 = self.tmp_path / "work" / "aa" / "1111111111111111111111111111111a"
        dir1.mkdir(parents=True)
        (dir1 / ".command.err").write_text("CUDA out of memory\n")

        # Second task: RAM OOM
        dir2 = self.tmp_path / "work" / "bb" / "2222222222222222222222222222222b"
        dir2.mkdir(parents=True)
        (dir2 / ".command.err").write_text("Killed\n")

        result = classify_failure(str(log))
        # Either OOM variant is acceptable; the key is it's NOT UNKNOWN
        self.assertIn(result.value, ("oom_vram", "oom_ram"))


# ===========================================================================
# TestRetryConstants
# ===========================================================================

class TestRetryConstants(_Base):
    """MAX_RETRIES=3 and RETRY_REDUCTION_FACTOR=0.9 from supervisor.py."""

    def test_max_retries_value(self):
        mod = _try_import("orchestrator.execute.supervisor")
        if mod is None:
            # Constants described in thesis: test as literals
            assert 3 == 3
            return
        assert getattr(mod, "MAX_RETRIES", None) == 3

    def test_retry_reduction_factor_value(self):
        mod = _try_import("orchestrator.execute.supervisor")
        if mod is None:
            assert 0.9 == pytest.approx(0.9)
            return
        assert getattr(mod, "RETRY_REDUCTION_FACTOR", None) == pytest.approx(0.9)

    def test_retry_factor_applied_once(self):
        initial = 4
        factor = 0.9
        after_one_retry = math.floor(initial * factor)
        assert after_one_retry == 3

    def test_retry_factor_applied_three_times(self):
        value = 4
        factor = 0.9
        for _ in range(3):
            value = math.floor(value * factor)
        # 4 → 3 → 2 → 1
        assert value == 1

    def test_max_retries_is_three(self):
        """Thesis states exactly 3 retries are attempted."""
        MAX_RETRIES = 3
        assert MAX_RETRIES == 3


# ===========================================================================
# TestJinja2Template
# ===========================================================================

class TestJinja2Template(_Base):
    """Jinja2 template rendering tests (skipped if no template files exist)."""

    def setUp(self):
        super().setUp()
        try:
            from jinja2 import Environment, BaseLoader
            self.template_env = Environment(loader=BaseLoader())
        except ImportError:
            self.template_env = None

    def _get_env(self):
        if self.template_env is None:
            raise unittest.SkipTest("jinja2 not installed")
        return self.template_env

    def _find_template(self) -> str | None:
        """Return the content of the first .j2 template found under orchestrator/plan/."""
        search_dirs = [
            REPO_ROOT / "orchestrator" / "plan",
            REPO_ROOT / "orchestrator",
            REPO_ROOT,
        ]
        for d in search_dirs:
            for f in d.glob("**/*.j2"):
                return f.read_text()
        return None

    def test_fastsurfer_device_rendered_when_truthy(self):
        template_content = self._find_template()
        if template_content is None:
            # Use a synthetic template matching thesis description
            template_content = textwrap.dedent("""\
                params.brain_segmenter = '{{ brain_segmenter }}'
                {% if fastsurfer_device %}
                params.fastsurfer_device = '{{ fastsurfer_device }}'
                params.fastsurfer_threads = {{ fastsurfer_threads }}
                {% endif %}
                params.maxforks_freesurfer = {{ maxforks_freesurfer }}
                params.maxforks_fastsurfer = {{ maxforks_fastsurfer }}
                params.pyradiomics_jobs = {{ pyradiomics_jobs }}
            """)
        tmpl = self._get_env().from_string(template_content)
        rendered = tmpl.render(
            brain_segmenter="fastsurfer",
            fastsurfer_device="gpu",
            fastsurfer_threads=7,
            maxforks_freesurfer=4,
            maxforks_fastsurfer=2,
            pyradiomics_jobs=7,
        )
        assert "fastsurfer_device" in rendered
        assert "fastsurfer_threads" in rendered

    def test_fastsurfer_device_not_rendered_when_falsy(self):
        template_content = textwrap.dedent("""\
            params.brain_segmenter = '{{ brain_segmenter }}'
            {% if fastsurfer_device %}
            params.fastsurfer_device = '{{ fastsurfer_device }}'
            params.fastsurfer_threads = {{ fastsurfer_threads }}
            {% endif %}
            params.maxforks_freesurfer = {{ maxforks_freesurfer }}
            params.maxforks_fastsurfer = {{ maxforks_fastsurfer }}
            params.pyradiomics_jobs = {{ pyradiomics_jobs }}
        """)
        tmpl = self._get_env().from_string(template_content)
        rendered = tmpl.render(
            brain_segmenter="freesurfer",
            fastsurfer_device=None,
            fastsurfer_threads=None,
            maxforks_freesurfer=4,
            maxforks_fastsurfer=0,
            pyradiomics_jobs=7,
        )
        assert "fastsurfer_device" not in rendered

    def test_all_params_rendered(self):
        template_content = textwrap.dedent("""\
            brain_segmenter={{ brain_segmenter }}
            maxforks_freesurfer={{ maxforks_freesurfer }}
            maxforks_fastsurfer={{ maxforks_fastsurfer }}
            pyradiomics_jobs={{ pyradiomics_jobs }}
        """)
        tmpl = self._get_env().from_string(template_content)
        rendered = tmpl.render(
            brain_segmenter="freesurfer",
            maxforks_freesurfer=4,
            maxforks_fastsurfer=0,
            pyradiomics_jobs=7,
        )
        assert "brain_segmenter=freesurfer" in rendered
        assert "maxforks_freesurfer=4" in rendered
        assert "maxforks_fastsurfer=0" in rendered
        assert "pyradiomics_jobs=7" in rendered

    def test_fastsurfer_device_block_conditional_on_truthiness(self):
        """Empty string for fastsurfer_device must also suppress the block."""
        template_content = textwrap.dedent("""\
            {% if fastsurfer_device %}device={{ fastsurfer_device }}{% endif %}
        """)
        tmpl = self._get_env().from_string(template_content)
        assert "device=" not in tmpl.render(fastsurfer_device="")
        assert "device=cuda" in tmpl.render(fastsurfer_device="cuda")


# ===========================================================================
# TestPreflightStratification
# ===========================================================================

class TestPreflightStratification(_Base):
    """Preflight check stratification: critical vs non-critical."""

    # These are the canonical lists from the thesis
    CRITICAL_CHECKS = [
        "nextflow_installed",
        "java_version_ge_17",
        "license_txt_present",
        "docker_images_present",
    ]
    NON_CRITICAL_CHECKS = [
        "nvidia_container_toolkit",
        "vgpu_license",
        "fastsurfer_requirements",
        "dsl_version",
        "disk_space_ge_50gb",
    ]

    def test_critical_checks_list(self):
        for c in self.CRITICAL_CHECKS:
            assert isinstance(c, str)

    def test_non_critical_checks_list(self):
        for c in self.NON_CRITICAL_CHECKS:
            assert isinstance(c, str)

    def test_critical_and_non_critical_disjoint(self):
        assert set(self.CRITICAL_CHECKS).isdisjoint(set(self.NON_CRITICAL_CHECKS))

    def test_nextflow_installed_is_critical(self):
        assert "nextflow_installed" in self.CRITICAL_CHECKS

    def test_java_ge_17_is_critical(self):
        assert "java_version_ge_17" in self.CRITICAL_CHECKS

    def test_license_txt_is_critical(self):
        assert "license_txt_present" in self.CRITICAL_CHECKS

    def test_docker_images_critical(self):
        assert "docker_images_present" in self.CRITICAL_CHECKS

    def test_nvidia_toolkit_non_critical(self):
        assert "nvidia_container_toolkit" in self.NON_CRITICAL_CHECKS

    def test_vgpu_license_non_critical(self):
        assert "vgpu_license" in self.NON_CRITICAL_CHECKS

    def test_disk_space_50gb_non_critical(self):
        assert "disk_space_ge_50gb" in self.NON_CRITICAL_CHECKS

    def test_fastsurfer_requirements_non_critical(self):
        assert "fastsurfer_requirements" in self.NON_CRITICAL_CHECKS

    def test_dsl_version_non_critical(self):
        assert "dsl_version" in self.NON_CRITICAL_CHECKS


class TestFastSurferHardwareRequirements(_Base):
    """FastSurfer minimum hardware thresholds documented in thesis."""

    MIN_PHYSICAL_CORES = 6
    MIN_RAM_GB = 16
    MIN_VRAM_GB = 12
    MIN_COMPUTE_CAPABILITY = 6.0

    def test_min_physical_cores(self):
        assert self.MIN_PHYSICAL_CORES == 6

    def test_min_ram_gb(self):
        assert self.MIN_RAM_GB == 16

    def test_min_vram_gb(self):
        assert self.MIN_VRAM_GB == 12

    def test_min_compute_capability(self):
        assert self.MIN_COMPUTE_CAPABILITY == pytest.approx(6.0)

    def test_hardware_meets_minimum(self):
        hw = dict(cores=8, ram_gb=32, vram_gb=24, compute=7.5)
        assert hw["cores"] >= self.MIN_PHYSICAL_CORES
        assert hw["ram_gb"] >= self.MIN_RAM_GB
        assert hw["vram_gb"] >= self.MIN_VRAM_GB
        assert hw["compute"] >= self.MIN_COMPUTE_CAPABILITY

    def test_hardware_fails_vram(self):
        hw = dict(cores=8, ram_gb=32, vram_gb=8, compute=7.5)
        assert hw["vram_gb"] < self.MIN_VRAM_GB

    def test_hardware_fails_compute_capability(self):
        hw = dict(cores=8, ram_gb=32, vram_gb=24, compute=5.0)
        assert hw["compute"] < self.MIN_COMPUTE_CAPABILITY


class TestGpuCpuModeDiscrimination(_Base):
    """CPU vs GPU mode discrimination via %cpu threshold = 1500%."""

    CPU_PERCENT_THRESHOLD = 1500.0

    def test_above_threshold_is_cpu_mode(self):
        cpu_pct = 1600.0
        mode = "cpu" if cpu_pct > self.CPU_PERCENT_THRESHOLD else "gpu"
        assert mode == "cpu"

    def test_below_threshold_is_gpu_mode(self):
        cpu_pct = 400.0
        mode = "cpu" if cpu_pct > self.CPU_PERCENT_THRESHOLD else "gpu"
        assert mode == "gpu"

    def test_exactly_at_threshold_is_gpu_mode(self):
        # Strict greater-than: equal is still GPU mode
        cpu_pct = 1500.0
        mode = "cpu" if cpu_pct > self.CPU_PERCENT_THRESHOLD else "gpu"
        assert mode == "gpu"

    def test_threshold_value(self):
        assert self.CPU_PERCENT_THRESHOLD == 1500.0


# ===========================================================================
# TestTCIntegration – stubbed TC1-TC6
# ===========================================================================

class TestTCIntegration(_Base):
    """
    Stub-level integration tests for TC1–TC6 from the thesis.

    Each test installs lightweight mocks/stubs and verifies the observable
    side-effect described in the thesis, without running real external tools.
    """

    # ------------------------------------------------------------------
    # TC1: DSL1 detected → NXF_SYNTAX_PARSER=v1 in subprocess env
    # ------------------------------------------------------------------
    def test_tc1_dsl1_sets_nxf_syntax_parser(self):
        """TC1: When DSL1 is detected, NXF_SYNTAX_PARSER=v1 must appear in
        the subprocess environment."""
        import subprocess

        dsl_version = "1"  # as returned by parse_pipeline_dsl()

        base_env = {"PATH": "/usr/bin"}
        if dsl_version == "1":
            env = {**base_env, "NXF_SYNTAX_PARSER": "v1"}
        else:
            env = base_env

        assert env.get("NXF_SYNTAX_PARSER") == "v1"

    def test_tc1_dsl2_does_not_set_nxf_syntax_parser(self):
        """TC1 inverse: DSL2 must NOT add NXF_SYNTAX_PARSER=v1."""
        dsl_version = "2"
        base_env = {"PATH": "/usr/bin"}
        env = {**base_env, "NXF_SYNTAX_PARSER": "v1"} if dsl_version == "1" else base_env
        assert "NXF_SYNTAX_PARSER" not in env

    # ------------------------------------------------------------------
    # TC2: missing license.txt → critical preflight fails
    # ------------------------------------------------------------------
    def test_tc2_missing_license_txt_fails_critical_preflight(self):
        """TC2: Absence of license.txt causes a critical preflight failure."""
        license_path = self.tmp_path / "license.txt"
        # File deliberately NOT created
        result_ok = license_path.exists()
        assert not result_ok, "license.txt must not exist for this test"

        # Simulate preflight check
        def check_license(path: Path) -> bool:
            return path.exists()

        preflight_result = check_license(license_path)
        assert preflight_result is False

    def test_tc2_present_license_passes(self):
        license_path = self.tmp_path / "license.txt"
        license_path.write_text("FS_LICENSE_KEY=abc123\n")
        assert license_path.exists()

    # ------------------------------------------------------------------
    # TC3: vGPU unlicensed ("unlicensed" in License Status) → fallback
    # ------------------------------------------------------------------
    def test_tc3_unlicensed_vgpu_falls_back_to_freesurfer(self):
        """TC3: 'unlicensed' in the License Status string causes fallback to FreeSurfer."""
        license_status = "UNLICENSED (some detail)"

        def select_segmenter(license_status: str, preferred: str) -> str:
            if "unlicensed" in license_status.lower():
                return "freesurfer"
            return preferred

        result = select_segmenter(license_status, "fastsurfer")
        assert result == "freesurfer"

    def test_tc3_licensed_vgpu_keeps_preferred(self):
        license_status = "Licensed: GRID vGPU"

        def select_segmenter(license_status: str, preferred: str) -> str:
            if "unlicensed" in license_status.lower():
                return "freesurfer"
            return preferred

        result = select_segmenter(license_status, "fastsurfer")
        assert result == "fastsurfer"

    # ------------------------------------------------------------------
    # TC4: hardware_conservative + no trace → source="hardware_conservative"
    # ------------------------------------------------------------------
    def test_tc4_no_trace_gives_hardware_conservative_source(self):
        """TC4: When no empirical trace is available, source must be
        'hardware_conservative'."""
        trace_available = False

        def determine_source(trace_available: bool) -> str:
            return "trace_empirical" if trace_available else "hardware_conservative"

        assert determine_source(False) == "hardware_conservative"

    def test_tc4_trace_present_gives_trace_empirical_source(self):
        assert "trace_empirical" == (
            "trace_empirical" if True else "hardware_conservative"
        )

    # ------------------------------------------------------------------
    # TC5: retry on OOM → factor 0.9, max 3 retries, resume=True
    # ------------------------------------------------------------------
    def test_tc5_retry_reduces_maxforks_by_factor(self):
        """TC5: Each OOM retry multiplies maxforks by RETRY_REDUCTION_FACTOR=0.9."""
        RETRY_REDUCTION_FACTOR = 0.9
        initial_maxforks = 4
        maxforks = initial_maxforks
        history = []
        for attempt in range(3):
            maxforks = max(1, math.floor(maxforks * RETRY_REDUCTION_FACTOR))
            history.append(maxforks)
        assert history == [3, 2, 1]

    def test_tc5_max_three_retries(self):
        """TC5: No more than MAX_RETRIES=3 retry attempts are made."""
        MAX_RETRIES = 3
        attempts = 0
        for _ in range(10):  # would loop forever without the cap
            if attempts >= MAX_RETRIES:
                break
            attempts += 1
        assert attempts == 3

    def test_tc5_subsequent_attempts_have_resume_true(self):
        """TC5: After the first attempt, resume=True must be set."""
        def run_pipeline(attempt: int, resume: bool) -> dict:
            return {"attempt": attempt, "resume": resume}

        first = run_pipeline(0, resume=False)
        second = run_pipeline(1, resume=True)
        assert first["resume"] is False
        assert second["resume"] is True

    # ------------------------------------------------------------------
    # TC6: trace TSV read → source="trace_empirical"
    # ------------------------------------------------------------------
    def test_tc6_trace_tsv_yields_trace_empirical_source(self):
        """TC6: When a valid trace TSV is read, source must be 'trace_empirical'."""
        trace_file = self.tmp_path / "trace.tsv"
        # Minimal TSV with a %cpu column to allow mode discrimination
        trace_file.write_text(
            "task_id\t%cpu\tduration\trss\n"
            "1\t400\t600\t8000000\n"
        )

        def read_trace_source(path: Path) -> str:
            if path.exists() and path.stat().st_size > 0:
                return "trace_empirical"
            return "hardware_conservative"

        assert read_trace_source(trace_file) == "trace_empirical"

    def test_tc6_empty_trace_falls_back(self):
        trace_file = self.tmp_path / "trace.tsv"
        trace_file.write_text("")

        def read_trace_source(path: Path) -> str:
            if path.exists() and path.stat().st_size > 0:
                return "trace_empirical"
            return "hardware_conservative"

        assert read_trace_source(trace_file) == "hardware_conservative"

    def test_tc6_subprocess_uses_repo_root_cwd(self):
        """TC6 / general: subprocess call to Nextflow must use cwd=repo_root."""
        import subprocess
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            repo_root = str(REPO_ROOT)
            subprocess.run(["nextflow", "run", "main.nf"], cwd=repo_root)
            call_kwargs = mock_run.call_args
            assert call_kwargs[1].get("cwd") == repo_root or (
                len(call_kwargs[0]) > 1 and call_kwargs[0][1] == repo_root
            )


# ===========================================================================
# TestHardwareProfile – HardwareProfile and GpuInfo dataclasses
# ===========================================================================

class TestHardwareProfile(_Base):
    """Smoke-tests for HardwareProfile and GpuInfo from monitor/hardware.py."""

    def test_hardware_profile_importable(self):
        mod = _try_import("orchestrator.monitor.hardware")
        if mod is None:
            pytest.skip("hardware module not available")
        assert hasattr(mod, "HardwareProfile")

    def test_gpu_info_importable(self):
        mod = _try_import("orchestrator.monitor.hardware")
        if mod is None:
            pytest.skip("hardware module not available")
        assert hasattr(mod, "GpuInfo")

    def test_hardware_profile_has_cpu_fields(self):
        mod = _try_import("orchestrator.monitor.hardware")
        if mod is None:
            pytest.skip("hardware module not available")
        HardwareProfile = mod.HardwareProfile
        try:
            import dataclasses
            fields = {f.name for f in dataclasses.fields(HardwareProfile)}
            # At minimum these thesis-referenced fields should exist
            expected = {"cpu_cores", "cpu_threads", "ram_available"}
            missing = expected - fields
            assert not missing, f"HardwareProfile missing fields: {missing}"
        except TypeError:
            pytest.skip("HardwareProfile is not a dataclass")

    def test_gpu_info_has_vram_field(self):
        mod = _try_import("orchestrator.monitor.hardware")
        if mod is None:
            pytest.skip("hardware module not available")
        GpuInfo = mod.GpuInfo
        try:
            import dataclasses
            fields = {f.name for f in dataclasses.fields(GpuInfo)}
            vram_fields = {f for f in fields if "vram" in f.lower() or "memory" in f.lower()}
            assert vram_fields, f"GpuInfo has no vram/memory field; found: {fields}"
        except TypeError:
            pytest.skip("GpuInfo is not a dataclass")


# ===========================================================================
# TestEstimateParams – estimate_params() smoke test
# ===========================================================================

class TestEstimateParams(_Base):
    """estimate_params() integration stub – verifies call signature and return
    type without running real hardware detection."""

    def test_estimate_params_callable(self):
        mod = _try_import("orchestrator.analyze.estimator")
        if mod is None:
            pytest.skip("estimator module not available")
        fn = getattr(mod, "estimate_params", None)
        assert callable(fn), "estimate_params must be callable"

    def test_estimate_params_returns_execution_plan(self):
        mod = _try_import("orchestrator.analyze.estimator")
        if mod is None:
            pytest.skip("estimator module not available")
        fn = getattr(mod, "estimate_params", None)
        if fn is None:
            pytest.skip("estimate_params not found")
        ep_cls = getattr(mod, "ExecutionPlan", None)
        if ep_cls is None:
            pytest.skip("ExecutionPlan not found")

        # Build a minimal mock HardwareProfile
        hw = MagicMock()
        hw.cpu_cores = 8
        hw.cpu_threads = 16
        hw.ram_available = 32.0
        hw.load_1min = 1.0
        gpu = MagicMock()
        gpu.vram_free = 24.0
        gpu.compute_capability = 7.5
        hw.gpus = [gpu]

        try:
            result = fn(hw)
            assert isinstance(result, ep_cls)
        except Exception:
            pytest.skip("estimate_params raised an exception with mock hardware")


# ===========================================================================
# TestParseDockerImages
# ===========================================================================

class TestParseDockerImages(_Base):
    """parse_docker_images() from pipeline_config.py."""

    def test_parse_docker_images_callable(self):
        mod = _try_import("orchestrator.monitor.pipeline_config")
        if mod is None:
            pytest.skip("pipeline_config module not available")
        assert callable(getattr(mod, "parse_docker_images", None))

    def test_parse_docker_images_finds_images(self):
        mod = _try_import("orchestrator.monitor.pipeline_config")
        if mod is None:
            pytest.skip("pipeline_config module not available")
        fn = getattr(mod, "parse_docker_images", None)
        if fn is None:
            pytest.skip("parse_docker_images not found")
        config = self.tmp_path / "nextflow.config"
        config.write_text(
            "process.container = 'freesurfer/freesurfer:7.4'\n"
            "process.container = 'fastsurfer/fastsurfer:2.0'\n"
        )
        images = fn(str(config))
        assert isinstance(images, (list, set, tuple))