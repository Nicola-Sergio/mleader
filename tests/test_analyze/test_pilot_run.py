"""Unit tests for the multi-subject pilot run (sampling + VRAM aggregation)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# tests/test_execute/test_pilot_run.py -> repo root is two levels above tests/
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestrator.analyze import pilot_run as pr


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _make_dataset(self, counts):
        """counts: {group: n_images}. Creates <root>/data/raw/<group>/sub-XXX.nii"""
        dirs = []
        for group, k in counts.items():
            d = self.root / "data" / "raw" / group
            d.mkdir(parents=True)
            for i in range(k):
                (d / f"sub-{group}-{i:03d}.nii").write_text("x")
            dirs.append(f"data/raw/{group}")
        return dirs


class TestSampleSubjects(_Base):
    def test_samples_exactly_n(self):
        dirs = self._make_dataset({"bvFTD_T1": 20, "HC_T1": 20})
        sel = pr.sample_subjects(dirs, n=10, seed=42, repo_root=self.root)
        self.assertEqual(len(sel), 10)
        self.assertTrue(all(p.suffix == ".nii" for p in sel))

    def test_deterministic_same_seed(self):
        dirs = self._make_dataset({"bvFTD_T1": 20, "svPPA_T1": 20})
        a = pr.sample_subjects(dirs, n=10, seed=42, repo_root=self.root)
        b = pr.sample_subjects(dirs, n=10, seed=42, repo_root=self.root)
        self.assertEqual(a, b)

    def test_different_seed_differs(self):
        dirs = self._make_dataset({"bvFTD_T1": 50})
        a = pr.sample_subjects(dirs, n=10, seed=1, repo_root=self.root)
        b = pr.sample_subjects(dirs, n=10, seed=2, repo_root=self.root)
        self.assertNotEqual(a, b)

    def test_returns_all_when_fewer_than_n(self):
        dirs = self._make_dataset({"HC_T1": 3})
        sel = pr.sample_subjects(dirs, n=10, seed=42, repo_root=self.root)
        self.assertEqual(len(sel), 3)

    def test_pool_spans_all_groups(self):
        dirs = self._make_dataset({"bvFTD_T1": 100, "svPPA_T1": 100,
                                   "nfvPPA_T1": 100, "HC_T1": 100})
        sel = pr.sample_subjects(dirs, n=40, seed=7, repo_root=self.root)
        groups = {p.parent.name for p in sel}
        self.assertEqual(groups, {"bvFTD_T1", "svPPA_T1", "nfvPPA_T1", "HC_T1"})

    def test_picks_up_nii_gz(self):
        d = self.root / "data" / "raw" / "HC_T1"
        d.mkdir(parents=True)
        (d / "a.nii.gz").write_text("x")
        (d / "b.nii").write_text("x")
        sel = pr.sample_subjects(["data/raw/HC_T1"], n=10, seed=42, repo_root=self.root)
        self.assertEqual(len(sel), 2)

    def test_missing_dir_ignored(self):
        sel = pr.sample_subjects(["data/raw/does_not_exist"], n=10, seed=42,
                                 repo_root=self.root)
        self.assertEqual(sel, [])


class TestReadDatasetDirs(_Base):
    def test_reads_list_from_config(self):
        (self.root / "nextflow.config").write_text(
            'params.dataset = ["data/raw/nfvPPA_T1","data/raw/svPPA_T1",\n'
            '                  "data/raw/bvFTD_T1","data/raw/HC_T1"]\n'
            'params.maxforks_freesurfer = 80\n'
        )
        self.assertEqual(
            pr.read_dataset_dirs(self.root),
            ["data/raw/nfvPPA_T1", "data/raw/svPPA_T1", "data/raw/bvFTD_T1", "data/raw/HC_T1"],
        )

    def test_reads_from_nextflow_subfolder(self):
        sub = self.root / "nextflow"
        sub.mkdir()
        (sub / "nextflow.config").write_text(
            "profiles {\n    x {\n    params.dataset = [\n"
            '            "data/raw/HC_T1",\n            "data/raw/bvFTD_T1"\n        ]\n    }\n}\n'
        )
        self.assertEqual(pr.read_dataset_dirs(self.root),
                         ["data/raw/HC_T1", "data/raw/bvFTD_T1"])

    def test_missing_config_returns_empty(self):
        self.assertEqual(pr.read_dataset_dirs(self.root), [])

    def test_missing_param_returns_empty(self):
        (self.root / "nextflow.config").write_text("params.other = 1\n")
        self.assertEqual(pr.read_dataset_dirs(self.root), [])


class TestProfileAggregation(_Base):
    def _license(self):
        lic = self.root / "license.txt"
        lic.write_text("x")
        return str(lic)

    def _niis(self, k):
        out = []
        for i in range(k):
            f = self.root / f"s{i}.nii"
            f.write_text("x")
            out.append(str(f))
        return out

    def test_returns_max_times_margin(self):
        niis = self._niis(3)
        with patch.object(pr, "_profile_one", side_effect=[2.0, 3.0, 2.5]):
            cost = pr.profile_fastsurfer_vram(niis, license_path=self._license())
        self.assertAlmostEqual(cost, 3.0 * pr.EMPIRICAL_SAFETY_MARGIN, places=6)

    def test_ignores_failed_subjects(self):
        niis = self._niis(3)
        with patch.object(pr, "_profile_one", side_effect=[None, 4.0, None]):
            cost = pr.profile_fastsurfer_vram(niis, license_path=self._license())
        self.assertAlmostEqual(cost, 4.0 * pr.EMPIRICAL_SAFETY_MARGIN, places=6)

    def test_all_failed_returns_none(self):
        niis = self._niis(1)
        with patch.object(pr, "_profile_one", side_effect=[None]):
            cost = pr.profile_fastsurfer_vram(niis, license_path=self._license())
        self.assertIsNone(cost)

    def test_single_path_backward_compatible(self):
        f = self.root / "s.nii"; f.write_text("x")
        with patch.object(pr, "_profile_one", return_value=5.0):
            cost = pr.profile_fastsurfer_vram(str(f), license_path=self._license())
        self.assertAlmostEqual(cost, 5.0 * pr.EMPIRICAL_SAFETY_MARGIN, places=6)

    def test_missing_license_returns_none(self):
        f = self.root / "s.nii"; f.write_text("x")
        cost = pr.profile_fastsurfer_vram([str(f)], license_path=str(self.root / "nope.txt"))
        self.assertIsNone(cost)

    def test_profile_one_computes_delta(self):
        f = self.root / "s.nii"; f.write_text("x")
        lic = self.root / "license.txt"; lic.write_text("x")
        calls = {"n": 0}

        def fake_vram():
            calls["n"] += 1
            return 1000.0 if calls["n"] == 1 else 3048.0

        with patch.object(pr, "_sample_vram_usage_mb", side_effect=fake_vram), \
             patch.object(pr.subprocess, "run", return_value=None), \
             patch.object(pr.time, "sleep", return_value=None):
            delta = pr._profile_one(f, lic, "img", 0.0)
        self.assertIsNotNone(delta)
        self.assertAlmostEqual(delta, 2.0, places=2)


class TestDockerCommand(_Base):
    def test_runs_container_as_calling_user(self):
        f = self.root / "s.nii"; f.write_text("x")
        lic = self.root / "license.txt"; lic.write_text("x")
        captured = {}

        def fake_run(cmd, *a, **kw):
            captured["cmd"] = cmd

        out = self.root / "out"
        with patch.object(pr, "PILOT_OUTPUT_DIR", out), \
             patch.object(pr, "_sample_vram_usage_mb", return_value=1000.0), \
             patch.object(pr.subprocess, "run", side_effect=fake_run), \
             patch.object(pr.time, "sleep", return_value=None):
            pr._profile_one(f, lic, "img", 0.0)
        cmd = captured["cmd"]
        i = cmd.index("--user")
        self.assertEqual(cmd[i + 1], f"{pr.os.getuid()}:{pr.os.getgid()}")
        self.assertTrue(out.is_dir())
        self.assertIn(f"{out}:/output", cmd)


class TestRunPilot(_Base):
    def test_run_pilot_samples_then_profiles(self):
        d = self.root / "data" / "raw" / "HC_T1"
        d.mkdir(parents=True)
        for i in range(15):
            (d / f"s{i:03d}.nii").write_text("x")
        lic = self.root / "license.txt"; lic.write_text("x")
        captured = {}

        def fake_profile(subjects, **kw):
            captured["n"] = len(subjects)
            return 3.45

        with patch.object(pr, "profile_fastsurfer_vram", side_effect=fake_profile):
            cost = pr.run_pilot(["data/raw/HC_T1"], license_path=str(lic),
                                repo_root=self.root, n=10, seed=42)
        self.assertEqual(captured["n"], 10)
        self.assertEqual(cost, 3.45)


if __name__ == "__main__":
    unittest.main()