"""Ctrl+C handling of the supervisor (no real Nextflow)."""
import os
import signal
import types
from unittest.mock import patch

import orchestrator.execute.supervisor as sup


def test_sigint_is_deferred_not_fatal(tmp_path):
    """SIGINT during the run must not raise, it must be reported."""
    def fake_run(cmd, **kw):
        os.kill(os.getpid(), signal.SIGINT)
        return types.SimpleNamespace(returncode=130)

    with patch.object(sup.subprocess, "run", side_effect=fake_run):
        proc, interrupted = sup._run_nextflow(["nextflow"], str(tmp_path), {})
    assert interrupted is True
    assert proc.returncode == 130


def test_sigint_handler_restored(tmp_path):
    before = signal.getsignal(signal.SIGINT)
    with patch.object(sup.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)):
        _, interrupted = sup._run_nextflow(["nextflow"], str(tmp_path), {})
    assert interrupted is False
    assert signal.getsignal(signal.SIGINT) is before


def test_terminal_restored_even_on_error(tmp_path):
    with patch.object(sup, "_save_terminal", return_value="saved"), \
         patch.object(sup, "_restore_terminal") as restore, \
         patch.object(sup.subprocess, "run", side_effect=FileNotFoundError):
        try:
            sup._run_nextflow(["nextflow"], str(tmp_path), {})
        except FileNotFoundError:
            pass
    restore.assert_called_once_with("saved")


def test_supervise_returns_interrupted(tmp_path):
    cfg = tmp_path / "adaptive_profile.config"
    cfg.write_text("")
    with patch.object(sup, "_run_nextflow",
                      return_value=(types.SimpleNamespace(returncode=130), True)):
        r = sup.supervise("p.nf", str(cfg), repo_root=str(tmp_path))
    assert not r.success and r.failure_cause == "interrupted" and r.returncode == 130
    assert r.attempts == 1


def test_supervise_success_unchanged(tmp_path):
    cfg = tmp_path / "adaptive_profile.config"
    cfg.write_text("")
    with patch.object(sup, "_run_nextflow",
                      return_value=(types.SimpleNamespace(returncode=0), False)):
        r = sup.supervise("p.nf", str(cfg), repo_root=str(tmp_path))
    assert r.success