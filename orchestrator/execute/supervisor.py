"""
Supervisor — Execute component.
Lancia Nextflow con il profilo adattivo generato dal Plan,
monitora l'esecuzione e gestisce il retry adattivo in caso di OOM.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .log_parser import classify_failure, FailureCause


MAX_RETRIES = 2
RETRY_REDUCTION_FACTOR = 0.6  # riduce il parametro del 40% ad ogni retry


@dataclass
class RunResult:
    success: bool
    returncode: int
    attempts: int
    failure_cause: Optional[str] = None
    log_path: Optional[str] = None


def _build_nextflow_cmd(
    pipeline: str,
    config_path: str,
    resume: bool = True,
    extra_args: list[str] = None,
) -> list[str]:
    cmd = [
        "nextflow", "run", pipeline,
        "-c", config_path,
        "-profile", "adaptive_profile",
        "-with-trace", "reports/trace.txt",
        "-with-report", "reports/report.html",
    ]
    if resume:
        cmd.append("-resume")
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def _read_maxforks_from_config(config_path: str) -> Optional[int]:
    """Legge il valore corrente di maxForks dal config generato."""
    try:
        content = Path(config_path).read_text()
        for line in content.split("\n"):
            if "params.maxforks" in line and "=" in line:
                val = line.split("=")[1].strip()
                return int(val)
    except (OSError, ValueError):
        return None


def _update_maxforks_in_config(config_path: str, new_value: int) -> None:
    """Aggiorna maxForks nel config generato per il retry."""
    content = Path(config_path).read_text()
    lines = []
    for line in content.split("\n"):
        if "params.maxforks" in line and "=" in line:
            lines.append(f"        params.maxforks          = {new_value}")
        elif "maxForks = " in line and "feature_extraction" not in line:
            lines.append(f"                maxForks = {new_value}")
        else:
            lines.append(line)
    Path(config_path).write_text("\n".join(lines))
    print(f"[Execute] maxForks aggiornato a {new_value} per il retry")


def supervise(
    pipeline: str,
    config_path: str,
    auto: bool = False,
    extra_args: list[str] = None,
) -> RunResult:
    """
    Lancia Nextflow e supervisiona il lifecycle della pipeline.
    In caso di OOM riduce maxForks e riprova con -resume.
    """
    Path("reports").mkdir(exist_ok=True)
    attempts = 0

    while attempts <= MAX_RETRIES:
        attempts += 1
        cmd = _build_nextflow_cmd(pipeline, config_path, resume=(attempts > 1), extra_args=extra_args)

        print(f"\n[Execute] Tentativo {attempts}/{MAX_RETRIES + 1}")
        print(f"[Execute] Comando: {' '.join(cmd)}")

        try:
            proc = subprocess.run(cmd, check=False)
        except FileNotFoundError:
            print("[Execute] ERRORE: nextflow non trovato nel PATH")
            return RunResult(success=False, returncode=127, attempts=attempts, failure_cause="nextflow_not_found")
        except KeyboardInterrupt:
            print("\n[Execute] Interrotto dall'utente")
            return RunResult(success=False, returncode=130, attempts=attempts, failure_cause="interrupted")

        if proc.returncode == 0:
            print("[Execute] Pipeline completata con successo.")
            return RunResult(success=True, returncode=0, attempts=attempts)

        # Classifica il fallimento leggendo .nextflow.log
        cause = classify_failure(".nextflow.log")
        print(f"[Execute] Fallimento rilevato: {cause.value}")

        if cause == FailureCause.OOM_VRAM and attempts <= MAX_RETRIES:
            current = _read_maxforks_from_config(config_path)
            if current and current > 1:
                new_val = max(1, int(current * RETRY_REDUCTION_FACTOR))
                print(f"[Execute] OOM VRAM — riduco maxForks: {current} → {new_val}")
                _update_maxforks_in_config(config_path, new_val)
                time.sleep(5)
                continue

        elif cause == FailureCause.OOM_RAM and attempts <= MAX_RETRIES:
            current = _read_maxforks_from_config(config_path)
            if current and current > 1:
                new_val = max(1, int(current * RETRY_REDUCTION_FACTOR))
                print(f"[Execute] OOM RAM — riduco maxForks: {current} → {new_val}")
                _update_maxforks_in_config(config_path, new_val)
                time.sleep(5)
                continue

        else:
            # Causa non recuperabile automaticamente
            print(f"[Execute] Causa non recuperabile: {cause.value}")
            print("[Execute] Consulta .nextflow.log per i dettagli.")
            return RunResult(
                success=False,
                returncode=proc.returncode,
                attempts=attempts,
                failure_cause=cause.value,
            )

    print(f"[Execute] Numero massimo di retry ({MAX_RETRIES}) raggiunto.")
    return RunResult(success=False, returncode=1, attempts=attempts, failure_cause="max_retries")
