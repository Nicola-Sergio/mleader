"""
Environment checker — Monitor component.
Verifica nextflow version, docker GPU runtime, licenze, container buildati.
"""

from __future__ import annotations
from typing import Optional

import re
import subprocess
from pathlib import Path

from .hardware import HardwareProfile

def check_nextflow_version(profile: HardwareProfile) -> None:
    """
    Rileva la versione di Nextflow installata.
    """
    try:
        out = subprocess.check_output(
            ["nextflow", "-v"],
            stderr=subprocess.STDOUT,
            timeout=15,
        ).decode().strip()
        # Output tipo: "nextflow version 24.10.5.5933"
        match = re.search(r"(\d+\.\d+[\.\d]*)", out)
        profile.nextflow_version = match.group(1) if match else out
    except (subprocess.SubprocessError, FileNotFoundError):
        profile.nextflow_version = None


def check_docker_gpu_runtime(profile: HardwareProfile) -> None:
    """
    Verifica se nvidia-container-toolkit è installato sull'host.

    Approccio: which nvidia-container-toolkit. Docker stesso adotta questa
    strategia internamente (vecchio daemon/nvidia_linux.go).
    Riferimento: https://github.com/moby/moby/issues/40903
    """
    if profile.gpu is None:
        profile.docker_gpu_runtime = False
        return

    try:
        result = subprocess.run(
            ["which", "nvidia-container-toolkit"],
            capture_output=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        print(f"[Monitor] Impossibile eseguire 'which': {e}")
        profile.docker_gpu_runtime = False
        return

    profile.docker_gpu_runtime = result.returncode == 0

    if not profile.docker_gpu_runtime:
        print(
            "[Monitor] nvidia-container-toolkit non trovato nel PATH. "
            "Installazione: "
            "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
        )


def check_fs_license(profile: HardwareProfile, repo_root: str = ".") -> None:
    """
    Verifica che il file di licenza FreeSurfer (license.txt) sia presente
    nella root del repo FTD.
    """
    license_path = Path(repo_root) / "license.txt"
    profile.fs_license_present = license_path.exists()


def check_containers(
    profile: HardwareProfile,
    repo_root: str = ".",
    compose_file: Optional[str] = None,
) -> None:
    """
    Verifica quali container Docker richiesti dalla pipeline sono già
    buildati, leggendo i nomi dal docker-compose file invece di
    hardcodarli nel modulo.

    Se compose_file è specificato, usa quello. Altrimenti cerca
    docker-compose.yml e docker-compose.yaml nella root del repo.
    """
    from .pipeline_config import parse_docker_images

    required = parse_docker_images(repo_root, compose_file)

    if not required:
        print(
            "[Monitor] Nessun docker-compose trovato o nessuna immagine "
            "definita — check container saltato."
        )
        profile.containers_built = []
        profile.containers_missing = []
        return

    try:
        out = subprocess.check_output(
            ["docker", "images", "--format", "{{.Repository}}"],
            stderr=subprocess.DEVNULL,
            timeout=15,
        ).decode().strip()
        available = set(out.split("\n"))
    except (subprocess.SubprocessError, FileNotFoundError):
        available = set()

    profile.containers_built = [c for c in required if c in available]
    profile.containers_missing = [c for c in required if c not in available]


def run_environment_checks(
    profile: HardwareProfile,
    repo_root: str = ".",
    compose_file: Optional[str] = None,
) -> None:
    """
    Esegue tutti i check di ambiente e popola il profilo.
    """
    check_nextflow_version(profile)
    check_docker_gpu_runtime(profile)
    check_fs_license(profile, repo_root)
    check_containers(profile, repo_root, compose_file)
