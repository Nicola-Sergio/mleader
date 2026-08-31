"""
Environment checker — Monitor component.
Checks nextflow version, docker GPU runtime, licenses, container buildati.
"""

from __future__ import annotations
from typing import Optional

import re
import subprocess
from pathlib import Path

from .hardware import HardwareProfile

def check_nextflow_version(profile: HardwareProfile) -> None:
    """
    Checks the version of Nextflow installed.
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
    Checks if nvidia-container-toolkit is installed on the host.

    Approach: which nvidia-container-toolkit. Docker itself adopts this
    strategy internally (old daemon/nvidia_linux.go).
    Reference: https://github.com/moby/moby/issues/40903
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
            "[Monitor] nvidia-container-toolkit not found in PATH. "
            "Install it to enable GPU support in Docker containers. "
        )


def check_fs_license(profile: HardwareProfile, repo_root: str = ".") -> None:
    """
    Checks if the FreeSurfer license file (license.txt) is present
    in the root of the FTD repository.
    """
    license_path = Path(repo_root) / "license.txt"
    profile.fs_license_present = license_path.exists()


def check_containers(
    profile: HardwareProfile,
    repo_root: str = ".",
    compose_file: Optional[str] = None,
) -> None:
    """
    Checks which Docker containers required by the pipeline are already
    built, reading the names from the docker-compose file instead of
    hardcoding them in the module.

    If compose_file is specified, uses that. Otherwise, looks for
    docker-compose.yml and docker-compose.yaml in the root of the repo.
    """
    from .pipeline_config import parse_docker_images

    required = parse_docker_images(repo_root, compose_file)

    if not required:
        print(
            "[Monitor] No docker-compose found or no images defined — container check skipped."
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
    Runs all the environment checks and populates the profile.
    """
    check_nextflow_version(profile)
    check_docker_gpu_runtime(profile)
    check_fs_license(profile, repo_root)
    check_containers(profile, repo_root, compose_file)
