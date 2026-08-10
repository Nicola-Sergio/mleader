import yaml
from typing import Optional
from pathlib import Path

def parse_docker_images(repo_root: str, compose_file: Optional[str] = None) -> list[str]:
    """
    Reads docker-compose.yml and extracts the names of the images
    defined in the services, instead of hardcoding them in the module.

    If compose_file is specified, looks for that file only.
    Otherwise, looks for docker-compose.yml and docker-compose.yaml
    in order, using the first one found.

    """
    if compose_file:
        candidates = [Path(repo_root) / compose_file]
    else:
        candidates = [
            Path(repo_root) / "docker-compose.yml",
            Path(repo_root) / "docker-compose.yaml",
        ]

    compose_path = None
    for candidate in candidates:
        if candidate.exists():
            compose_path = candidate
            break

    if compose_path is None:
        return []

    try:
        with open(compose_path) as f:
            compose = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return []

    images = []
    services = compose.get("services", {})
    for service_name, service_config in services.items():
        image = service_config.get("image")
        if image:
            images.append(image)

    return images