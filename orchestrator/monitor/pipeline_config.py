import yaml
from typing import Optional
from pathlib import Path

def parse_docker_images(repo_root: str, compose_file: Optional[str] = None) -> list[str]:
    """
    Legge docker-compose.yml e estrae i nomi delle immagini
    definite nei servizi, invece di hardcodarli nel modulo.

    Se compose_file è specificato, cerca solo quel file.
    Altrimenti cerca docker-compose.yml e docker-compose.yaml
    in ordine, usando il primo trovato.

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