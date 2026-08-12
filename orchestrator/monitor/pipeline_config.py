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

def parse_pipeline_dsl(
    repo_root: str,
    pipeline_file: str = "nextflow/preprocessing.nf",
) -> Optional[str]:
    """
    Reads the DSL version declared in the pipeline file or nextflow.config.
    Returns "1", "2", or None if not declared.
    """
    # check file .nf
    nf_path = Path(repo_root) / pipeline_file
    if nf_path.exists():
        try:
            content = nf_path.read_text(errors="replace")
            match = re.search(r"nextflow\.enable\.dsl\s*=\s*([12])", content)
            if match:
                return match.group(1)
        except OSError:
            pass

    # check in nextflow.config
    config_content = _read_config_text(repo_root)
    if config_content:
        match = re.search(r"nextflow\.enable\.dsl\s*=\s*([12])", config_content)
        if match:
            return match.group(1)

    return None