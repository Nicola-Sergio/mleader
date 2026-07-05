"""
Config generator — Plan component.
Genera adaptive_profile.config da un ExecutionPlan,
senza toccare il nextflow.config originale del repo FTD.
Principio da Geniac: genera un file separato, non modificare la pipeline.
"""

from __future__ import annotations

import math
import socket
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ..analyze.estimator import ExecutionPlan
from ..monitor.hardware import HardwareProfile


TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_NAME = "adaptive_profile.config.j2"


def generate_config(
    plan: ExecutionPlan,
    profile: HardwareProfile,
    output_path: str = "adaptive_profile.config",
) -> str:
    """
    Genera il file adaptive_profile.config a partire dall'ExecutionPlan
    e lo scrive su disco.

    Restituisce il path del file generato.
    """
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(TEMPLATE_NAME)

    # Info GPU per il commento di intestazione
    if profile.gpu:
        gpu_info = f"{profile.gpu.name} ({profile.gpu.vram_total_gb:.0f}GB VRAM)"
    else:
        gpu_info = "non disponibile"

    # maxForks per feature_extraction: limitato da CPU e RAM
    # PyRadiomics lancia pyradiomics_jobs thread per istanza
    # quindi il numero di istanze parallele è limitato dai thread rimanenti
    available_threads_for_instances = max(1, profile.cpu_threads - plan.pyradiomics_jobs)
    maxforks_feature_extraction = max(1, available_threads_for_instances // max(1, plan.pyradiomics_jobs))

    # Nome del processo segmentatore nel .nf
    segmenter_process_name = (
        "fastsurfer" if plan.brain_segmenter == "fastsurfer" else "freesurfer"
    )

    context = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hostname": socket.gethostname(),
        "gpu_info": gpu_info,
        "ram_available_gb": plan.ram_available_gb,
        "cpu_threads": plan.cpu_threads,
        "source": plan.source,
        "brain_segmenter": plan.brain_segmenter,
        "fastsurfer_device": plan.fastsurfer_device,
        "fastsurfer_threads": plan.fastsurfer_threads,
        "maxforks_segmenter": plan.maxforks_segmenter,
        "pyradiomics_jobs": plan.pyradiomics_jobs,
        "segmenter_process_name": segmenter_process_name,
        "maxforks_feature_extraction": maxforks_feature_extraction,
    }

    rendered = template.render(**context)
    output = Path(output_path)
    output.write_text(rendered)

    print(f"[Plan] Config generato: {output.resolve()}")
    return str(output.resolve())
