"""
Config generator — Plan component.
Generates adaptive_profile.config from an ExecutionPlan,
without touching the original nextflow.config of the FTD repo.
Principle from Geniac: generate a separate file, do not modify the pipeline.
"""

from __future__ import annotations

import socket
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ..analyze.estimator import ExecutionPlan
from ..monitor.hardware import HardwareProfile


TEMPLATE_DIR  = Path(__file__).parent / "templates"
TEMPLATE_NAME = "adaptive_profile.config.j2"


def generate_config(
    plan: ExecutionPlan,
    profile: HardwareProfile,
    output_path: str = "adaptive_profile.config",
) -> str:
    """
    Generates adaptive_profile.config from the ExecutionPlan and writes it to disk.
    Returns the absolute path of the generated file.

    The generated profile sets params.maxforks_freesurfer and
    params.maxforks_fastsurfer separately, so each Nextflow process
    reads its own parameter:

        process freesurfer {
            maxForks params.maxforks_freesurfer
        }
        process fastsurfer {
            maxForks params.maxforks_fastsurfer
        }
    """
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(TEMPLATE_NAME)

    if profile.gpu:
        gpu_info = f"{profile.gpu.name} ({profile.gpu.vram_total_gb:.0f}GB VRAM)"
    else:
        gpu_info = "not available"

    context = {
        "generated_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hostname":           socket.gethostname(),
        "gpu_info":           gpu_info,
        "ram_available_gb":   plan.ram_available_gb,
        "cpu_threads":        plan.cpu_threads,
        "cpu_cores_free":     plan.cpu_cores_free,
        "source":             plan.source,
        "brain_segmenter":    plan.brain_segmenter,
        "fastsurfer_device":  plan.fastsurfer_device,
        "fastsurfer_threads": plan.fastsurfer_threads,
        "maxforks_freesurfer": plan.maxforks_freesurfer,
        "maxforks_fastsurfer": plan.maxforks_fastsurfer,
        "pyradiomics_jobs":   plan.pyradiomics_jobs,
    }

    rendered = template.render(**context)
    output   = Path(output_path)
    output.write_text(rendered)

    print(f"[Plan] Config generated: {output.resolve()}")
    return str(output.resolve())