# execute/knowledge_collector.py

import csv
import socket
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..monitor.hardware import HardwareProfile
from ..analyze.trace_reader import extract_peak_rss_gb, _parse_memory_to_gb


DATASET_PATH = Path(__file__).parent.parent / "knowledge_base" / "dataset.csv"

DATASET_FIELDS = [
    # host
    "timestamp", "host_id", "cpu_model", "cpu_cores", "cpu_threads",
    "ram_total_gb", "ram_available_gb", "gpu_name", "vram_total_gb", "is_vm",
    # pipeline
    "pipeline", "process_name", "maxforks", "fastsurfer_device",
    "fastsurfer_threads", "pyradiomics_jobs", "n_subjects",
    # misurato
    "peak_rss_gb", "peak_vmem_gb", "duration_min", "cpu_pct",
    "rchar_gb", "wchar_gb",
]


def _parse_duration_to_min(value: str) -> Optional[float]:
    """
    Converts Nextflow duration strings to minutes.
    Examples: "2h 36m 9s" → 156.15, "18.5s" → 0.31, "3m 30s" → 3.5
    """
    if not value or value == '-':
        return None

    total_min = 0.0
    import re

    days = re.search(r'(\d+)d', value)
    hours = re.search(r'(\d+)h', value)
    mins = re.search(r'(\d+)m', value)
    secs = re.search(r'(\d+\.?\d*)s', value)

    if days:
        total_min += float(days.group(1)) * 24 * 60
    if hours:
        total_min += float(hours.group(1)) * 60
    if mins:
        total_min += float(mins.group(1))
    if secs:
        total_min += float(secs.group(1)) / 60

    return round(total_min, 2) if total_min > 0 else None


def collect_from_trace(
    trace_path: str,
    profile: HardwareProfile,
    pipeline: str,
    maxforks: int,
    fastsurfer_device: Optional[str],
    fastsurfer_threads: Optional[int],
    pyradiomics_jobs: int,
    dataset_path: str = None,
) -> int:
    """
    Reads a Nextflow trace TSV, enriches each completed task row
    with hardware information from the profile, and appends to dataset.csv.

    Returns the number of rows added.
    """
    output_path = Path(dataset_path or DATASET_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # determina se siamo su VM
    try:
        import subprocess
        result = subprocess.run(
            ["systemd-detect-virt"],
            capture_output=True, timeout=5
        )
        is_vm = result.returncode == 0 and result.stdout.decode().strip() != "none"
    except Exception:
        is_vm = None

    host_id = socket.gethostname()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # info GPU
    gpu_name = profile.gpu.name if profile.gpu else None
    vram_total = profile.gpu.vram_total_gb if profile.gpu else None

    rows_added = 0
    write_header = not output_path.exists()

    with open(trace_path) as tsv_f, \
         open(output_path, "a", newline="") as csv_f:

        writer = csv.DictWriter(csv_f, fieldnames=DATASET_FIELDS)
        if write_header:
            writer.writeheader()

        reader = csv.DictReader(tsv_f, delimiter='\t')
        for row in reader:
            name = row.get('name', '')
            status = row.get('status', '')

            if status != 'COMPLETED':
                continue

            # determina il processo
            if 'freesurfer' in name.lower():
                process_name = 'freesurfer'
            elif 'fastsurfer' in name.lower():
                process_name = 'fastsurfer'
            elif 'feature_extraction' in name.lower():
                process_name = 'feature_extraction'
            else:
                continue  # ignora altri processi

            # estrai valori
            peak_rss = _parse_memory_to_gb(row.get('peak_rss', '-'))
            peak_vmem = _parse_memory_to_gb(row.get('peak_vmem', '-'))
            duration = _parse_duration_to_min(row.get('realtime', '-'))
            cpu_pct = row.get('%cpu', '-').replace('%', '').strip()
            rchar = _parse_memory_to_gb(row.get('rchar', '-'))
            wchar = _parse_memory_to_gb(row.get('wchar', '-'))

            writer.writerow({
                "timestamp":         timestamp,
                "host_id":           host_id,
                "cpu_model":         profile.cpu_model if hasattr(profile, 'cpu_model') else None,
                "cpu_cores":         profile.cpu_cores,
                "cpu_threads":       profile.cpu_threads,
                "ram_total_gb":      round(profile.ram_total_gb, 1),
                "ram_available_gb":  round(profile.ram_available_gb, 1),
                "gpu_name":          gpu_name,
                "vram_total_gb":     vram_total,
                "is_vm":             is_vm,
                "pipeline":          pipeline,
                "process_name":      process_name,
                "maxforks":          maxforks,
                "fastsurfer_device": fastsurfer_device,
                "fastsurfer_threads":fastsurfer_threads,
                "pyradiomics_jobs":  pyradiomics_jobs,
                "n_subjects":        None,  # da popolare se noto
                "peak_rss_gb":       peak_rss,
                "peak_vmem_gb":      peak_vmem,
                "duration_min":      duration,
                "cpu_pct":           cpu_pct if cpu_pct != '-' else None,
                "rchar_gb":          rchar,
                "wchar_gb":          wchar,
            })
            rows_added += 1

    return rows_added