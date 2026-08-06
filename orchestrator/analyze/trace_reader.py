"""
Trace reader — Analyze component.

Reads Nextflow trace TSV files and extracts resource consumption metrics
(peak_rss, duration, %cpu) for each process of interest.

Limitations:
- VRAM is NOT available in Nextflow trace files for on-premise Docker setups.
  GPU memory metrics are only collected when using Fusion + Seqera Platform
  (Nextflow >= 26.03.3-edge, Fusion >= 2.5.10). See: github.com/nextflow-io/nextflow/issues/4286
  For VRAM estimation, use dry_run.py instead.
- peak_rss = RAM host (physical memory), not VRAM.
- For fastsurfer on GPU, peak_rss represents the host RAM used by the
  post-processing CPU component, not the GPU VRAM used by the CNN inference.

FastSurfer device detection:
  FastSurfer can run on GPU (cuda) or CPU. The device used is not explicitly
  recorded in the trace, but can be inferred from %cpu:
  - GPU mode:  %cpu ~ 800-1400% (only post-processing runs on CPU, ~10-12 cores)
  - CPU mode:  %cpu > 1500%     (entire pipeline runs on CPU, all threads used)
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional


# Threshold to distinguish fastsurfer GPU vs CPU mode from %cpu value.
# GPU mode: CNN inference on GPU, only post-processing on CPU (~10-12 cores = ~1100%)
# CPU mode: everything on CPU (all threads, typically > 1500%)
FASTSURFER_GPU_CPU_THRESHOLD = 1500.0


def _parse_memory_to_gb(value: str) -> Optional[float]:
    """
    Converts Nextflow memory strings to GB.

    Handles: "2 GB", "2.1 GB", "12.7 GB", "73.5 MB", "512 KB"
    Returns None for missing values ("-" or empty).
    """
    if not value or value.strip() in ('-', ''):
        return None
    value = value.strip()
    try:
        if 'GB' in value or value.endswith(' G'):
            return float(value.replace('GB', '').replace(' G', '').strip())
        elif 'MB' in value or value.endswith(' M'):
            return float(value.replace('MB', '').replace(' M', '').strip()) / 1024
        elif 'KB' in value or value.endswith(' K'):
            return float(value.replace('KB', '').replace(' K', '').strip()) / 1024 / 1024
    except ValueError:
        return None
    return None


def _parse_cpu_pct(value: str) -> Optional[float]:
    """
    Parses %cpu string to float.
    Examples: "100.0%" -> 100.0, "1156.8%" -> 1156.8, "-" -> None
    """
    if not value or value.strip() in ('-', ''):
        return None
    try:
        return float(value.strip().replace('%', ''))
    except ValueError:
        return None


def _parse_duration_to_min(value: str) -> Optional[float]:
    """
    Converts Nextflow duration strings to minutes.
    Examples: "2h 36m 9s" -> 156.15, "10h 57m 27s" -> 657.45, "18.5s" -> 0.31
    """
    if not value or value.strip() in ('-', ''):
        return None

    total_min = 0.0
    days  = re.search(r'(\d+)d', value)
    hours = re.search(r'(\d+)h', value)
    mins  = re.search(r'(\d+)m\b', value)
    secs  = re.search(r'(\d+\.?\d*)s', value)

    if days:  total_min += float(days.group(1))  * 24 * 60
    if hours: total_min += float(hours.group(1)) * 60
    if mins:  total_min += float(mins.group(1))
    if secs:  total_min += float(secs.group(1))  / 60

    return round(total_min, 3) if total_min > 0 else None


def detect_fastsurfer_device(cpu_pct: float) -> str:
    """
    Infers whether fastsurfer ran on GPU (cuda) or CPU based on %cpu.

    Rationale:
    - GPU mode: CNN inference runs on GPU. Only post-processing (recon-surf)
      runs on CPU, using ~10-12 cores -> %cpu typically 800-1400%.
    - CPU mode: entire pipeline (CNN + post-processing) runs on CPU,
      using all available threads -> %cpu typically > 1500%.

    Threshold: 1500% (conservative, avoids misclassification near the boundary).
    """
    if cpu_pct < FASTSURFER_GPU_CPU_THRESHOLD:
        return "cuda"
    return "cpu"


def _read_tsv_rows(tsv_path: str) -> list[dict]:
    """Reads a TSV file and returns all rows as list of dicts."""
    path = Path(tsv_path)
    if not path.exists():
        return []
    try:
        with open(path) as f:
            return list(csv.DictReader(f, delimiter='\t'))
    except OSError:
        return []


def extract_process_stats(
    tsv_paths: list[str],
    process_filter: str,
    device_filter: Optional[str] = None,
) -> dict:
    """
    Extracts resource consumption statistics for a given process
    across one or more trace TSV files.

    Parameters
    ----------
    tsv_paths : list of str
        Paths to Nextflow trace TSV files to read.
    process_filter : str
        Substring to match against the 'name' column
        (e.g. "freesurfer", "fastsurfer", "feature_extraction").
    device_filter : str, optional
        For fastsurfer only: "cuda" or "cpu".
        If None, includes all fastsurfer rows regardless of device.

    Returns
    -------
    dict with keys:
        count               int    number of COMPLETED tasks found
        peak_rss_max_gb     float  maximum peak_rss observed (worst case)
        peak_rss_mean_gb    float  mean peak_rss
        duration_max_min    float  maximum duration in minutes
        duration_mean_min   float  mean duration in minutes
        device              str    detected device ("cuda"/"cpu"/None)
    Returns empty dict if no valid data found.
    """
    rss_values      = []
    duration_values = []
    detected_devices = []

    for tsv_path in tsv_paths:
        rows = _read_tsv_rows(tsv_path)

        for row in rows:
            name   = row.get('name', '')
            status = row.get('status', '')

            # filter: only COMPLETED tasks for the requested process
            if status != 'COMPLETED':
                continue
            if process_filter not in name.lower():
                continue

            cpu_pct = _parse_cpu_pct(row.get('%cpu', '-'))

            # for fastsurfer: optionally filter by device
            if process_filter == 'fastsurfer' and cpu_pct is not None:
                detected_device = detect_fastsurfer_device(cpu_pct)
                if device_filter is not None and detected_device != device_filter:
                    continue
                detected_devices.append(detected_device)

            rss  = _parse_memory_to_gb(row.get('peak_rss', '-'))
            dur  = _parse_duration_to_min(row.get('realtime', '-'))

            if rss is not None:
                rss_values.append(rss)
            if dur is not None:
                duration_values.append(dur)

    if not rss_values:
        return {}

    # determine predominant device for fastsurfer
    device = None
    if detected_devices:
        cuda_count = detected_devices.count('cuda')
        cpu_count  = detected_devices.count('cpu')
        device = 'cuda' if cuda_count >= cpu_count else 'cpu'

    return {
        'count':            len(rss_values),
        'peak_rss_max_gb':  round(max(rss_values), 3),
        'peak_rss_mean_gb': round(sum(rss_values) / len(rss_values), 3),
        'duration_max_min': round(max(duration_values), 1) if duration_values else None,
        'duration_mean_min':round(sum(duration_values) / len(duration_values), 1) if duration_values else None,
        'device':           device,
    }


def find_trace_files(
    repo_root: str,
    pipeline: str,
    custom_dir: Optional[str] = None,
) -> list[str]:
    """
    Finds all trace TSV files for a given pipeline.

    Default paths:
        preprocessing -> <repo_root>/reports/traces/preprocessing/*.tsv
        training      -> <repo_root>/reports/traces/training/*.tsv

    If custom_dir is provided, searches there instead.

    Returns list of absolute paths sorted by modification time (oldest first).
    """
    if custom_dir:
        folder = Path(custom_dir)
    else:
        folder = Path(repo_root) / "reports" / "traces" / pipeline

    if not folder.exists():
        return []

    files = sorted(
        folder.glob("*.tsv"),
        key=lambda p: p.stat().st_mtime
    )
    return [str(f) for f in files]


def get_peak_rss_for_process(
    process_name: str,
    repo_root: str,
    pipeline: str = "preprocessing",
    custom_traces_dir: Optional[str] = None,
    device_filter: Optional[str] = None,
) -> tuple[Optional[float], int]:
    """
    Main entry point: returns the maximum peak_rss observed for a process
    across all available trace files.

    Returns
    -------
    (peak_rss_max_gb, n_observations)
    peak_rss_max_gb is None if no data available.
    """
    tsv_files = find_trace_files(repo_root, pipeline, custom_traces_dir)
    if not tsv_files:
        return None, 0

    stats = extract_process_stats(tsv_files, process_name, device_filter)
    if not stats:
        return None, 0

    return stats['peak_rss_max_gb'], stats['count']


def summarize_traces(
    repo_root: str,
    pipeline: str = "preprocessing",
    custom_traces_dir: Optional[str] = None,
) -> dict:
    """
    Returns a summary of all resource metrics available in the trace files
    for a given pipeline. Useful for reporting in the CLI summary.

    Returns dict with process names as keys and their stats as values.
    """
    tsv_files = find_trace_files(repo_root, pipeline, custom_traces_dir)
    if not tsv_files:
        return {}

    summary = {}

    for process in ['freesurfer', 'fastsurfer', 'feature_extraction',
                    'nifti_converter', 'roi_creator']:
        # for fastsurfer, separate GPU and CPU stats
        if process == 'fastsurfer':
            for device in ['cuda', 'cpu']:
                stats = extract_process_stats(tsv_files, process, device)
                if stats:
                    summary[f'fastsurfer_{device}'] = stats
        else:
            stats = extract_process_stats(tsv_files, process)
            if stats:
                summary[process] = stats

    return summary