# MLEADeR

Infrastructure-aware module for adaptive MLOps pipeline orchestration.

Automatically detects hardware resources, checks environment compatibility,
estimates optimal pipeline parameters from empirical data, and launches
Nextflow pipelines with a generated configuration profile.

Developed as part of a Bachelor's thesis on adaptive MLOps orchestration
for neuroimaging pipelines (FTD — Frontotemporal Dementia).

---

## Architecture

The module implements a **MAPE-K loop** (Monitor → Analyze → Plan → Execute):

```
Monitor   — detects hardware (CPU, RAM, GPU, VRAM) and checks environment
            (Nextflow version, Docker GPU runtime, FreeSurfer license, container images)
Analyze   — reads empirical data from previous Nextflow trace files and
            estimates optimal parameters (maxForks, pyradiomics_jobs, segmenter)
Plan      — generates adaptive_profile.config with the computed parameters
Execute   — launches Nextflow with the generated profile and handles
            adaptive retry on OOM failures
```

---

## Requirements

- Python >= 3.10
- Nextflow
- Docker
- `psutil`, `jinja2`, `pyyaml` (installed automatically)
- FreeSurfer License (used for FastSurfer too)
- pytest (for running unit tests)

Optional (for GPU support):
- NVIDIA GPU with compute capability >= 6.0 (Pascal 2016+)
- nvidia-container-toolkit
- >= 12 GB VRAM, >= 16 GB RAM, >= 6 CPU cores (FastSurfer official requirements)

---

## Installation

```bash
git clone https://github.com/Nicola-Sergio/mleader.git
cd mleader
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Verify installation:

```bash
mleader --help
```

---

## Usage

### Check only (no pipeline launch)

Runs Monitor + Analyze + Plan and prints a summary without launching anything.
Useful to verify what parameters the module would use before committing to a run.

```bash
mleader check --repo-root /path/to/CMND-MLHOps_DataPreparation/ --pipeline-type <preprocessing | training>
```

### Run preprocessing pipeline

```bash
mleader run \
  --pipeline nextflow/preprocessing.nf \
  --repo-root /path/to/CMND-MLHOps_DataPreparation/ \
  --pipeline-type preprocessing
```

The module will:
1. Detect hardware and check environment
2. Read trace files from `<repo-root>/reports/traces/preprocessing/` 
3. Estimate parameters from empirical data
4. Generate `adaptive_profile.config`
5. Ask for confirmation before launching (unless `--auto` is passed)
6. Launch Nextflow with the generated profile

### Run training pipeline

```bash
mleader run \
  --pipeline nextflow/training.nf \
  --repo-root /path/to/CMND-MLHOps_DataPreparation/ \
  --pipeline-type training
```
> **Note on the training pipeline**: the module does not configure any
> parameter relevant to the training pipeline. The training pipeline already
> explores different hyperparameter combinations by construction and has no
> per-subject parallelism to size, so there is nothing to tune on the basis of
> the host resources. The `training` pipeline type is supported to leave the
> architecture open to future extensions (for example, resource-aware
> configuration of the training processes). Today, running it through
> `mleader` only provides the environment checks and the supervised launch.

### Pilot run (VRAM profiling for FastSurfer)
 
Nextflow traces do not record VRAM, so the module can measure it directly.
With `--pilot-run` it samples subjects at random from the folders listed in
`params.dataset` (read from `<repo-root>/nextflow.config`), runs FastSurfer
(`--seg_only`) on each of them while sampling the GPU memory, and uses the
result to size `maxforks_fastsurfer`.
 
```bash
mleader check \
  --repo-root /path/to/CMND-MLHOps_DataPreparation/ \
  --pipeline-type preprocessing \
  --pilot-run
```
 
- The sample is drawn with a fixed seed (`--pilot-run-seed`, default `42`), so
  the same dataset and seed always select the same subjects.
- The default sample size is 10 subjects (`--pilot-run-n`).
- The VRAM cost per subject is the **maximum** peak observed across the
  sampled subjects, minus the baseline already in use, multiplied by a safety
  margin of 1.15.
- To profile one specific image instead of a random sample, pass
  `--pilot-run-sample /path/to/subject.nii`.
- The pilot run only affects the FastSurfer parameters. If FreeSurfer is the
  segmenter selected by the throughput comparison, the configuration of the
  pipeline does not change.

### Launch without confirmation prompt

```bash
mleader run \
  --pipeline nextflow/preprocessing.nf \
  --repo-root /path/to/CMND-MLHOps_DataPreparation/ \
  --pipeline-type preprocessing \
  --auto
```

---

## CLI Reference

### `mleader check`

| Flag | Default | Description |
|---|---|---|
| `--repo-root` | required | Root directory of the FTD pipeline repository |
| `--output-config` | `adaptive_profile.config` | Path where the generated config is written |
| `--pipeline-type` | `preprocessing` | Which trace folder to read (`preprocessing` or `training`) |
| `--traces-dir` | `<repo-root>/reports/traces/<pipeline-type>/` | Custom directory for trace TSV files |
| `--compose-file` | auto-detected | Custom docker-compose filename |
| `--pilot-run` | False | Run a VRAM pilot run on a random sample of subjects drawn from `params.dataset` |
| `--pilot-run-n` | `10` | Number of subjects sampled by `--pilot-run` |
| `--pilot-run-seed` | `42` | Seed of the random sampling, so the same subjects are picked on every run |
| `--pilot-run-sample` | None | Path to a sample `.nii` file for VRAM profiling via pilot run |

### `mleader run`

All flags from `check`, plus:

| Flag | Default | Description |
|---|---|---|
| `--pipeline` | required | Path to the `.nf` file to run |
| `--auto` | False | Launch without asking for confirmation |

---

## Parameter estimation logic

The module estimates parameters in the following priority order:

### `brain_segmenter`

```
GPU present + nvidia-container-toolkit installed + FastSurfer requirements met { fastsurfer (cuda) }
otherwise { freesurfer (CPU fallback) }
```

FastSurfer official requirements (source: https://github.com/Deep-MI/FastSurfer):
- CPU: 6+ cores (Intel or AMD)
- RAM: 16 GB system memory
- GPU: NVIDIA 2016+ (compute capability >= 6.0)
- VRAM: 12 GB graphics memory

### `maxForks`

For **freesurfer** (single-threaded, 1 core per subject):

```
If trace data available:
    maxForks = min(floor(ram_available / peak_rss_max), cpu_cores_free)
Else (cold start):
    maxForks = cpu_cores_free
```

For **fastsurfer on GPU**:

```
VRAM measured (warm-up or pilot run):
    maxForks = max(1, min(floor(vram_free / vram_per_subject), cpu_cores_free))
Trace data available (host RAM as proxy):
    maxForks = max(1, min(floor(ram_available / peak_rss_max), cpu_cores_free))
Cold start:
    maxForks = 1
No usable GPU:
    maxForks = 0
```

### `pyradiomics_jobs`

Always hardware-based (PyRadiomics is a single-instance process):

```
pyradiomics_jobs = vcpus - 1
```

### Parameter source

The `source` field in the generated config header describes where the
parameters came from:

| Source | Meaning |
|---|---|
| `trace_empirical` | derived from peak_rss in previous trace TSV files |
| `pilot_run` | derived from VRAM measured during a fastsurfer pilot-run |
| `trace_empirical_ram_proxy` | fastsurfer GPU — used RAM host as VRAM proxy |
| `hardware_conservative` | cold start — no historical data, uses cpu_cores_free |

---

## Trace files

The module reads Nextflow trace TSV files to derive empirical resource estimates.
Default locations:

```
<repo-root>/reports/traces/preprocessing/   ← for preprocessing pipeline
<repo-root>/reports/traces/training/        ← for training pipeline
```

All `.tsv` files in those directories are read and the **maximum `peak_rss`**
across all COMPLETED tasks is used as the worst-case estimate.

> **Note on VRAM**: Nextflow trace files do not record GPU VRAM usage in
> on-premise Docker setups (only available with Fusion + Seqera Platform,
> Nextflow >= 26.03.3-edge). For VRAM estimation, use `--pilot-run-sample`.
> See: https://github.com/nextflow-io/nextflow/issues/4286

---

## Environment variables

The following environment variables must be set before running the module
if using a DSL1 Nextflow pipeline:

```bash
export NXF_SYNTAX_PARSER=v1
```

---

## Generated output

The module generates `adaptive_profile.config` in the current directory
(or at `--output-config` path). Example output:

```groovy
// adaptive_profile.config
// Generated by mleader
// Date:             2026-08-06 14:32:11
// Host:             my-host
// GPU:              NVIDIA H200-35C (35GB VRAM)
// RAM available:    65.5GB
// CPU threads:      32
// CPU cores free:   30
// Parameter source: trace_empirical
//
// Do NOT edit manually.
// Regenerate with: mleader run --repo-root <path>

profiles {
    adaptive_profile {

        // ── Segmenter ─────────────────────────────────────────────────
        params.brain_segmenter   = "freesurfer"

        // ── Parallelism ───────────────────────────────────────────────
        params.maxforks         = 23
        params.pyradiomics_jobs = 31
    }
}
```

The pipeline is then launched as:

```bash
nextflow run nextflow/preprocessing.nf \
  -c adaptive_profile.config \
  -profile adaptive_profile
```

---

## Preflight checks

The module performs the following checks before estimating parameters:

### Critical (block the launch)

| Check | Condition |
|---|---|
| Nextflow installed | `nextflow -v` succeeds |
| FreeSurfer license | `license.txt` present in repo root |
| Docker images | all images from `docker-compose.yml` are built |

### Non-critical (warnings + fallbacks)

| Check | Condition | Fallback |
|---|---|---|
| nvidia-container-toolkit | `which nvidia-container-toolkit` | brain_segmenter = freesurfer |
| vGPU license | `nvidia-smi -q` license status | brain_segmenter = freesurfer |
| FastSurfer requirements | VRAM, RAM, cores, compute capability | brain_segmenter = freesurfer |
| Disk space | < 50 GB free | warning with actionable hints |

---

## License

MIT License — see `LICENSE` file.
