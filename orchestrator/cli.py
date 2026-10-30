"""
CLI entry point - mleader.
Orchestrates the entire MAPE-K loop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .banner import print_banner
from .monitor import run_monitor
from .analyze import run_analyze
from .analyze.pilot_run import read_dataset_dirs, sample_subjects
from .plan import run_plan
from .execute import run_execute


def _resolve_output_config(repo_root: str, output_config: str) -> str:
    """
    The adaptive config is consumed by Nextflow (launched with cwd=repo_root)
    and read/rewritten by the retry logic; therefore, if --output-config is
    a relative path, it must be written INSIDE repo_root, not in the folder from
    which MLEADeR is launched. An absolute --output-config path is respected as-is.
    """
    p = Path(output_config)
    if p.is_absolute():
        return str(p)
    return str(Path(repo_root) / p)


def _resolve_pipeline_file(pipeline: str | None, pipeline_type: str) -> str:
    """
    Return the .nf file the Monitor inspects to detect the Nextflow DSL.

    An explicit --pipeline always wins (run). Commands that do not take it
    (check) fall back to the conventional nextflow/<pipeline_type>.nf.
    """
    return pipeline or f"nextflow/{pipeline_type}.nf"


def _print_summary(profile, plan, config_path: str) -> None:
    """Prints a human-readable summary before launching."""
    print("\n" + "=" * 60)
    print("  Summary of Adaptive Configuration")
    print("=" * 60)

    print(f"\n  Host:              {profile.cpu_cores} physical cores, {profile.cpu_threads} threads")
    print(f"  Available RAM:     {profile.ram_available_gb:.1f} GB")

    if profile.gpu:
        print(f"  GPU:               {profile.gpu.name}")
        print(f"  Available VRAM:    {profile.gpu.vram_free_gb:.1f} GB")
        print(f"  Docker GPU:        {'✓' if profile.docker_gpu_runtime else '✗'}")
    else:
        print("  GPU:               not available")

    print(f"\n  Segmentator:       {plan.brain_segmenter}", end="")
    if plan.fastsurfer_device:
        print(f" ({plan.fastsurfer_device})", end="")
    print()

    print(f"  maxForks FreeSurfer:  {plan.maxforks_freesurfer} parallel subjects")
    print(f"  maxForks FastSurfer:  {plan.maxforks_fastsurfer} parallel subjects"
          + (" (not available)" if plan.maxforks_fastsurfer == 0 else ""))
    if plan.fastsurfer_threads:
        print(f"  fastsurfer_threads:   {plan.fastsurfer_threads} threads per instance")
    print(f"  pyradiomics_jobs:     {plan.pyradiomics_jobs} parallel jobs")
    print(f"  Parameter Source:     {plan.source}")

    if profile.preflight_warnings:
        print("\n  WARNINGS:")
        for w in profile.preflight_warnings:
            print(f"  ⚠  {w}")

    if profile.fallbacks:
        print("\n  FALLBACKS APPLIED:")
        for k, v in profile.fallbacks.items():
            print(f"  →  {k} = {v}")

    print(f"\n  Config generated:   {config_path}")
    print("=" * 60 + "\n")


def _confirm_launch(pipeline: str) -> bool:
    """Ask the user for confirmation before launching."""
    try:
        answer = input(f"Launch '{pipeline}' with the adaptive profile? [y/N] ").strip().lower()
        return answer == "y"
    except (EOFError, KeyboardInterrupt):
        return False

def main() -> None:
    """Entry point: runs the CLI and turns Ctrl+C into a clean exit."""
    try:
        _main()
    except KeyboardInterrupt:
        print("\n[mleader] Interrupted by user.")
        sys.exit(130)

def _main() -> None:
    # Printed before argparse so it also appears with --help.
    print_banner()

    parser = argparse.ArgumentParser(
        prog="mleader",
        description="Infrastructure-aware orchestrator per pipeline MLOps FTD",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── Comando: run ──────────────────────────────────────────────────
    run_parser = subparsers.add_parser(
        "run",
        help="Executes the complete MAPE-K loop and launches the pipeline",
    )
    run_parser.add_argument(
        "--pipeline",
        required=True,
        help="Path to the .nf file to launch (e.g., nextflow/preprocessing.nf)",
    )
    run_parser.add_argument(
        "--repo-root",
        default=".",
        help="Root of the FTD repo (default: current directory)",
    )
    run_parser.add_argument(
        "--output-config",
        default="adaptive_profile.config",
        help="Path where to write the generated config",
    )
    run_parser.add_argument(
        "--auto",
        action="store_true",
        help="Launch the pipeline without asking for confirmation",
    )
    run_parser.add_argument(
        "--pilot-run-sample",
        default=None,
        help="Path to a single .nii file to use for the pilot run instead of a random sample",
    )
    run_parser.add_argument(
        "--traces-dir",
        default=None,
        help="Custom directory containing trace TSV files "
             "(default: <repo-root>/reports/traces/<pipeline>/)",
    )
    run_parser.add_argument(
        "--pipeline-type",
        default="preprocessing",
        choices=["preprocessing", "training"],
        help="Pipeline type - determines which trace folder to read (default: preprocessing)",
    )
    run_parser.add_argument(
        "--compose-file",
        default=None,
        help="Docker-compose file name (default: searching for docker-compose.yml "
             "or docker-compose.yaml in the repo root)",
    )

    # ── Comando: check ────────────────────────────────────────────────
    check_parser = subparsers.add_parser(
        "check",
        help="Executes only Monitor + Analyze and shows the summary without launching",
    )
    check_parser.add_argument("--repo-root", default=".")
    check_parser.add_argument("--output-config", default="adaptive_profile.config")
    check_parser.add_argument(
        "--pilot-run-sample",
        default=None,
        help="Path to a single .nii file to use for the pilot run instead of a random sample",
    )
    check_parser.add_argument(
        "--compose-file",
        default=None,
        help="Docker-compose file name (default: searching for docker-compose.yml "
             "or docker-compose.yaml in the repo root)",
    )
    check_parser.add_argument(
        "--traces-dir",
        default=None,
        help="Custom directory containing trace TSV files "
             "(default: <repo-root>/reports/traces/<pipeline>/)",
    )
    check_parser.add_argument(
        "--pipeline-type",
        default="preprocessing",
        choices=["preprocessing", "training"],
        help="Pipeline type - determines which trace folder to read (default: preprocessing)",
    )

    # ── Pilot run (flag comuni a run e check) ─────────────────────────
    for p in (run_parser, check_parser):
        p.add_argument(
            "--pilot-run",
            action="store_true",
            help="Run a VRAM pilot run on a random sample of subjects (FastSurfer --seg_only)",
        )
        p.add_argument(
            "--pilot-run-n", type=int, default=10,
            help="Number of subjects sampled for the pilot run (default: 10)",
        )
        p.add_argument(
            "--pilot-run-seed", type=int, default=42,
            help="Seed of the random sampling (default: 42)",
        )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # ── Validazione repo_root ─────────────────────────────────────────
    repo_root_path = Path(args.repo_root).resolve()
    if not repo_root_path.exists():
        print(f"\n[ERRORE] The specified path does not exist: {repo_root_path}")
        print("Verify the value of --repo-root and try again.")
        sys.exit(1)
    if not repo_root_path.is_dir():
        print(f"\n[ERRORE] The specified path is not a directory: {repo_root_path}")
        sys.exit(1)
    # Normalizza args.repo_root al path assoluto risolto
    args.repo_root = str(repo_root_path)

    pipeline_type = getattr(args, "pipeline_type", "preprocessing")

    # ── MONITOR ───────────────────────────────────────────────────────
    pipeline_file = _resolve_pipeline_file(getattr(args, "pipeline", None), pipeline_type)
    profile = run_monitor(
        repo_root=args.repo_root,
        compose_file=getattr(args, "compose_file", None),
        pipeline=pipeline_file,
    )

    if not profile.preflight_passed:
        print("\n[ERRORE] Failed preflight checks - Unable to proceed:\n")
        for err in profile.preflight_errors:
            print(f"  ✗  {err}\n")
        sys.exit(1)

    # ── PILOT RUN: scelta dei soggetti ────────────────────────────────
    pilot_run_sample = getattr(args, "pilot_run_sample", None)
    if getattr(args, "pilot_run", False):
        dataset_dirs = read_dataset_dirs(args.repo_root)
        subjects = sample_subjects(
            dataset_dirs,
            n=args.pilot_run_n,
            seed=args.pilot_run_seed,
            repo_root=args.repo_root,
        )
        if subjects:
            print(f"[PilotRun] Sampled {len(subjects)} subject(s) with seed={args.pilot_run_seed}:")
            for s in subjects:
                print(f"  - {s}")
            pilot_run_sample = subjects
        else:
            print(f"[PilotRun] No subjects found under params.dataset: {dataset_dirs}")

    # ── ANALYZE ───────────────────────────────────────────────────────
    plan = run_analyze(
        profile,
        repo_root=args.repo_root,
        pipeline=pipeline_type,
        pilot_run=bool(pilot_run_sample),
        sample_nii=pilot_run_sample,
        license_path=f"{args.repo_root}/license.txt",
        custom_traces_dir=getattr(args, "traces_dir", None),
    )

    # ── PLAN ──────────────────────────────────────────────────────────
    output_config = _resolve_output_config(args.repo_root, args.output_config)
    config_path = run_plan(plan, profile, output_path=output_config)

    # ── Riepilogo ─────────────────────────────────────────────────────
    _print_summary(profile, plan, config_path)

    if args.command == "check":
        print("[check] Only analysis requested - pipeline not launched.")
        sys.exit(0)

    # ── EXECUTE ───────────────────────────────────────────────────────
    if not args.auto:
        if not _confirm_launch(args.pipeline):
            print("Launch cancelled.")
            sys.exit(0)

    result = run_execute(
        pipeline=args.pipeline,
        config_path=config_path,
        repo_root=args.repo_root,
        profile=profile,
        pipeline_type=pipeline_type,
        auto=args.auto,
    )

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()