"""
CLI entry point — ftd-orchestrator.
Orchestrates the entire MAPE-K loop.
"""

from __future__ import annotations

import argparse
import sys

from .monitor import run_monitor
from .analyze import run_analyze
from .plan import run_plan
from .execute import run_execute


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

    print(f"  maxForks:          {plan.maxforks_segmenter} parallel subjects")
    if plan.fastsurfer_threads:
        print(f"  fastsurfer_threads:{plan.fastsurfer_threads} threads per instance")
    print(f"  pyradiomics_jobs:  {plan.pyradiomics_jobs} parallel jobs")
    print(f"  Parameter Source:   {plan.source}")

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
    parser = argparse.ArgumentParser(
        prog="ftd-orchestrator",
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
        "--dry-run-sample",
        default=None,
        help="Path to a sample .nii file for VRAM profiling via dry-run (optional)",
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
        help="Pipeline type — determines which trace folder to read (default: preprocessing)",
    )
    run_parser.add_argument(
    "--compose-file",
    default=None,
    help="Docker-compose file name (default: searching for docker-compose.yml "
         "oe docker-compose.yaml in the repo root)",
)

    # ── Comando: check ────────────────────────────────────────────────
    check_parser = subparsers.add_parser(
        "check",
        help="Executes only Monitor + Analyze and shows the summary without launching",
    )
    check_parser.add_argument("--repo-root", default=".")
    check_parser.add_argument("--output-config", default="adaptive_profile.config")
    check_parser.add_argument(
        "--compose-file",
        default=None,
        help="Docker-compose file name (default: searching for docker-compose.yml "
            "oe docker-compose.yaml in the repo root)",
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
        help="Pipeline type — determines which trace folder to read (default: preprocessing)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # ── Validazione repo_root ─────────────────────────────────────────
    from pathlib import Path
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

    # ── MONITOR ───────────────────────────────────────────────────────
    profile = run_monitor(
        repo_root=args.repo_root,
        compose_file=getattr(args, "compose_file", None)
        )

    if not profile.preflight_passed:
        print("\n[ERRORE] Failed preflight checks — Unable to proceed:\n")
        for err in profile.preflight_errors:
            print(f"  ✗  {err}\n")
        sys.exit(1)

    # ── ANALYZE ───────────────────────────────────────────────────────
    dry_run_sample = getattr(args, "dry_run_sample", None)
    plan = run_analyze(
        profile,
        repo_root=args.repo_root,
        pipeline=getattr(args, "pipeline_type", "preprocessing"),
        dry_run=bool(dry_run_sample),
        sample_nii=dry_run_sample,
        license_path=f"{args.repo_root}/license.txt",
        custom_traces_dir=getattr(args, "traces_dir", None),
    )

    # ── PLAN ──────────────────────────────────────────────────────────
    config_path = run_plan(plan, profile, output_path=args.output_config)

    # ── Riepilogo ─────────────────────────────────────────────────────
    _print_summary(profile, plan, config_path)

    if args.command == "check":
        print("[check] Only analysis requested — pipeline not launched.")
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
    auto=args.auto,
    )

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()