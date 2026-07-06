"""
CLI entry point — ftd-orchestrator.
Coordina il loop MAPE-K completo.
"""

from __future__ import annotations

import argparse
import sys

from .monitor import run_monitor
from .analyze import run_analyze
from .plan import run_plan
from .execute import run_execute


def _print_summary(profile, plan, config_path: str) -> None:
    """Stampa un riepilogo human-readable prima del lancio."""
    print("\n" + "=" * 60)
    print("  RIEPILOGO CONFIGURAZIONE ADATTIVA")
    print("=" * 60)

    print(f"\n  Host:              {profile.cpu_cores} core fisici, {profile.cpu_threads} thread")
    print(f"  RAM disponibile:   {profile.ram_available_gb:.1f} GB")

    if profile.gpu:
        print(f"  GPU:               {profile.gpu.name}")
        print(f"  VRAM disponibile:  {profile.gpu.vram_free_gb:.1f} GB")
        print(f"  Docker GPU:        {'✓' if profile.docker_gpu_runtime else '✗'}")
    else:
        print("  GPU:               non disponibile")

    print(f"\n  Segmentatore:      {plan.brain_segmenter}", end="")
    if plan.fastsurfer_device:
        print(f" ({plan.fastsurfer_device})", end="")
    print()

    print(f"  maxForks:          {plan.maxforks_segmenter} soggetti in parallelo")
    if plan.fastsurfer_threads:
        print(f"  fastsurfer_threads:{plan.fastsurfer_threads} thread per istanza")
    print(f"  pyradiomics_jobs:  {plan.pyradiomics_jobs} job paralleli")
    print(f"  Fonte parametri:   {plan.source}")

    if profile.preflight_warnings:
        print("\n  AVVISI:")
        for w in profile.preflight_warnings:
            print(f"  ⚠  {w}")

    if profile.fallbacks:
        print("\n  FALLBACK APPLICATI:")
        for k, v in profile.fallbacks.items():
            print(f"  →  {k} = {v}")

    print(f"\n  Config generato:   {config_path}")
    print("=" * 60 + "\n")


def _confirm_launch(pipeline: str) -> bool:
    """Chiede conferma all'utente prima di lanciare."""
    try:
        answer = input(f"Lanciare '{pipeline}' con il profilo adattivo? [y/N] ").strip().lower()
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
        help="Esegue il loop MAPE-K completo e lancia la pipeline",
    )
    run_parser.add_argument(
        "--pipeline",
        required=True,
        help="Path al file .nf da lanciare (es. nextflow/preprocessing.nf)",
    )
    run_parser.add_argument(
        "--repo-root",
        default=".",
        help="Root del repo FTD (default: directory corrente)",
    )
    run_parser.add_argument(
        "--output-config",
        default="adaptive_profile.config",
        help="Path dove scrivere il config generato",
    )
    run_parser.add_argument(
        "--auto",
        action="store_true",
        help="Lancia la pipeline senza chiedere conferma",
    )
    run_parser.add_argument(
        "--dry-run-sample",
        default=None,
        help="Path al file .nii campione per il profiling VRAM (opzionale)",
    )

    # ── Comando: check ────────────────────────────────────────────────
    check_parser = subparsers.add_parser(
        "check",
        help="Esegue solo Monitor + Analyze e mostra il riepilogo senza lanciare",
    )
    check_parser.add_argument("--repo-root", default=".")
    check_parser.add_argument("--output-config", default="adaptive_profile.config")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # ── Validazione repo_root ─────────────────────────────────────────
    from pathlib import Path
    repo_root_path = Path(args.repo_root).resolve()
    if not repo_root_path.exists():
        print(f"\n[ERRORE] Il path specificato non esiste: {repo_root_path}")
        print("Verifica il valore di --repo-root e riprova.")
        sys.exit(1)
    if not repo_root_path.is_dir():
        print(f"\n[ERRORE] Il path specificato non è una directory: {repo_root_path}")
        sys.exit(1)
    # Normalizza args.repo_root al path assoluto risolto
    args.repo_root = str(repo_root_path)

    # ── MONITOR ───────────────────────────────────────────────────────
    profile = run_monitor(repo_root=args.repo_root)

    if not profile.preflight_passed:
        print("\n[ERRORE] Preflight check falliti — impossibile procedere:\n")
        for err in profile.preflight_errors:
            print(f"  ✗  {err}\n")
        sys.exit(1)

    # ── ANALYZE ───────────────────────────────────────────────────────
    dry_run_sample = getattr(args, "dry_run_sample", None)
    plan = run_analyze(
        profile,
        dry_run=bool(dry_run_sample),
        sample_nii=dry_run_sample,
        license_path=f"{args.repo_root}/license.txt",
    )

    # ── PLAN ──────────────────────────────────────────────────────────
    config_path = run_plan(plan, profile, output_path=args.output_config)

    # ── Riepilogo ─────────────────────────────────────────────────────
    _print_summary(profile, plan, config_path)

    if args.command == "check":
        print("[check] Solo analisi richiesta — pipeline non lanciata.")
        sys.exit(0)

    # ── EXECUTE ───────────────────────────────────────────────────────
    if not args.auto:
        if not _confirm_launch(args.pipeline):
            print("Lancio annullato.")
            sys.exit(0)

    result = run_execute(
        pipeline=args.pipeline,
        config_path=config_path,
        auto=args.auto,
    )

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
