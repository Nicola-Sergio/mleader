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