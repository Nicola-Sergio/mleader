"""
Preflight checks — Monitor component.
Verifica i vincoli critici e non critici prima del lancio della pipeline.
Vincoli critici: bloccano il lancio.
Vincoli non critici: generano warning e attivano fallback.
"""

from __future__ import annotations

from .hardware import HardwareProfile


# Versione minima di Nextflow supportata (DSL1 compatibile)
NEXTFLOW_MIN_VERSION = (22, 0, 0)

# VRAM minima per lanciare fastsurfer su GPU con almeno 1 soggetto
FASTSURFER_MIN_VRAM_GB = 6.0


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Converte stringa versione in tupla di interi per confronto."""
    try:
        parts = version_str.split(".")
        return tuple(int(p) for p in parts[:3])
    except (ValueError, AttributeError):
        return (0, 0, 0)


def run_preflight_checks(profile: HardwareProfile) -> None:
    """
    Esegue tutti i preflight check e popola:
    - profile.preflight_errors   (critici — bloccano il lancio)
    - profile.preflight_warnings (non critici — attivano fallback)
    - profile.fallbacks          (decisioni di fallback applicate)
    - profile.preflight_passed   (True se nessun errore critico)
    """
    errors = []
    warnings = []
    fallbacks = {}

    # ── VINCOLI CRITICI ──────────────────────────────────────────────

    # 1. Nextflow installato
    if profile.nextflow_version is None:
        errors.append(
            "Nextflow non trovato. "
            "Installa Nextflow: https://www.nextflow.io/ "
            "(versione raccomandata: 24.10.5)"
        )

    # 2. Licenza FreeSurfer presente
    if not profile.fs_license_present:
        errors.append(
            "File license.txt non trovato nella root del repo. "
            "Ottieni la licenza FreeSurfer: "
            "https://surfer.nmr.mgh.harvard.edu/fswiki/License "
            "e posizionala come license.txt nella root del progetto."
        )

    # 3. Container obbligatori buildati
    if profile.containers_missing:
        missing_str = ", ".join(profile.containers_missing)
        errors.append(
            f"Container Docker mancanti: {missing_str}. "
            "Esegui 'docker compose build' nella root del repo FTD."
        )

    # 4. Docker disponibile (check indiretto: se containers_missing è popolato
    #    significa che docker è raggiungibile; se è vuoto ed è andato in errore
    #    è già gestito sopra)

    # ── VINCOLI NON CRITICI (fallback) ───────────────────────────────

    # 5. GPU disponibile ma nvidia-container-toolkit non installato
    if profile.gpu is not None and not profile.docker_gpu_runtime:
        warnings.append(
            "GPU rilevata ma nvidia-container-toolkit non è installato "
            "(verificato con 'which nvidia-container-toolkit'). "
            "Installazione: "
            "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html "
            "Fallback: brain_segmenter = freesurfer (CPU)."
        )
        fallbacks["brain_segmenter"] = "freesurfer"
        fallbacks["fastsurfer_device"] = None

    # 5b. GPU disponibile, toolkit installato, ma licenza vGPU mancante/scaduta
    if (
        profile.gpu is not None
        and profile.docker_gpu_runtime
        and profile.gpu.is_vgpu
        and profile.gpu.vgpu_license_status
        and "unlicensed" in profile.gpu.vgpu_license_status.lower()
    ):
        warnings.append(
            f"GPU virtualizzata (vGPU) rilevata ma non licenziata "
            f"(License Status: {profile.gpu.vgpu_license_status}). "
            "Senza licenza valida i container con accesso GPU falliranno "
            "a runtime con errori opachi. Verifica configurazione del "
            "license server o valuta il passaggio a MIG passthrough. "
            "Fallback: brain_segmenter = freesurfer (CPU)."
        )
        fallbacks["brain_segmenter"] = "freesurfer"
        fallbacks["fastsurfer_device"] = None

    # 6. GPU assente — nessun errore, solo fallback silenzioso
    if profile.gpu is None:
        fallbacks["brain_segmenter"] = "freesurfer"
        fallbacks["fastsurfer_device"] = None

    # 7. VRAM disponibile insufficiente per fastsurfer
    if (
        profile.gpu is not None
        and profile.docker_gpu_runtime
        and profile.gpu.vram_free_gb < FASTSURFER_MIN_VRAM_GB
    ):
        warnings.append(
            f"VRAM disponibile ({profile.gpu.vram_free_gb:.1f}GB) "
            f"inferiore al minimo richiesto da FastSurfer ({FASTSURFER_MIN_VRAM_GB}GB). "
            "Fallback: brain_segmenter = freesurfer (CPU)."
        )
        fallbacks["brain_segmenter"] = "freesurfer"
        fallbacks["fastsurfer_device"] = None

    # 8. RAM disponibile bassa
    if profile.ram_available_gb < 16.0:
        warnings.append(
            f"RAM disponibile bassa ({profile.ram_available_gb:.1f}GB). "
            "maxForks verrà impostato conservativamente."
        )

    # 9. Spazio disco basso sul workDir
    if profile.disk_free_gb < 50.0:
        warnings.append(
            f"Spazio disco disponibile basso ({profile.disk_free_gb:.1f}GB). "
            "La pipeline di preprocessing produce dati volumetrici pesanti. "
            "Si raccomandano almeno 50GB liberi."
        )

    # ── RISULTATO FINALE ─────────────────────────────────────────────
    profile.preflight_errors = errors
    profile.preflight_warnings = warnings
    profile.fallbacks = fallbacks
    profile.preflight_passed = len(errors) == 0
