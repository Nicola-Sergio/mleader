"""
Plan — terza fase del loop MAPE-K.
Genera il file adaptive_profile.config.
"""

from .config_generator import generate_config


def run_plan(plan, profile, output_path: str = "adaptive_profile.config") -> str:
    """
    Esegue la fase Plan: genera adaptive_profile.config.
    Restituisce il path del file generato.
    """
    print("[Plan] Generazione configurazione adattiva...")
    config_path = generate_config(plan, profile, output_path)
    return config_path
