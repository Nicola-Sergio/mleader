"""
Plan — third phase of the MAPE-K loop.
Generates the adaptive_profile.config file.
"""

from .config_generator import generate_config


def run_plan(plan, profile, output_path: str = "adaptive_profile.config") -> str:
    """
    Executes the Plan phase: generates adaptive_profile.config.
    Returns the path of the generated file.
    """
    print("[Plan] Generating adaptive configuration...")
    config_path = generate_config(plan, profile, output_path)
    return config_path
