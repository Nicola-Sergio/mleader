"""ASCII banner shown when the CLI starts."""

from __future__ import annotations

import sys

BANNER = """\
███╗   ███╗██╗     ███████╗ █████╗ ██████╗ ███████╗██████╗
████╗ ████║██║     ██╔════╝██╔══██╗██╔══██╗██╔════╝██╔══██╗
██╔████╔██║██║     █████╗  ███████║██║  ██║█████╗  ██████╔╝
██║╚██╔╝██║██║     ██╔══╝  ██╔══██║██║  ██║██╔══╝  ██╔══██╗
██║ ╚═╝ ██║███████╗███████╗██║  ██║██████╔╝███████╗██║  ██║
╚═╝     ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝
"""


def print_banner() -> None:
    """Print the banner only on an interactive terminal that can show it."""
    if not sys.stdout.isatty():
        return
    try:
        print("\n" + BANNER)
    except UnicodeEncodeError:
        # Terminals without UTF-8 (some Windows consoles): fall back to plain text.
        print("MLEADeR\n")