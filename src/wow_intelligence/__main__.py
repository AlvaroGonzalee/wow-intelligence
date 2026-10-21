"""Parse command-line arguments and launch the terminal chat."""

import argparse
from importlib.metadata import version

from wow_intelligence.chat import run_chat


def main() -> None:
    """Show help or version information when requested; otherwise start a session."""
    parser = argparse.ArgumentParser(
        prog="wow-intelligence",
        description="Experimental terminal chat using local Ollama.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"WoW Intelligence {version('wow-intelligence')}",
    )
    parser.parse_args()
    run_chat()


if __name__ == "__main__":
    main()
