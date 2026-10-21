"""Parse command-line arguments and launch the terminal chat."""

import argparse
from importlib.metadata import version

from wow_intelligence.chat import run_chat
from wow_intelligence.config import OllamaConfig


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
    try:
        config = OllamaConfig.from_env()
    except ValueError as exc:
        parser.error(str(exc))
    run_chat(config)


if __name__ == "__main__":
    main()
