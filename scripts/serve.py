"""CLI entrypoint to launch the Finance RAG Web Inspection & Monitoring UI.

Usage:
    poetry run python scripts/serve.py
    poetry run python scripts/serve.py --port 8000 --host 127.0.0.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from src.ui.server import run_server

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Finance RAG Web Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
    console.print(f"[bold green]⚡ Finance RAG — Web Inspection & Monitoring Dashboard[/bold green]")
    console.print(f"Server running at: [bold underline blue]{url}[/bold underline blue]")
    console.print("[dim]Press Ctrl+C to stop the server[/dim]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

    try:
        run_server(host=args.host, port=args.port)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user.[/yellow]")


if __name__ == "__main__":
    main()
