"""CLI module for local-rag-cli."""

import sys
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.table import Table

from local_rag_cli.config import settings
from local_rag_cli.ingest import ingest_directory
from local_rag_cli.rag import chat_loop, query_index

app = typer.Typer(help="Local RAG CLI for Mac M4 Pro")
console = Console()


@app.command()
def health():
    """Check health of vector store and LLM connections."""
    console.print("[bold blue]Checking connections...[/bold blue]\n")

    # Check Vector Store
    vector_store_ok = False
    try:
        if settings.VECTOR_STORE_TYPE == "chromadb":
            import chromadb

            client = chromadb.PersistentClient(path=settings.CHROMADB_PATH)
            # ChromaDB doesn't have a direct health check, so we try to list collections
            client.list_collections()
            vector_store_ok = True
            console.print("[green]✓ ChromaDB: OK[/green]")
        elif settings.VECTOR_STORE_TYPE == "qdrant":
            response = httpx.get(
                f"{settings.QDRANT_URL}/collections",
                timeout=10.0,
                headers={"api-key": settings.QDRANT_API_KEY}
                if settings.QDRANT_API_KEY
                else {},
            )
            if response.status_code == 200:
                vector_store_ok = True
                console.print("[green]✓ Qdrant: OK[/green]")
            else:
                console.print(f"[red]✗ Qdrant: HTTP {response.status_code}[/red]")
        else:
            console.print(
                f"[red]✗ Unknown vector store type: {settings.VECTOR_STORE_TYPE}[/red]"
            )
    except Exception as e:
        console.print(f"[red]✗ Vector Store ({settings.VECTOR_STORE_TYPE}): {e}[/red]")

    # Check LLM
    llm_ok = False
    try:
        response = httpx.get(
            f"{settings.LLM_BASE_URL}/models",
            timeout=10.0,
            headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"}
            if settings.LLM_API_KEY
            else {},
        )
        if response.status_code == 200:
            llm_ok = True
            console.print("[green]✓ LLM: OK[/green]")
        else:
            console.print(f"[red]✗ LLM: HTTP {response.status_code}[/red]")
    except Exception as e:
        console.print(f"[red]✗ LLM: {e}[/red]")

    console.print()
    if vector_store_ok and llm_ok:
        console.print("[bold green]All systems operational![/bold green]")
        sys.exit(0)
    else:
        console.print("[bold red]Some systems are not available.[/bold red]")
        sys.exit(1)


@app.command()
def version():
    """Show version information."""
    from local_rag_cli import __version__

    console.print(f"[bold]local-rag-cli[/bold] version {__version__}")


@app.command()
def ingest(
    paths: list[Path] = typer.Argument(
        ..., help="One or more directory paths to ingest"
    ),
):
    """Ingest documents and images from one or more directories."""
    from local_rag_cli.ingest import ingest_directories

    ingest_directories(paths)


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
):
    """Query the indexed documents."""
    try:
        response = query_index(question)
        console.print("[bold green]Answer:[/bold green]")
        console.print(response)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def chat():
    """Start interactive chat session."""
    chat_loop()


if __name__ == "__main__":
    app()
