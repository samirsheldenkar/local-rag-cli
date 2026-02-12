"""RAG query engine module."""

from typing import List, Dict, Any

from llama_index.core.llms import ChatMessage
from llama_index.core.base.response.schema import Response
from llama_index.llms.ollama import Ollama
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from local_rag_cli.config import settings
from local_rag_cli.storage import get_multimodal_index

console = Console()

# Maximum characters to show in source excerpts
EXCERPT_MAX_CHARS = 200


def get_llm():
    """Get LLM instance configured for local inference."""
    return Ollama(
        model=settings.LLM_MODEL,
        base_url=settings.LLM_BASE_URL,
        request_timeout=settings.REQUEST_TIMEOUT,
    )


def query_index(question: str) -> Response:
    """Query the multimodal index with a question.

    Returns the full LlamaIndex Response object, which includes
    source_nodes with metadata and relevance scores.
    """
    try:
        # Get the multimodal index
        index = get_multimodal_index()

        # Get LLM
        llm = get_llm()

        # Create query engine
        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=5,
        )

        # Execute query
        response = query_engine.query(question)

        return response
    except Exception as e:
        console.print(f"[red]Error querying index: {e}[/red]")
        raise


def format_sources(response: Response) -> List[Dict[str, Any]]:
    """Extract and format source references from a response.

    Returns a list of dicts with keys: file_name, score, excerpt.
    """
    sources = []
    seen_files = set()

    for node_with_score in response.source_nodes:
        metadata = node_with_score.node.metadata
        file_name = metadata.get("file_name", metadata.get("file_path", "Unknown"))
        file_path = metadata.get("file_path", "")
        score = node_with_score.score

        # Get text excerpt
        text = node_with_score.node.get_content()
        excerpt = text[:EXCERPT_MAX_CHARS].strip()
        if len(text) > EXCERPT_MAX_CHARS:
            excerpt += "..."

        # Deduplicate by file name but keep best score
        if file_name not in seen_files:
            seen_files.add(file_name)
            sources.append({
                "file_name": file_name,
                "file_path": file_path,
                "score": score,
                "excerpt": excerpt,
            })

    return sources


def print_sources(response: Response) -> None:
    """Print formatted source references from a response."""
    sources = format_sources(response)

    if not sources:
        console.print("[dim]No source references found.[/dim]")
        return

    table = Table(
        title="Sources",
        show_header=True,
        header_style="bold cyan",
        title_style="bold cyan",
        expand=True,
    )
    table.add_column("File", style="green", no_wrap=True, ratio=1)
    table.add_column("Score", style="yellow", justify="right", width=8)
    table.add_column("Excerpt", ratio=3)

    for source in sources:
        score_str = f"{source['score']:.3f}" if source["score"] is not None else "N/A"
        table.add_row(
            source["file_name"],
            score_str,
            source["excerpt"],
        )

    console.print()
    console.print(table)


def chat_loop():
    """Interactive chat loop with the RAG system."""
    console.print("[bold green]Local RAG CLI Chat[/bold green]")
    console.print("[dim]Type 'exit' to quit, 'help' for commands[/dim]\n")

    llm = get_llm()

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.lower() == "help":
                console.print("[bold]Commands:[/bold]")
                console.print("  [green]exit/quit[/green] - Exit the chat")
                console.print("  [green]help[/green] - Show this help message")
                console.print("")
                continue

            # Query the index
            response = query_index(user_input)

            console.print("[bold green]Assistant:[/bold green]")
            console.print(Markdown(str(response)))
            print_sources(response)
            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]\n")
