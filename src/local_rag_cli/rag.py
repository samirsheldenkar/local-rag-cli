"""RAG query engine module."""

from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown

from local_rag_cli.config import settings
from local_rag_cli.storage import get_multimodal_index

console = Console()


def get_llm():
    """Get LLM instance configured for local inference."""
    return OpenAI(
        model=settings.LLM_MODEL,
        api_base=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY or "not-needed",
        timeout=settings.REQUEST_TIMEOUT,
    )


def query_index(question: str) -> str:
    """Query the multimodal index with a question."""
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

        return str(response)
    except Exception as e:
        console.print(f"[red]Error querying index: {e}[/red]")
        raise


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
            console.print(Markdown(response))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]\n")
