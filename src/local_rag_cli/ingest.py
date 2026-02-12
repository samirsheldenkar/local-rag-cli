"""Ingestion pipeline for documents and images."""

from pathlib import Path
from typing import List

from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from local_rag_cli.embeddings import OpenCLIPEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from local_rag_cli.config import settings
from local_rag_cli.storage import (
    ensure_collections_exist,
    get_image_vector_store,
    get_text_vector_store,
)

console = Console()


def get_embedding_models():
    """Get embedding models for text and images."""
    text_embedding = HuggingFaceEmbedding(
        model_name=settings.TEXT_EMBEDDING_MODEL,
    )
    image_embedding = OpenCLIPEmbedding(
        model_name=settings.IMAGE_EMBEDDING_MODEL,
        pretrained=settings.IMAGE_EMBEDDING_PRETRAINED,
    )
    return text_embedding, image_embedding


def ingest_directory(path: Path) -> None:
    """Ingest all files from a directory."""
    if not path.exists():
        console.print(f"[red]Path does not exist: {path}[/red]")
        return

    if not path.is_dir():
        console.print(f"[red]Path is not a directory: {path}[/red]")
        return

    # Ensure collections exist
    ensure_collections_exist()

    # Get embedding models
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading embedding models...", total=None)
        text_embedding, image_embedding = get_embedding_models()
        progress.update(task, completed=True)

        # Load documents
        task = progress.add_task(f"Loading documents from {path}...", total=None)
        reader = SimpleDirectoryReader(
            input_dir=path,
            filename_as_id=True,
            recursive=True,
        )
        documents = reader.load_data()
        progress.update(task, completed=True)

        if not documents:
            console.print("[yellow]No documents found in directory.[/yellow]")
            return

        console.print(f"[green]Loaded {len(documents)} documents[/green]")

        # Separate text and image documents
        text_docs = []
        image_docs = []
        for doc in documents:
            if doc.metadata.get("file_type", "").startswith("image/"):
                image_docs.append(doc)
            else:
                text_docs.append(doc)

        console.print(f"[blue]Text documents: {len(text_docs)}[/blue]")
        console.print(f"[blue]Image documents: {len(image_docs)}[/blue]")

        # Ingest text documents
        if text_docs:
            task = progress.add_task("Processing text documents...", total=None)
            text_store = get_text_vector_store()

            pipeline = IngestionPipeline(
                transformations=[
                    SentenceSplitter(chunk_size=1024, chunk_overlap=200),
                    text_embedding,
                ],
                vector_store=text_store,
            )
            nodes = pipeline.run(documents=text_docs)
            progress.update(task, completed=True)
            console.print(f"[green]Indexed {len(nodes)} text chunks[/green]")

        # Ingest image documents
        if image_docs:
            task = progress.add_task("Processing image documents...", total=None)
            image_store = get_image_vector_store()

            pipeline = IngestionPipeline(
                transformations=[
                    image_embedding,
                ],
                vector_store=image_store,
            )
            nodes = pipeline.run(documents=image_docs)
            progress.update(task, completed=True)
            console.print(f"[green]Indexed {len(nodes)} images[/green]")

    console.print("[bold green]Ingestion complete![/bold green]")


def ingest_directories(paths: list[Path]) -> None:
    """Ingest all files from multiple directories."""
    # Ensure collections exist
    ensure_collections_exist()

    # Validate all paths first
    valid_paths = []
    for path in paths:
        if not path.exists():
            console.print(f"[red]Path does not exist: {path}[/red]")
            continue
        if not path.is_dir():
            console.print(f"[red]Path is not a directory: {path}[/red]")
            continue
        valid_paths.append(path)

    if not valid_paths:
        console.print("[red]No valid directories to ingest.[/red]")
        return

    # Get embedding models
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading embedding models...", total=None)
        text_embedding, image_embedding = get_embedding_models()
        progress.update(task, completed=True)

        all_text_docs = []
        all_image_docs = []

        # Load documents from all directories
        for path in valid_paths:
            task = progress.add_task(f"Loading documents from {path}...", total=None)
            reader = SimpleDirectoryReader(
                input_dir=path,
                filename_as_id=True,
                recursive=True,
            )
            documents = reader.load_data()
            progress.update(task, completed=True)

            # Separate text and image documents
            for doc in documents:
                if doc.metadata.get("file_type", "").startswith("image/"):
                    all_image_docs.append(doc)
                else:
                    all_text_docs.append(doc)

        total_docs = len(all_text_docs) + len(all_image_docs)
        if not total_docs:
            console.print("[yellow]No documents found in any directory.[/yellow]")
            return

        console.print(f"[green]Loaded {total_docs} documents total[/green]")
        console.print(f"[blue]Text documents: {len(all_text_docs)}[/blue]")
        console.print(f"[blue]Image documents: {len(all_image_docs)}[/blue]")

        # Ingest text documents
        if all_text_docs:
            task = progress.add_task("Processing text documents...", total=None)
            text_store = get_text_vector_store()

            pipeline = IngestionPipeline(
                transformations=[
                    SentenceSplitter(chunk_size=1024, chunk_overlap=200),
                    text_embedding,
                ],
                vector_store=text_store,
            )
            nodes = pipeline.run(documents=all_text_docs)
            progress.update(task, completed=True)
            console.print(f"[green]Indexed {len(nodes)} text chunks[/green]")

        # Ingest image documents
        if all_image_docs:
            task = progress.add_task("Processing image documents...", total=None)
            image_store = get_image_vector_store()

            pipeline = IngestionPipeline(
                transformations=[
                    image_embedding,
                ],
                vector_store=image_store,
            )
            nodes = pipeline.run(documents=all_image_docs)
            progress.update(task, completed=True)
            console.print(f"[green]Indexed {len(nodes)} images[/green]")

    console.print("[bold green]All directories ingested successfully![/bold green]")
