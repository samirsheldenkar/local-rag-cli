"""Local RAG CLI package."""

from local_rag_cli.cli import app
from local_rag_cli.config import Settings, settings
from local_rag_cli.ingest import ingest_directories

__version__ = "0.1.0"
__all__ = ["Settings", "settings", "app", "ingest_directories"]
