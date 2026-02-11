"""Local RAG CLI configuration module."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Vector Store Configuration
    VECTOR_STORE_TYPE: str = "chromadb"  # Options: "chromadb", "qdrant"

    # ChromaDB Configuration (persistent mode)
    CHROMADB_PATH: str = "./chromadb_data"

    # Qdrant Configuration
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None

    # LLM Configuration (OpenAI-compatible)
    LLM_BASE_URL: str = "http://localhost:1234/v1"
    LLM_API_KEY: str | None = None
    LLM_MODEL: str = "local-model"

    # Embedding Models
    TEXT_EMBEDDING_MODEL: str = "BAAI/bge-m3"
    IMAGE_EMBEDDING_MODEL: str = "ViT-B/32"

    # Request Configuration
    REQUEST_TIMEOUT: float = 600.0  # 10 minutes for slow local inference

    # Logging
    LOG_LEVEL: str = "INFO"


# Global settings instance
settings = Settings()
