"""Storage module for vector store management."""

from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from local_rag_cli.config import settings


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client instance."""
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )


def get_text_vector_store() -> QdrantVectorStore:
    """Get vector store for text documents."""
    client = get_qdrant_client()
    return QdrantVectorStore(
        client=client,
        collection_name="rag_text",
    )


def get_image_vector_store() -> QdrantVectorStore:
    """Get vector store for images."""
    client = get_qdrant_client()
    return QdrantVectorStore(
        client=client,
        collection_name="rag_images",
    )


def get_multimodal_index() -> MultiModalVectorStoreIndex:
    """Get multimodal index combining text and image stores."""
    text_store = get_text_vector_store()
    image_store = get_image_vector_store()

    return MultiModalVectorStoreIndex(
        vector_store=text_store,
        image_vector_store=image_store,
    )


def ensure_collections_exist() -> None:
    """Ensure Qdrant collections exist."""
    client = get_qdrant_client()

    # Create text collection if it doesn't exist
    try:
        client.get_collection("rag_text")
    except Exception:
        client.create_collection(
            collection_name="rag_text",
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

    # Create image collection if it doesn't exist
    try:
        client.get_collection("rag_images")
    except Exception:
        client.create_collection(
            collection_name="rag_images",
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
