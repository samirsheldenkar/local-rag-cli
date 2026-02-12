"""Storage module for vector store management."""

from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

import chromadb

from local_rag_cli.config import settings
from local_rag_cli.embeddings import OpenCLIPEmbedding


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client instance."""
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )


def get_chroma_client() -> chromadb.PersistentClient:
    """Get ChromaDB persistent client instance."""
    return chromadb.PersistentClient(path=settings.CHROMADB_PATH)


def get_text_vector_store():
    """Get vector store for text documents."""
    if settings.VECTOR_STORE_TYPE == "chromadb":
        client = get_chroma_client()
        collection = client.get_or_create_collection("rag_text")
        return ChromaVectorStore(
            chroma_collection=collection,
        )
    elif settings.VECTOR_STORE_TYPE == "qdrant":
        client = get_qdrant_client()
        return QdrantVectorStore(
            client=client,
            collection_name="rag_text",
        )
    else:
        raise ValueError(f"Unknown vector store type: {settings.VECTOR_STORE_TYPE}")


def get_image_vector_store():
    """Get vector store for images."""
    if settings.VECTOR_STORE_TYPE == "chromadb":
        client = get_chroma_client()
        collection = client.get_or_create_collection("rag_images")
        return ChromaVectorStore(
            chroma_collection=collection,
        )
    elif settings.VECTOR_STORE_TYPE == "qdrant":
        client = get_qdrant_client()
        return QdrantVectorStore(
            client=client,
            collection_name="rag_images",
        )
    else:
        raise ValueError(f"Unknown vector store type: {settings.VECTOR_STORE_TYPE}")


def get_multimodal_index() -> MultiModalVectorStoreIndex:
    """Get multimodal index combining text and image stores."""
    text_store = get_text_vector_store()
    image_store = get_image_vector_store()

    image_embed = OpenCLIPEmbedding(
        model_name=settings.IMAGE_EMBEDDING_MODEL,
        pretrained=settings.IMAGE_EMBEDDING_PRETRAINED,
    )

    return MultiModalVectorStoreIndex(
        vector_store=text_store,
        image_vector_store=image_store,
        image_embed_model=image_embed,
    )


def ensure_collections_exist() -> None:
    """Ensure vector store collections exist."""
    if settings.VECTOR_STORE_TYPE == "chromadb":
        client = get_chroma_client()
        # ChromaDB creates collections automatically with get_or_create_collection
        # but we call it here to ensure they exist at startup
        client.get_or_create_collection("rag_text")
        client.get_or_create_collection("rag_images")
    elif settings.VECTOR_STORE_TYPE == "qdrant":
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
