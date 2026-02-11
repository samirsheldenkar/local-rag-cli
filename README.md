# Local RAG CLI

A CLI-based RAG (Retrieval Augmented Generation) system built on LlamaIndex, optimized for Apple Silicon (Mac Mini M4 Pro). Supports ingestion of PDF, Office documents, and images using multimodal embeddings. Uses ChromaDB as the default vector store with Qdrant as an alternative option.

## Features

- **Local LLM Support**: Works with LMStudio, llama.cpp, or any OpenAI-compatible API
- **Multimodal RAG**: Indexes both text (PDF, DOCX, XLSX) and images
- **Vector Database**: Uses ChromaDB by default; Qdrant available as an alternative
- **Apple Silicon Optimized**: Designed for Mac M4 Pro with local inference

## Installation

```bash
# Clone and install
git clone <repository>
cd local-rag-cli
uv pip install -e .
```

## Configuration

Create a `.env` file or set environment variables:

```env
# Vector Store Configuration (default: chromadb)
# Options: "chromadb", "qdrant"
VECTOR_STORE_TYPE=chromadb

# ChromaDB Configuration (persistent mode)
CHROMADB_PATH=./chromadb_data

# Qdrant Configuration (only needed if VECTOR_STORE_TYPE=qdrant)
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-api-key  # Optional

# LLM Configuration (OpenAI-compatible)
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=your-api-key  # Optional
LLM_MODEL=local-model

# Embedding Models
TEXT_EMBEDDING_MODEL=BAAI/bge-m3
IMAGE_EMBEDDING_MODEL=ViT-B/32

# Request Configuration
REQUEST_TIMEOUT=600.0
```

## Usage

### Check System Health

```bash
local-rag-cli health
```

### Ingest Documents

```bash
local-rag-cli ingest ./documents
```

### Query Documents

```bash
local-rag-cli query "What is the capital of France?"
```

### Interactive Chat

```bash
local-rag-cli chat
```

## Development

```bash
# Run tests
uv run pytest

# Run in development mode
uv run python -m local_rag_cli --help
```

## Requirements

- Python 3.12+
- For Qdrant: A running Qdrant instance (local or remote)
- For ChromaDB: No additional server required (uses persistent local storage)
- Local LLM server (LMStudio, llama.cpp, etc.)
