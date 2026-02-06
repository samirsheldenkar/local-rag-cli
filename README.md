# Local RAG CLI

A CLI-based RAG (Retrieval Augmented Generation) system built on LlamaIndex and Qdrant, optimized for Apple Silicon (Mac Mini M4 Pro). Supports ingestion of PDF, Office documents, and images using multimodal embeddings.

## Features

- **Local LLM Support**: Works with LMStudio, llama.cpp, or any OpenAI-compatible API
- **Multimodal RAG**: Indexes both text (PDF, DOCX, XLSX) and images
- **Vector Database**: Uses Qdrant for efficient vector storage and retrieval
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
# Qdrant Configuration
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
- Qdrant instance (local or remote)
- Local LLM server (LMStudio, llama.cpp, etc.)
