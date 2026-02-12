# Local RAG CLI

A CLI-based RAG (Retrieval Augmented Generation) system built on LlamaIndex, optimized for Apple Silicon (Mac Mini M4 Pro). Supports ingestion of PDF, Office documents, and images using multimodal embeddings. Uses ChromaDB as the default vector store with Qdrant as an alternative option.

## Features

- **Local LLM Support**: Works with Ollama, llama.cpp, or any OpenAI-compatible API
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

# LLM Configuration (Ollama)
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.1:8b

# Embedding Models
TEXT_EMBEDDING_MODEL=BAAI/bge-m3
IMAGE_EMBEDDING_MODEL=ViT-B-32
IMAGE_EMBEDDING_PRETRAINED=laion2b_s34b_b79k

# Request Configuration
REQUEST_TIMEOUT=600.0
```

### Recommended Models

| Setting | Default | Alternative | Notes |
|---------|---------|-------------|-------|
| `TEXT_EMBEDDING_MODEL` | `BAAI/bge-m3` | `BAAI/bge-large-en-v1.5` | bge-m3 is multilingual; bge-large is best for English-only |
| `IMAGE_EMBEDDING_MODEL` | `ViT-B-32` | `ViT-L-14` | Larger model = better image understanding |
| `IMAGE_EMBEDDING_PRETRAINED` | `laion2b_s34b_b79k` | `datacomp_xl_s13b_b90k` | LAION-2B is the standard; DataComp is newer |
| `LLM_MODEL` | `llama3.1:8b` | `qwen2.5:14b-q4_K_M` | 14B fits in 24GB with room to spare |

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
- [Ollama](https://ollama.com) for local LLM inference (or any OpenAI-compatible API)
- For Qdrant: A running Qdrant instance (local or remote)
- For ChromaDB: No additional server required (uses persistent local storage)
