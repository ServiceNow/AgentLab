# EmbeddingAgent

A retrieval-augmented agent that uses semantic similarity through embeddings to filter and retrieve the most relevant parts of the accessibility tree (AXTree) based on the current goal and task history.

## Overview

``EmbeddingAgent`` extends ``GenericAgent`` with semantic retrieval capabilities powered by OpenAI's embedding models. Instead of processing the entire accessibility tree, it chunks the content and uses cosine similarity between embeddings to retrieve only the most semantically relevant sections, providing better context understanding than keyword-based approaches.

## Key Features

- **Semantic retrieval**: Uses OpenAI embedding models for semantic similarity matching
- **Multiple embedding providers**: Supports both OpenAI and Azure OpenAI APIs
- **Configurable similarity measures**: Cosine similarity and dot product options
- **Token-aware chunking**: Splits accessibility trees using tiktoken for optimal token usage
- **Embedding normalization**: Optional L2 normalization for improved similarity computation
- **History integration**: Combines goal and history for contextual retrieval queries

## Architecture

```text
Query (goal + history) → Embedding Model → Query Vector
                                              ↓
AXTree → Chunks → Embedding Model → Chunk Vectors → Cosine Similarity → Top-K Chunks → LLM → Action
```

## Usage

### Basic Configuration

```python
from agentlab.agents.embedding_agent import EmbeddingRetrieverAgent, EmbeddingRetrieverAgentArgs
from agentlab.agents.embedding_agent.embedding_retriever import OpenAIRetrieverArgs

# Configure embedding retriever
retriever_args = OpenAIRetrieverArgs(
    client="openai",                    # or "azure"
    model_name="text-embedding-3-small", # OpenAI embedding model
    top_k=10,                           # Number of chunks to retrieve
    chunk_size=200,                     # Tokens per chunk
    overlap=10,                         # Token overlap between chunks
    measure="cosine",                   # Similarity measure
    normalize_embeddings=True,          # L2 normalization
    use_recursive_text_splitter=False   # Use LangChain splitter
)

# Create agent
agent_args = EmbeddingRetrieverAgentArgs(
    chat_model_args=your_chat_model_args,
    flags=your_flags,
    retriever_args=retriever_args
)

agent = agent_args.make_agent()
```

### Pre-configured Agents

```python
from agentlab.agents.embedding_agent.agent_configs import (
    EMBEDDING_RETRIEVER_AGENT,      # Chunk size is 200 tokens
    EMBEDDING_RETRIEVER_AGENT_100   # Chunk size is 100 tokens
)

# Use default Azure OpenAI configuration
agent = EMBEDDING_RETRIEVER_AGENT.make_agent()

# Use configuration with smaller chunks
agent = EMBEDDING_RETRIEVER_AGENT_100.make_agent()
```

### Environment Setup

#### For OpenAI API

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

#### For Azure OpenAI API

```bash
export AZURE_OPENAI_API_KEY="your-azure-api-key"
export AZURE_OPENAI_API_BASE="your-azure-endpoint"
```

## Configuration Parameters

### OpenAIRetrieverArgs

- `client` (str, default="openai"): API provider - "openai" or "azure"
- `model_name` (str, default="text-embedding-small-3"): Embedding model name
- `top_k` (int, default=5): Number of most relevant chunks to retrieve
- `chunk_size` (int, default=100): Number of tokens per chunk
- `overlap` (int, default=10): Token overlap between consecutive chunks
- `measure` (str, default="cosine"): Similarity measure - "cosine" or "dot"
- `normalize_embeddings` (bool, default=True): Apply L2 normalization to embeddings
- `use_recursive_text_splitter` (bool, default=False): Use LangChain's recursive text splitter


## Citation

If you use this agent in your work, please consider citing:

```bibtex

```