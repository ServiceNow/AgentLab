# BM25Agent

A retrieval-augmented agent that uses BM25 (Best Matching 25) algorithm to filter and retrieve the most relevant parts of the accessibility tree (AXTree) based on the current goal and task history.

## Overview

``BM25Agent`` extends ``GenericAgent`` with intelligent content retrieval capabilities. Instead of processing the entire accessibility tree, it chunks the content and uses BM25 ranking to retrieve only the most relevant sections, reducing token usage and improving focus on task-relevant elements.

## Key Features

- **BM25-based retrieval**: Uses the BM25 algorithm to rank and retrieve relevant content chunks
- **Token-aware chunking**: Splits accessibility trees using tiktoken for optimal token usage
- **Configurable parameters**: Adjustable chunk size, overlap, and top-k retrieval
- **History integration**: Can optionally include task history in retrieval queries
- **Memory efficient**: Reduces context size by filtering irrelevant content

## Architecture

```text
Query (goal + history) → BM25 Retriever → Top-K Chunks → LLM → Action
                            ↑
                          AXTree
```

## Usage

### Basic Configuration

```python
from agentlab.agents.bm25_agent import BM25RetrieverAgent, BM25RetrieverAgentArgs
from agentlab.agents.bm25_agent.bm25_retriever import BM25RetrieverArgs
from agentlab.agents.bm25_agent.bm25_retriever_agent import BM25RetrieverAgentFlags

# Configure retriever parameters
retriever_args = BM25RetrieverArgs(
    chunk_size=200,                     # Tokens per chunk
    overlap=10,                         # Token overlap between chunks
    top_k=10,                           # Number of chunks to retrieve
    use_recursive_text_splitter=False   # Use Langchain text splitter
)

# Configure agent flags
retriever_flags = BM25RetrieverAgentFlags(
    use_history=True       # Include task history in queries
)

# Create agent
agent_args = BM25RetrieverAgentArgs(
    chat_model_args=your_chat_model_args,
    flags=your_flags,
    retriever_args=retriever_args,
    retriever_flags=retriever_flags
)

agent = agent_args.make_agent()
```

### Pre-configured Agents

```python
from agentlab.agents.bm25_agent.agent_configs import (
    BM25_RETRIEVER_AGENT,       # Chunk size is 200 tokens
    BM25_RETRIEVER_AGENT_100    # Chunk size is 100 tokens
)

# Use default configuration
agent = BM25_RETRIEVER_AGENT.make_agent()
```

## Configuration Parameters

### BM25RetrieverArgs

- `chunk_size` (int, default=100): Number of tokens per chunk
- `overlap` (int, default=10): Token overlap between consecutive chunks  
- `top_k` (int, default=5): Number of most relevant chunks to retrieve
- `use_recursive_text_splitter` (bool, default=False): Use LangChain's recursive text splitter. Using this text splitter will override the ``chunk_size`` an ``overlap`` parameters.

### BM25RetrieverAgentFlags

- `use_history` (bool, default=False): Include interaction history in retrieval queries

## Citation

If you use this agent in your work, please consider citing:

```bibtex

```