# FocusAgent

<p align="center">
  <img src="assets/logo.png" alt="FocusAgent Logo" width="50" height="50">
</p>

A two-stage LLM-powered agent that uses intelligent line-by-line filtering to focus on relevant parts of the accessibility tree (AxTree). The agent employs a separate LLM as a "retriever" to identify and extract only the most relevant lines for task completion.

## Overview

The FocusAgent introduces a novel two-stage approach where a dedicated LLM analyzes the full accessibility tree and selectively extracts relevant lines before the main agent processes the filtered content. This approach provides more intelligent and context-aware filtering compared to traditional chunking methods, as the retriever LLM can understand the semantic relationships and task relevance of different UI elements.

## Key Features

- **Two-stage LLM architecture**: Separate retriever LLM for intelligent content filtering
- **Line-based precision**: Filters at line granularity for maximum control
- **Multiple retriever types**: Different filtering strategies (line, defender, restrictive, neutral)
- **Structure preservation**: Optional tree structure maintenance during filtering
- **Visual integration**: Can incorporate screenshots for enhanced understanding
- **Adaptive strategies**: Configurable filtering approaches based on task requirements
- **Attack resilience**: Special defender mode for security-sensitive scenarios

## Architecture

```text
Goal (+ History ) + AxTree (+ Screenshot) → Retriever LLM → Filtered AxTree → Main LLM → Action
                                                ↓
                                            Line Ranges
```

## Usage

### Basic Configuration

```python
from agentlab.agents.focus_agent import FocusAgent, FocusAgentArgs
from agentlab.agents.focus_agent.llm_retriever_prompt import LlmRetrieverPromptFlags

# Configure retriever prompt flags
retriever_flags = LlmRetrieverPromptFlags(
    use_abstract_example=False,     # Include abstract example in prompt
    use_concrete_example=False,     # Include concrete example in prompt
    use_screenshot=False,           # Include screenshot analysis
    use_history=False              # Include interaction history
)

# Create agent
agent_args = FocusAgentArgs(
    chat_model_args=your_main_model_args,
    retriever_chat_model_args=your_retriever_model_args,  # Can be different model
    flags=your_flags,               # Main agent (GenericAgent) flags
    retriever_prompt_flags=retriever_flags,
    keep_structure=False,           # Preserve tree structure
    retriever_type="line",          # Retriever strategy
    max_retry=4
)

agent = agent_args.make_agent()
```

### Pre-configured Agents

```python
from agentlab.agents.focus_agent.agent_configs import (
    FOCUS_AGENT_4_1_MINI,
    FOCUS_AGENT_CLAUDE_3_7_RETRIEVER_4_1_MINI,
    FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI
)

# GPT-4.1 Mini for both main and retriever
agent = FOCUS_AGENT_4_1_MINI.make_agent()

# Claude 3.7 Sonnet for main, GPT-4.1 Mini for retriever
agent = FOCUS_AGENT_CLAUDE_3_7_RETRIEVER_4_1_MINI.make_agent()

# GPT-4.1 for main, GPT-4.1 Mini for retriever  
agent = FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI.make_agent()
```

## Configuration Parameters

### FocusAgentArgs

- `chat_model_args` (ChatModelArgs): Configuration for the main action-generating LLM
- `retriever_chat_model_args` (ChatModelArgs): Configuration for the retriever LLM
- `flags` (GenericPromptFlags): Main agent prompt configuration
- `retriever_prompt_flags` (LlmRetrieverPromptFlags): Retriever prompt configuration
- `keep_structure` (bool, default=False): Preserve tree indentation and structure
- `strategy` (str): Structure preservation strategy - "bid", "role", or "bid+role"
- `retriever_type` (str, default="line"): Filtering strategy type
- `max_retry` (int, default=4): Maximum retry attempts
- `benchmark` (str): Target benchmark name for optimization

### LlmRetrieverPromptFlags

- `use_abstract_example` (bool, default=False): Include abstract formatting example
- `use_concrete_example` (bool, default=False): Include concrete example with explanations
- `use_screenshot` (bool, default=False): Provide screenshot to retriever LLM
- `use_history` (bool, default=False): Include interaction history in retrieval

## Retriever Types

### Line Retriever ("line")

**Purpose**: Standard intelligent line filtering
**Approach**: Analyzes content for task relevance and extracts necessary lines
**Use Case**: General web automation tasks

### Defender Retriever ("defender")

**Purpose**: Security-focused filtering with attack resistance
**Approach**: Specifically designed to filter out malicious content and attacks
**Use Case**: Security-sensitive scenarios, adversarial environments

### Restrictive Retriever ("restrictive")

**Purpose**: Aggressive pruning for minimal context
**Approach**: Only keeps absolutely essential lines, removes uncertain content
**Use Case**: Token-constrained scenarios, simple tasks

### Neutral Retriever ("neutral")

**Purpose**: Balanced filtering approach
**Approach**: Moderate filtering without aggressive pruning
**Use Case**: Complex tasks requiring more context## How It Works

1. **Line Numbering**: Accessibility tree gets line numbers added for precise referencing
2. **Retriever Analysis**: Dedicated LLM analyzes the numbered tree with goal and context
3. **Line Range Extraction**: Retriever returns specific line ranges to keep: `[(10,12), (45,67)]`
4. **Content Filtering**: Original tree is filtered to include only specified lines
5. **Structure Processing**: Optional structure preservation or simple line concatenation
6. **Action Generation**: Main LLM processes filtered content to generate actions

## Structure Preservation

When `keep_structure=True`, the agent can maintain tree hierarchy using different strategies:

- **"bid"**: Preserve elements with bid attributes 
- **"bid+role"**: Preserve elements with both bid and role attributes

## Use Cases

- **Complex web applications**: Multi-step workflows requiring intelligent content selection
- **Security-sensitive tasks**: Using defender mode to filter malicious content
- **Token optimization**: Aggressive filtering for cost reduction
- **Visual-dependent tasks**: Screenshot integration for better understanding
- **Large page navigation**: Intelligent filtering of extensive content

## Performance Considerations

- **Dual LLM calls**: Each action requires both retriever and main LLM calls
- **Latency**: Additional processing time for two-stage approach
- **Cost**: Retriever calls add to overall API usage (mitigated by using smaller models)
- **Accuracy**: Retriever quality directly impacts main agent performance

## Example Output

The agent provides comprehensive debugging information:

- `retriever_answer`: Raw response from retriever LLM with reasoning
- `retriever_prompt`: Full prompt sent to retriever LLM
- `pruned_tree`: Final filtered accessibility tree used for action generation

## Advanced Features

### Screenshot Integration

When `use_screenshot=True`, the retriever can analyze visual information alongside the accessibility tree for more accurate filtering.

### Attack Resistance

The defender retriever type specifically filters malicious content and prompt injection attempts, making it suitable for security-critical applications.

## Citation

If you use this agent in your work, please consider citing:

```bibtex

```