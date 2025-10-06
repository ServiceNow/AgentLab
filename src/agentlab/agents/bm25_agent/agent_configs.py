from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_4o
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

from .bm25_retriever import BM25RetrieverArgs
from .bm25_retriever_agent import BM25RetrieverAgentArgs, BM25RetrieverAgentFlags

FLAGS_GPT_4o = FLAGS_GPT_4o.copy()
FLAGS_GPT_4o.obs.use_think_history = True

BM25_RETRIEVER_AGENT = BM25RetrieverAgentArgs(
    agent_name="BM25RetrieverAgent-4.1",
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-2025-04-14"],
    flags=FLAGS_GPT_4o,
    retriever_args=BM25RetrieverArgs(
        top_k=10,
        chunk_size=200,
        overlap=10,
        use_recursive_text_splitter=False,
    ),
    retriever_flags=BM25RetrieverAgentFlags(
        use_history=True,
    ),
)

BM25_RETRIEVER_AGENT_100 = BM25RetrieverAgentArgs(
    agent_name="BM25RetrieverAgent-4.1-100",
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-2025-04-14"],
    flags=FLAGS_GPT_4o,
    retriever_args=BM25RetrieverArgs(
        top_k=10,
        chunk_size=100,
        overlap=10,
        use_recursive_text_splitter=False,
    ),
    retriever_flags=BM25RetrieverAgentFlags(
        use_history=True,
    ),
)

BM25_RETRIEVER_AGENT_50 = BM25RetrieverAgentArgs(
    agent_name="BM25RetrieverAgent-4.1-50",
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-2025-04-14"],
    flags=FLAGS_GPT_4o,
    retriever_args=BM25RetrieverArgs(
        top_k=10,
        chunk_size=50,
        overlap=5,
        use_recursive_text_splitter=False,
    ),
    retriever_flags=BM25RetrieverAgentFlags(
        use_history=True,
    ),
)

BM25_RETRIEVER_AGENT_500 = BM25RetrieverAgentArgs(
    agent_name="BM25RetrieverAgent-4.1-500",
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-2025-04-14"],
    flags=FLAGS_GPT_4o,
    retriever_args=BM25RetrieverArgs(
        top_k=10,
        chunk_size=500,
        overlap=10,
        use_recursive_text_splitter=False,
    ),
    retriever_flags=BM25RetrieverAgentFlags(
        use_history=True,
    ),
)
