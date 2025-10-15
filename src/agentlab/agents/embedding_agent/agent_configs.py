from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_4o
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

from .embedding_retriever import OpenAIRetrieverArgs
from .embedding_retriever_agent import EmbeddingRetrieverAgentArgs

FLAGS_GPT_4o = FLAGS_GPT_4o.copy()
FLAGS_GPT_4o.obs.use_think_history = True


EMBEDDING_RETRIEVER_AGENT = EmbeddingRetrieverAgentArgs(
    agent_name="EmbeddingRetrieverAgent-4.1",
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-2025-04-14"],
    flags=FLAGS_GPT_4o,
    retriever_args=OpenAIRetrieverArgs(
        client="azure",
        model_name="text-embedding-3-small",
        top_k=10,
        chunk_size=200,
        overlap=10,
        measure="cosine",
        normalize_embeddings=True,
        use_recursive_text_splitter=False,
    ),
)

EMBEDDING_RETRIEVER_AGENT_100 = EmbeddingRetrieverAgentArgs(
    agent_name="EmbeddingRetrieverAgent-4.1-100",
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-2025-04-14"],
    flags=FLAGS_GPT_4o,
    retriever_args=OpenAIRetrieverArgs(
        client="azure",
        model_name="text-embedding-3-small",
        top_k=10,
        chunk_size=100,
        overlap=10,
        measure="cosine",
        normalize_embeddings=True,
        use_recursive_text_splitter=False,
    ),
)

EMBEDDING_RETRIEVER_AGENT_50 = EmbeddingRetrieverAgentArgs(
    agent_name="EmbeddingRetrieverAgent-4.1-50",
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-2025-04-14"],
    flags=FLAGS_GPT_4o,
    retriever_args=OpenAIRetrieverArgs(
        client="azure",
        model_name="text-embedding-3-small",
        top_k=10,
        chunk_size=50,
        overlap=5,
        measure="cosine",
        normalize_embeddings=True,
        use_recursive_text_splitter=False,
    ),
)

EMBEDDING_RETRIEVER_AGENT_500 = EmbeddingRetrieverAgentArgs(
    agent_name="EmbeddingRetrieverAgent-4.1-500",
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-2025-04-14"],
    flags=FLAGS_GPT_4o,
    retriever_args=OpenAIRetrieverArgs(
        client="azure",
        model_name="text-embedding-3-small",
        top_k=10,
        chunk_size=500,
        overlap=10,
        measure="cosine",
        normalize_embeddings=True,
        use_recursive_text_splitter=False,
    ),
)
