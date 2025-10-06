import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch

from agentlab.agents.embedding_agent.agent_configs import FLAGS_GPT_4o
from agentlab.agents.embedding_agent.embedding_retriever import OpenAIRetrieverArgs
from agentlab.agents.embedding_agent.embedding_retriever_agent import (
    EmbeddingRetrieverAgentArgs,
)
from agentlab.analyze import inspect_results
from agentlab.experiments import launch_exp
from agentlab.experiments.loop import EnvArgs, ExpArgs
from agentlab.llm.chat_api import CheatMiniWoBLLMArgs


@dataclass
class MockEmbeddingRetriever:
    """Mock embedding retriever for testing purposes."""

    def __init__(self, args):
        self.args = args

    def encode(self, text: Union[str, list[str]]):
        # Return a simple mock embedding vector
        return [0.1] * 10

    def retrieve(self, query: str, chunks: Union[str, list[str]]):
        # Mock retrieval that returns reasonable scores and indices
        num_chunks = len(chunks)
        top_k = min(self.args.top_k, num_chunks)

        # Create mock similarity scores (higher for earlier chunks)
        scores = torch.tensor([0.9 - (i * 0.1) for i in range(top_k)])
        indices = torch.tensor(list(range(top_k)))

        return scores, indices


@dataclass
class MockOpenAIRetrieverArgs(OpenAIRetrieverArgs):
    """Mock version of OpenAI retriever args that doesn't require API keys."""

    def make_model(self):
        return MockEmbeddingRetriever(self)


def test_embedding_retriever_agent():
    """Test basic embedding retriever agent functionality with miniwob.click-test task."""
    exp_args = ExpArgs(
        agent_args=EmbeddingRetrieverAgentArgs(
            chat_model_args=CheatMiniWoBLLMArgs(),
            flags=FLAGS_GPT_4o,
            retriever_args=MockOpenAIRetrieverArgs(
                client="openai",
                model_name="text-embedding-3-small",
                top_k=5,
                chunk_size=100,
                overlap=10,
                measure="cosine",
                normalize_embeddings=True,
                use_recursive_text_splitter=False,
            ),
            max_retry=4,
        ),
        env_args=EnvArgs(task_name="miniwob.click-test", task_seed=42),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        launch_exp.run_experiments(
            1,
            [exp_args],
            Path(tmp_dir) / "embedding_retriever_agent_test",
            parallel_backend="joblib",
        )

        result_record = inspect_results.load_result_df(tmp_dir, progress_fn=None)

        target = {
            "n_steps": 1,
            "cum_reward": 1.0,
            "terminated": True,
            "truncated": False,
            "err_msg": None,
            "stack_trace": None,
            "agent.flags.obs.use_ax_tree": True,
            "agent.flags.obs.use_think_history": True,
        }

        for key, target_val in target.items():
            assert key in result_record
            assert result_record[key].iloc[0] == target_val


def test_embedding_retriever_agent_with_recursive_splitter():
    """Test embedding retriever agent with recursive text splitter enabled."""
    exp_args = ExpArgs(
        agent_args=EmbeddingRetrieverAgentArgs(
            chat_model_args=CheatMiniWoBLLMArgs(),
            flags=FLAGS_GPT_4o,
            retriever_args=MockOpenAIRetrieverArgs(
                client="openai",
                model_name="text-embedding-3-small",
                top_k=5,
                chunk_size=100,
                overlap=10,
                measure="cosine",
                normalize_embeddings=True,
                use_recursive_text_splitter=True,  # Enable recursive splitter
            ),
            max_retry=4,
        ),
        env_args=EnvArgs(task_name="miniwob.click-test", task_seed=42),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        launch_exp.run_experiments(
            1,
            [exp_args],
            Path(tmp_dir) / "embedding_agent_test_recursive",
            parallel_backend="joblib",
        )

        result_record = inspect_results.load_result_df(tmp_dir, progress_fn=None)

        target = {
            "n_steps": 1,
            "cum_reward": 1.0,
            "terminated": True,
            "truncated": False,
            "err_msg": None,
        }

        for key, target_val in target.items():
            assert key in result_record
            assert result_record[key].iloc[0] == target_val


if __name__ == "__main__":
    test_embedding_retriever_agent()
    test_embedding_retriever_agent_with_recursive_splitter()
