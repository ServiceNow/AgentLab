import tempfile
from pathlib import Path

from agentlab.agents.bm25_agent.agent_configs import FLAGS_GPT_4o
from agentlab.agents.bm25_agent.bm25_retriever import BM25RetrieverArgs
from agentlab.agents.bm25_agent.bm25_retriever_agent import (
    BM25RetrieverAgentArgs,
    BM25RetrieverAgentFlags,
)
from agentlab.analyze import inspect_results
from agentlab.experiments import launch_exp
from agentlab.experiments.loop import EnvArgs, ExpArgs
from agentlab.llm.chat_api import CheatMiniWoBLLMArgs


def test_bm25_retriever_agent():
    """Test basic BM25 retriever agent functionality with miniwob.click-test task."""
    exp_args = ExpArgs(
        agent_args=BM25RetrieverAgentArgs(
            chat_model_args=CheatMiniWoBLLMArgs(),
            flags=FLAGS_GPT_4o,
            retriever_args=BM25RetrieverArgs(
                chunk_size=100,
                overlap=10,
                top_k=5,
                use_recursive_text_splitter=False,
            ),
            retriever_flags=BM25RetrieverAgentFlags(
                use_history=True,
            ),
            max_retry=4,
        ),
        env_args=EnvArgs(task_name="miniwob.click-test", task_seed=42),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        launch_exp.run_experiments(
            1, [exp_args], Path(tmp_dir) / "bm25_retriever_agent_test", parallel_backend="joblib"
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


def test_bm25_retriever_agent_with_recursive_splitter():
    """Test BM25 retriever agent with recursive text splitter enabled."""
    exp_args = ExpArgs(
        agent_args=BM25RetrieverAgentArgs(
            chat_model_args=CheatMiniWoBLLMArgs(),
            flags=FLAGS_GPT_4o,
            retriever_args=BM25RetrieverArgs(
                chunk_size=100,
                overlap=10,
                top_k=5,
                use_recursive_text_splitter=True,  # Enable recursive splitter
            ),
            retriever_flags=BM25RetrieverAgentFlags(
                use_history=True,
            ),
            max_retry=4,
        ),
        env_args=EnvArgs(task_name="miniwob.click-test", task_seed=42),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        launch_exp.run_experiments(
            1, [exp_args], Path(tmp_dir) / "bm25_agent_test_recursive", parallel_backend="joblib"
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
    test_bm25_retriever_agent()
    test_bm25_retriever_agent_with_recursive_splitter()
