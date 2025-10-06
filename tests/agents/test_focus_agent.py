import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agentlab.agents.focus_agent.agent_configs import FLAGS_GPT_4o
from agentlab.agents.focus_agent.focus_agent import FocusAgentArgs
from agentlab.agents.focus_agent.llm_retriever_prompt import LlmRetrieverPromptFlags
from agentlab.analyze import inspect_results
from agentlab.experiments import launch_exp
from agentlab.experiments.loop import EnvArgs, ExpArgs
from agentlab.llm.chat_api import BaseModelArgs, CheatMiniWoBLLMArgs
from agentlab.llm.llm_utils import Discussion


@dataclass
class MockRetrieverLLM:
    """Mock retriever LLM for testing purposes."""

    def __call__(self, messages) -> dict:
        # Return a simple line range that should work for miniwob.click-test
        # This mimics what the retriever should return - line ranges to focus on
        answer = """<think>
I need to identify the relevant lines for the button click task.
</think>
<answer>
[(2, 2)]
</answer>"""
        return dict(role="assistant", content=answer)

    def get_stats(self):
        return {}


@dataclass
class MockRetrieverLLMArgs(BaseModelArgs):
    model_name: str = "test/mock_retriever"

    def make_model(self):
        return MockRetrieverLLM()


def test_focus_agent():
    """Test basic focus agent functionality with miniwob.click-test task."""
    exp_args = ExpArgs(
        agent_args=FocusAgentArgs(
            chat_model_args=CheatMiniWoBLLMArgs(),
            retriever_chat_model_args=MockRetrieverLLMArgs(),
            flags=FLAGS_GPT_4o,
            retriever_prompt_flags=LlmRetrieverPromptFlags(
                use_abstract_example=False,
                use_concrete_example=False,
                use_screenshot=False,
                use_history=False,
            ),
            max_retry=4,
            strategy="bid",
            keep_structure=False,
            retriever_type="line",
        ),
        env_args=EnvArgs(task_name="miniwob.click-test", task_seed=42),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        launch_exp.run_experiments(
            1, [exp_args], Path(tmp_dir) / "focus_agent_test", parallel_backend="joblib"
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


@dataclass
class CheatMiniWoBLLM_ParseRetryFocus:
    """For unit-testing purposes only. It only work with miniwob.click-test task."""

    n_retry: int
    retry_count: int = 0

    def __call__(self, messages) -> str:
        if self.retry_count < self.n_retry:
            self.retry_count += 1
            return dict(role="assistant", content="I'm retrying")

        if isinstance(messages, Discussion):
            prompt = messages.to_string()
        else:
            prompt = messages[1].get("content", "")
        match = re.search(r"^\s*\[(\d+)\].*button", prompt, re.MULTILINE | re.IGNORECASE)

        if match:
            bid = match.group(1)
            action = f'click("{bid}")'
        else:
            raise Exception("Can't find the button's bid")

        answer = f"""I'm clicking the button as requested.
<action>
{action}
</action>
"""
        return dict(role="assistant", content=answer)

    def get_stats(self):
        return {}


@dataclass
class CheatMiniWoBLLMArgs_ParseRetryFocus(BaseModelArgs):
    n_retry: int = 2
    model_name: str = "test/cheat_miniwob_click_test_parse_retry_focus"

    def make_model(self):
        return CheatMiniWoBLLM_ParseRetryFocus(n_retry=self.n_retry)


def test_focus_agent_parse_retry():
    """Test focus agent with parsing retry functionality."""
    exp_args = ExpArgs(
        agent_args=FocusAgentArgs(
            chat_model_args=CheatMiniWoBLLMArgs_ParseRetryFocus(n_retry=2),
            retriever_chat_model_args=MockRetrieverLLMArgs(),
            flags=FLAGS_GPT_4o,
            retriever_prompt_flags=LlmRetrieverPromptFlags(
                use_abstract_example=False,
                use_concrete_example=False,
                use_screenshot=False,
                use_history=False,
            ),
            max_retry=4,
            strategy="bid",
            keep_structure=False,
            retriever_type="line",
        ),
        env_args=EnvArgs(task_name="miniwob.click-test", task_seed=42),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        launch_exp.run_experiments(
            1, [exp_args], Path(tmp_dir) / "focus_agent_test", parallel_backend="joblib"
        )
        result_record = inspect_results.load_result_df(tmp_dir, progress_fn=None)

        target = {
            "stats.cum_n_retry": 2,
            "stats.cum_busted_retry": 0,
            "n_steps": 1,
            "cum_reward": 1.0,
        }

        for key, target_val in target.items():
            assert key in result_record
            assert result_record[key].iloc[0] == target_val


if __name__ == "__main__":
    test_focus_agent()
    test_focus_agent_parse_retry()
