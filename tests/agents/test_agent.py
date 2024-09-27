import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from browsergym.experiments.loop import EnvArgs, ExpArgs

from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_3_5
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.analyze import inspect_results
from agentlab.experiments import launch_exp
from agentlab.llm.chat_api import BaseModelArgs, CheatMiniWoBLLMArgs


def test_generic_agent():
    exp_args = ExpArgs(
        agent_args=GenericAgentArgs(
            chat_model_args=CheatMiniWoBLLMArgs(),
            flags=FLAGS_GPT_3_5,
        ),
        env_args=EnvArgs(task_name="miniwob.click-test", task_seed=42),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:

        launch_exp.run_experiments(1, [exp_args], Path(tmp_dir) / "generic_agent_test")

        result_record = inspect_results.load_result_df(tmp_dir, progress_fn=None)

        target = {
            "n_steps": 1,
            "cum_reward": 1.0,
            "terminated": True,
            "truncated": False,
            "err_msg": None,
            "stack_trace": None,
            "agent_args.flags.obs.use_ax_tree": True,
        }

        for key, target_val in target.items():
            assert key in result_record
            assert result_record[key].iloc[0] == target_val


@dataclass
class CheatMiniWoBLLM_Retry:
    """For unit-testing purposes only. It only work with miniwob.click-test task."""

    n_retry: int
    retry_count: int = 0

    def invoke(self, messages) -> str:
        if self.retry_count < self.n_retry:
            self.retry_count += 1
            return dict(role="assistant", content="I'm retrying")

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

    def __call__(self, messages) -> str:
        return self.invoke(messages)


@dataclass
class CheatMiniWoBLLMArgs_Retry(BaseModelArgs):
    n_retry: int = 2
    model_name: str = "test/cheat_miniwob_click_test_retry"

    def make_model(self):
        return CheatMiniWoBLLM_Retry(n_retry=self.n_retry)


def test_generic_agent_retry():
    exp_args = ExpArgs(
        agent_args=GenericAgentArgs(
            chat_model_args=CheatMiniWoBLLMArgs_Retry(n_retry=2),
            flags=FLAGS_GPT_3_5,
        ),
        env_args=EnvArgs(task_name="miniwob.click-test", task_seed=42),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        launch_exp.run_experiments(1, [exp_args], Path(tmp_dir) / "generic_agent_test")
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


def test_bust_retry():
    exp_args = ExpArgs(
        agent_args=GenericAgentArgs(
            chat_model_args=CheatMiniWoBLLMArgs_Retry(n_retry=10),
            flags=FLAGS_GPT_3_5,
        ),
        env_args=EnvArgs(task_name="miniwob.click-test", task_seed=42),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        launch_exp.run_experiments(1, [exp_args], Path(tmp_dir) / "generic_agent_test")
        result_record = inspect_results.load_result_df(tmp_dir, progress_fn=None)

        target = {
            "stats.cum_n_retry": 5,
            "stats.cum_busted_retry": 1,
            "n_steps": 0,
            "cum_reward": 0,
            "err_msg": None,
        }

        for key, target_val in target.items():
            assert key in result_record
            assert result_record[key].iloc[0] == target_val


if __name__ == "__main__":
    # test_generic_agent()
    test_bust_retry()
