import pytest
import re
import tempfile
import gradio as gr
import dataclasses

from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.experiments.agent import Agent
from browsergym.experiments.loop import AbstractAgentArgs, EnvArgs, ExpArgs, get_exp_result


from agentlab.analyze import agent_xray

SAVEDIR_BASE = "/mnt/home/projects/ui-copilot/results/tmp"


def test_update_gradio_df_from_savedir_base():
    """
    Tests whether the initial loading of results from a savedir base works.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        run_and_save_experiment(tmp_dir, task_seed=42)
        run_and_save_experiment(tmp_dir, task_seed=43)
        result_df, _ = agent_xray.update_gradio_df_from_savedir_base(tmp_dir)

        # exp_result = get_exp_result(result_df["exp_dir"][0])
        # exp_record = exp_result.get_exp_record()

        # assert that the two are dataframes of size 2
        assert len(result_df) == 2
        # assert len(gradio_result_df) == 2


# def test_on_change_savedir_base():
#     """
#     TODO: Tests whether changing the save directory works.
#     """
#     out_list = agent_xray.on_change_savedir_base(SAVEDIR_BASE)
#     result_df, gradio_result_df = agent_xray.update_gradio_df_from_savedir_base(SAVEDIR_BASE)

#     # out_list should have 3 elements
#     assert len(out_list) == 3

#     # out_list[0] should be same as gradio_result_df
#     assert out_list[0]["value"].equals(gradio_result_df)


# def test_on_select_df():
#     """
#     TODO: Tests whether changing the save directory works.
#     """
#     result_df, gradio_result_df = agent_xray.update_gradio_df_from_savedir_base(SAVEDIR_BASE)

#     row_id = 0
#     episode_id = 0
#     step_id = 0

#     row, episode_series, step_obj = agent_xray.get_row_info(row_id, episode_id, step_id)
#     out_list = []
#     exp_dict_gr.select(
#         fn=on_select_df,
#         inputs=[exp_dict_gr],
#         outputs=out_list,
#     )
#     print()


class MiniwobTestAgent(Agent):

    def __init__(self):
        self.action_space = HighLevelActionSet(subsets="bid")

    def get_action(self, obs: dict) -> tuple[str, dict]:
        match = re.search(r"^\s*\[(\d+)\].*button", obs["axtree_txt"], re.MULTILINE | re.IGNORECASE)

        if match:
            bid = match.group(1)
            action = f'click("{bid}")'
        else:
            raise Exception("Can't find the button's bid")

        return action, dict(think="I'm clicking the button as requested.")


@dataclasses.dataclass
class MiniwobTestAgentArgs(AbstractAgentArgs):
    def make_agent(self):
        return MiniwobTestAgent()

    @property
    def agent_name(self) -> str:
        return MiniwobTestAgent.__name__


def run_and_save_experiment(savedir, task_seed=42):
    exp_args = ExpArgs(
        agent_args=MiniwobTestAgentArgs(),
        env_args=EnvArgs(task_name="miniwob.click-test", task_seed=42),
    )

    exp_args.prepare(savedir)
    exp_args.run()


if __name__ == "__main__":
    # run_and_save_experiment("/mnt/home/projects/ui-copilot/results/sample")
    # test_on_select_df()
    # test_on_change_savedir_base()
    test_update_gradio_df_from_savedir_base()
