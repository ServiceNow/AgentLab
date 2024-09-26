from dataclasses import dataclass
import logging
from pathlib import Path
import time

from agentlab.agents.agent_args import AgentArgs
from .generic_agent import GenericAgentArgs, GenericAgent
from browsergym.experiments.loop import ExpResult, ExpArgs, yield_all_exp_results
from browsergym.experiments.agent import AgentInfo


class ReproChatModel:
    """A chat model that reproduces a conversation.

    Args:
        messages (list): A list of messages previously executed.
        delay (int): A delay to simulate the time it takes to generate a response.
    """

    def __init__(self, messages, delay=1) -> None:
        self.messages = messages
        self.delay = delay

    def invoke(self, messages):
        time.sleep(self.delay)
        # return the next message in the list
        return self.messages[len(messages)]


@dataclass
class ReproAgentArgs(GenericAgentArgs):

    # starting with "_" will prevent from being part of the index in the load_results function
    _repro_dir: str = None

    def make_agent(self):
        return ReproAgent(self.chat_model_args, self.flags, self.max_retry, self._repro_dir)


class ReproAgent(GenericAgent):

    def __init__(
        self,
        chat_model_args,
        flags,
        max_retry=4,
        repro_dir=None,
    ):
        self.exp_result = ExpResult(repro_dir)
        super().__init__(chat_model_args, flags, max_retry)

    def get_action(self, obs):

        # replace the chat model with a reproducible chat that will mimic the
        # same answers
        step = len(self.actions)
        step_info = self.exp_result.get_step_info(step)
        chat_messages = step_info.agent_info.get("chat_messages", None)
        if chat_messages is None:
            err_msg = self.exp_result.summary_info["err_msg"]

            agent_info = AgentInfo(
                markup_page=f"Agent had no chat messages. Perhaps there was an error. err_msg:\n{err_msg}",
            )
            return None, agent_info

        self.chat_llm = ReproChatModel(chat_messages)
        action, agent_info = super().get_action(obs)

        return _make_agent_stats(action, agent_info, step_info)


def _make_agent_stats(action, agent_info, step_info):
    # TODO
    return action, agent_info


def reproduce_study(original_study_dir: Path | str):
    """Reproduce a study by running the same experiments with the same agent."""

    original_study_dir = Path(original_study_dir)

    study_name = f"reproducibility_of_{original_study_dir.name}"

    exp_args_list = []
    for exp_result in yield_all_exp_results(original_study_dir, progress_fn=None):
        agent_args = make_repro_agent(exp_result.exp_args.agent_args, exp_dir=exp_result.exp_dir)
        exp_args_list.append(
            ExpArgs(
                agent_args=agent_args,
                env_args=exp_result.exp_args.env_args,
                logging_level=logging.DEBUG,
            )
        )
    return study_name, exp_args_list


def make_repro_agent(agent_args: AgentArgs, exp_dir: Path | str):
    """Create a reproducibility agent from an existing agent.

    Note, if a new flag was added, it was not saved in the original pickle. When
    loading the pickle it silently adds the missing flag and set it to its
    default value. The new repro agent_args will thus have the new flag set to
    its default value.

    Args:
        agent_args (AgentArgs): The original agent args.
        exp_dir (Path | str): The directory where the experiment was saved.

    """
    exp_dir = Path(exp_dir)
    assert isinstance(agent_args, GenericAgentArgs)
    assert exp_dir.exists()  # sanity check

    return ReproAgentArgs(
        agent_name=f"Repro_{agent_args.agent_name}",
        chat_model_args=agent_args.chat_model_args,
        flags=agent_args.flags,
        max_retry=agent_args.max_retry,
        _repro_dir=exp_dir,
    )
