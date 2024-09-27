import copy
from dataclasses import dataclass
import logging
from pathlib import Path
import time

from agentlab.agents.agent_args import AgentArgs
from .generic_agent import GenericAgentArgs, GenericAgent
from browsergym.experiments.loop import ExpResult, ExpArgs, yield_all_exp_results
from browsergym.experiments.agent import AgentInfo
import difflib


class ReproChatModel:
    """A chat model that reproduces a conversation.

    Args:
        messages (list): A list of messages previously executed.
        delay (int): A delay to simulate the time it takes to generate a response.
    """

    def __init__(self, old_messages, delay=1) -> None:
        self.old_messages = old_messages
        self.delay = delay

    def invoke(self, messages: list):
        self.new_messages = copy(messages)
        old_response = self.old_messages[len(messages)]
        self.new_messages.append(old_response)
        time.sleep(self.delay)
        # return the next message in the list
        return old_response


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
        old_chat_messages = step_info.agent_info.get("chat_messages", None)
        if old_chat_messages is None:
            err_msg = self.exp_result.summary_info["err_msg"]

            agent_info = AgentInfo(
                markup_page=f"Agent had no chat messages. Perhaps there was an error. err_msg:\n{err_msg}",
            )
            return None, agent_info

        self.chat_llm = ReproChatModel(old_chat_messages)
        action, agent_info = super().get_action(obs)

        return _make_agent_stats(
            action, agent_info, step_info, old_chat_messages, self.chat_llm.new_messages
        )


def _make_agent_stats(action, agent_info, step_info, old_chat_messages, new_chat_messages):

    # format all messages into a string
    old_msg_str = _format_messages(old_chat_messages)
    new_msg_str = _format_messages(new_chat_messages)
    html_diff = _make_diff(old_str=old_msg_str, new_str=new_msg_str)

    if isinstance(agent_info, dict):
        agent_info = AgentInfo(**agent_info)

    agent_info.html_page = html_diff
    agent_info.stats = _diff_stats(old_msg_str, new_msg_str)

    return action, agent_info


def _format_messages(messages: list[dict]):
    return "\n".join(f"{m['role']} message:\n{m['content']}\n" for m in messages)


def _make_diff(old_str, new_str):
    diff = difflib.HtmlDiff().make_file(
        old_str.splitlines(), new_str.splitlines(), fromdesc="Old Version", todesc="New Version"
    )
    return diff


def _diff_stats(str1: str, str2: str):
    lines1 = str1.splitlines()
    lines2 = str2.splitlines()

    diff = list(difflib.Differ().compare(lines1, lines2))

    # Count added and removed lines
    added = sum(1 for line in diff if line.startswith("+ "))
    removed = sum(1 for line in diff if line.startswith("- "))

    # Calculate difference ratio
    difference_ratio = (added + removed) / (2 * max(len(lines1), len(lines2)))

    return dict(lines_added=added, lines_removed=removed, difference_ratio=difference_ratio)


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
