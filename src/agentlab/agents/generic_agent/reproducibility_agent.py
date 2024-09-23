from dataclasses import dataclass
import time
from .generic_agent import GenericAgentArgs, GenericAgent
from browsergym.experiments.loop import ExpResult
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

    repro_dir: str = None

    def make_agent(self):
        return ReproAgent(self.chat_model_args, self.flags, self.max_retry, self.repro_dir)


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
        chat_messages = step_info["agent_info"]["chat_messages"]
        self.chat_llm = ReproChatModel(chat_messages)

        action, agent_info = super().get_action(obs)

        return _make_agent_stats(action, agent_info, step_info)


def _make_agent_stats(action, agent_info, step_info):
    # TODO
    return action, agent_info
