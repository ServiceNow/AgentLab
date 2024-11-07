from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Any

import bgym

try:
    from tapeagents.llms import LiteLLM
except ImportError as e:
    print("Please run install_tapeagents.sh to install tapeagents first.")
    raise e

import sys
sys.path.append(str(Path(__file__).parent.resolve() / "TapeAgents"))

from examples.workarena.agent import WorkArenaAgent
from examples.workarena.steps import (
    PageObservation,
    WorkArenaTape,
    WorkArenaTask,
    Action,
    GotoPageAction,
    ClickAction,
    SelectOptionAction,
    HoverAction,
    InputTextAction,
    PressAction,
    GoBackAction,
    GoForwardAction,
    ReflectionThought,
)
from .utils import flatten_axtree

from agentlab.llm.tracking import cost_tracker_decorator
from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.chat_api import BaseModelArgs

logger = logging.getLogger(__name__)


@dataclass
class TapeAgentArgs(AgentArgs):
    agent_name: str = "GuidedTapeAgent"
    chat_model_args: BaseModelArgs = None

    def make_agent(self) -> bgym.Agent:
        llm = LiteLLM(
            model_name=self.chat_model_args.model_name,
            use_cache=False,
            context_size=self.chat_model_args.max_total_tokens,
            parameters={"temperature": self.chat_model_args.temperature},
        )
        return GuidedTapeAgent(llm)

    def set_reproducibility_mode(self):
        self.chat_model_args.temperature = 0

    def prepare(self):
        return self.chat_model_args.prepare_server()

    def close(self):
        return self.chat_model_args.close_server()

class GuidedTapeAgent(bgym.Agent):
    steps: WorkArenaTape

    def __init__(self, llm: LiteLLM):
        self.tapeagent = WorkArenaAgent.create(llm)
        self.steps = WorkArenaTape(steps=[])

    @cost_tracker_decorator
    def get_action(self, obs: Any) -> tuple[str, dict]:
        # update tape with new observation
        text = flatten_axtree(
            obs["axtree_object"], filter_visible_only=True, ignore_navigation=False
        )
        obs_step = PageObservation(text=text, current_page=1, total_pages=1)
        self.tape = self.tape.append(obs_step)
        if not len(self.steps):  # first observation
            self.tape = self.tape.append(WorkArenaTask(task=obs["goal"]))

        # run agent and collect thoughts and last action
        thoughts = []
        tape_segment = []
        action = None
        for event in self.tapeagent.run(self.tape):
            if not event.step:
                continue
            step = event.step
            tape_segment.append(step.llm_dict())
            self.tape = self.tape.append(step)
            if isinstance(step, Action):
                action = step
                break
            elif isinstance(step, ReflectionThought):
                thoughts.append(step.llm_dict())

        # convert action step to an action string with function call
        assert action
        action_str = ""
        if isinstance(action, GotoPageAction):
            action_str = f"goto('{action.url}')"
        elif isinstance(action, ClickAction):
            action_str = (
                f"click('{action.bid}', button='{action.button}', modifiers={action.modifiers})"
            )
        elif isinstance(action, SelectOptionAction):
            action_str = f"select_option('{action.bid}', '{action.option}')"
        elif isinstance(action, HoverAction):
            action_str = f"hover('{action.bid}')"
        elif isinstance(action, InputTextAction):
            text = action.text.replace("'", "\\'")
            action_str = f"fill('{action.bid}', '{text}')"
        elif isinstance(action, PressAction):
            f"press('{action.bid}', '{action.key_comb}')"
        elif isinstance(action, GoBackAction):
            action_str = "go_back()"
        elif isinstance(action, GoForwardAction):
            action_str = "go_forward()"
        else:
            raise ValueError(f"Unknown action type: {action}")

        return (
            action_str,
            bgym.AgentInfo(
                extra_info={
                    "chat_model_args": asdict(self.chat_model_args),
                    "tape_segment": tape_segment,
                    "thoughts:": thoughts,
                },
                stats={},
            ),
        )
