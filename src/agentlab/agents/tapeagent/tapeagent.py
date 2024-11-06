from dataclasses import asdict, dataclass
import logging
from typing import TYPE_CHECKING, Any

import bgym


from tapeagents.agent import Agent as TapeAgent
from tapeagents.nodes import MonoNode
from tapeagents.utils import get_step_schemas_from_union_type
from tapeagent.llms import LiteLLM
from .prompts import PromptRegistry
from .steps import (
    PageObservation,
    WorkArenaAgentStep,
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

if TYPE_CHECKING:
    from agentlab.llm.chat_api import BaseModelArgs

logger = logging.getLogger(__name__)

@dataclass
class TapeAgentArgs(AgentArgs):
    chat_model_args: BaseModelArgs
    agent_name: str = "GuidedTapeAgent"

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


class WorkArenaNode(MonoNode):
    system_prompt: str = PromptRegistry.system_prompt
    steps_prompt: str = PromptRegistry.allowed_steps
    agent_step_cls = WorkArenaAgentStep

    def get_steps_description(self, tape: WorkArenaTape, agent: Any) -> str:
        return self.steps_prompt.format(allowed_steps=get_step_schemas_from_union_type(self.agent_step_cls))

    def prepare_tape(self, tape: WorkArenaTape, max_chars: int = 100):
        """
        Trim all page observations except the last two.
        """
        tape = super().prepare_tape(tape)
        page_positions = [i for i, step in enumerate(tape.steps) if isinstance(step, PageObservation)]
        if len(page_positions) < 3:
            return tape
        prev_page_position = page_positions[-2]
        steps = []
        for step in tape.steps[:prev_page_position]:
            if isinstance(step, PageObservation):
                short_text = f"{step.text[:max_chars]}\n..." if len(step.text) > max_chars else step.text
                new_step = step.model_copy(update=dict(text=short_text))
            else:
                new_step = step
            steps.append(new_step)
        steps += tape.steps[prev_page_position:]
        return tape.model_copy(update=dict(steps=steps))


class GuidedTapeAgent(bgym.Agent):
    steps: WorkArenaTape
    def __init__(self, llm: LiteLLM):
        self.tapeagent = TapeAgent.create(
            llm,
            nodes=[
                WorkArenaNode(name="set_goal", guidance=PromptRegistry.start),
                WorkArenaNode(name="reflect", guidance=PromptRegistry.reflect),
                WorkArenaNode(name="act", guidance=PromptRegistry.act, next_node="reflect"),
            ],
            max_iterations=2,
        ) 
        self.steps = WorkArenaTape(steps=[])


    @cost_tracker_decorator
    def get_action(self, obs: Any) -> tuple[str, dict]:
        # update tape with new observation
        text = flatten_axtree(obs["axtree_object"], filter_visible_only=True, ignore_navigation=False)
        obs_step = PageObservation(text=text, current_page=1,total_pages=1)
        self.tape = self.tape.append(obs_step)
        if not len(self.steps): # first observation
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
            action_str = f"click('{action.bid}', button='{action.button}', modifiers={action.modifiers})"
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

