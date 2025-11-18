import logging
import tempfile
from dataclasses import dataclass
from typing import Any, Literal

import bgym
import hydra
from litellm import ChatCompletionThinkingBlock
from omegaconf import DictConfig
from PIL import Image
from pydantic import Field
from tapeagents.agent import Agent
from tapeagents.core import (
    Action,
    ControlFlow,
    LLMOutputParsingFailureAction,
    Observation,
    SetNextNode,
    StopStep,
    TapeMetadata,
    Thought,
)
from tapeagents.core import Tape as BaseTape
from tapeagents.llms import LLMStream
from tapeagents.nodes import FatalError, StandardNode
from tapeagents.steps import ImageObservation
from tapeagents.tool_calling import ToolSpec
from termcolor import colored

from agentlab.agents.agent_args import AgentArgs
from agentlab.backends.browser.base import ToolSpec as AgentlabToolSpec

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ExtendedMetadata(TapeMetadata):
    name: str = ""
    task: dict = {}
    terminated: bool = False
    truncated: bool = False
    reward: float = 0.0
    attempt_number: int = 0
    other: dict = {}


class AgentResponse(Thought):
    kind: Literal["agent_response"] = "agent_response"
    response: str

    def llm_view(self, **kwargs) -> str:
        return self.response


class AgentThinking(Thought):
    kind: Literal["agent_thinking"] = "agent_thinking"
    thinking: str

    def llm_view(self, **kwargs) -> str:
        return self.thinking


class Tape(BaseTape):
    metadata: ExtendedMetadata = Field(default_factory=ExtendedMetadata)  # type: ignore


class ToolCallNode(StandardNode):
    use_known_actions: bool = True
    use_function_calls: bool = True

    def generate_steps(self, agent: Agent, tape: Tape, llm_stream: LLMStream):
        new_steps = []
        for event in llm_stream:
            if event.output.get("reasoning_content"):
                logger.info(colored(f"LLM reasoning:\n{event.output.reasoning_content}", "yellow"))
                new_steps.append(AgentThinking(thinking=event.output.reasoning_content))
            if event.output.get("thinking_blocks"):
                for block in event.output.thinking_blocks:
                    if isinstance(block, ChatCompletionThinkingBlock):
                        logger.info(colored(f"LLM thinking block:\n{block}", "yellow"))
                        new_steps.append(AgentThinking(thinking=block.content))
            if event.output.content:
                logger.info(colored(f"LLM output:\n{event.output.content}", "cyan"))
                new_steps.append(AgentResponse(response=event.output.content))
            if event.output.tool_calls:
                logger.info(colored(f"LLM tool calls:\n{event.output.tool_calls}", "magenta"))
                new_steps += [
                    self.tool_call_to_step(agent, tool_call)
                    for tool_call in event.output.tool_calls
                ]
        for step in new_steps:
            yield step
            if isinstance(step, LLMOutputParsingFailureAction):
                yield SetNextNode(next_node=self.name)  # loop to the same node to retry
                break
        if not new_steps:
            raise FatalError("No completions!")
        if (
            self.next_node
            and not isinstance(new_steps[-1], StopStep)
            and not any(isinstance(step, SetNextNode) for step in new_steps)
        ):
            yield SetNextNode(next_node=self.next_node)


def load_config(config_name: str) -> DictConfig:
    with hydra.initialize(config_path="conf", version_base="1.1"):
        config = hydra.compose(config_name=config_name)
    return config


@dataclass
class TapeAgentArgs(AgentArgs):
    config: DictConfig = None  # type: ignore

    def make_agent(self, actions: tuple[ToolSpec, ...] | None) -> bgym.Agent:
        if actions is None:
            agent = hydra.utils.instantiate(self.config.agent)
        else:
            tapeagents_actions = [
                ToolSpec(**tool.model_dump()) if isinstance(tool, AgentlabToolSpec) else tool
                for tool in actions
            ]
            tools_description = "\n".join([action.description() for action in actions])
            agent = hydra.utils.instantiate(
                self.config.agent,
                known_actions=tapeagents_actions,
                tools_description=tools_description,
            )
        return TapeAgent(agent=agent)


@dataclass
class TapeAgentInfo(bgym.AgentInfo):
    thoughts: list[Thought] = None  # type: ignore


class DictObservation(Observation):
    """
    Container for wrapping old dict observation into new Observation class.
    """

    kind: Literal["dict_observation"] = "dict_observation"  # type: ignore
    content: str


class MarkdownObservation(Observation):
    def llm_view(self, **kwargs) -> str:
        return f"## Markdown:\n{self.content}"

    def short_view(self, max_chars: int = 100) -> str:
        return self.llm_view()[:max_chars]


class GoalObservation(MarkdownObservation):
    """
    Contains task goal
    """

    kind: Literal["goal_observation"] = "goal_observation"  # type: ignore
    goal: str

    def llm_view(self, **kwargs) -> str:
        return f"## Goal:\n{self.goal}"


class HTMLPage(MarkdownObservation):
    """
    Contains page content
    """

    kind: Literal["html_page"] = "html_page"
    html: str

    def llm_view(self, **kwargs) -> str:
        return f"## Page Content:\n{self.html}"


class AXTreePage(MarkdownObservation):
    """
    Contains accessibility tree
    """

    kind: Literal["ax_tree_page"] = "ax_tree_page"
    axtree: str

    def llm_view(self, **kwargs) -> str:
        return f"## Accessibility Tree:\n{self.axtree}"


class ActionResult(MarkdownObservation):
    """
    Contains action result
    """

    kind: Literal["action_result"] = "action_result"
    result: str

    def llm_view(self, **kwargs) -> str:
        return f"## Action Result:\n{self.result}"


class TapeAgent(bgym.Agent):
    agent: Agent
    tape: Tape

    def __init__(self, agent: Agent):
        super().__init__()
        self.agent = agent
        self.tape = Tape(steps=[])

    def obs_preprocessor(self, obs: Any) -> list[Observation]:
        return obs

    def obs_to_steps(self, obs: Observation | list[Observation] | dict) -> list[Observation]:
        if isinstance(obs, Observation):
            obs = [obs]
        if isinstance(obs, dict):
            obs_steps = []
            if obs.get("goal_object"):
                obs_steps.append(GoalObservation(goal=obs["goal_object"][0]["text"]))
            if obs.get("action_result"):
                obs_steps.append(ActionResult(result=obs["action_result"]))
            if obs.get("pruned_html"):
                obs_steps.append(HTMLPage(html=obs["pruned_html"]))
            if obs.get("axtree_txt"):
                obs_steps.append(AXTreePage(axtree=obs["axtree_txt"]))
            if obs.get("screenshot"):
                if isinstance(obs["screenshot"], Image.Image):
                    tmp_image_path = tempfile.mktemp(suffix=".png")
                    obs["screenshot"].save(tmp_image_path)
                    obs_steps.append(ImageObservation(image_path=tmp_image_path))
                else:
                    raise ValueError(f"Expected Image.Image, got {type(obs['screenshot'])}")
            if obs.get("last_action_error"):
                obs_steps.append(ActionResult(result=f"Action error:\n{obs['last_action_error']}"))
            assert len(obs_steps) > 0, f"Unknown dict observation, keys: {obs.keys()}"
            obs = obs_steps
        assert isinstance(obs, list), f"Expected list of Observations, got {type(obs)}"
        obs_view = "\n".join([o.short_view() for o in obs])
        logger.info(colored(f"Observations:\n{obs_view}", "green"))
        return obs

    def get_action(
        self, obs: Observation | list[Observation] | dict
    ) -> tuple[Action, TapeAgentInfo]:
        self.tape += self.obs_to_steps(obs)
        thoughts: list[Thought] = []
        action = None
        while not action:
            for event in self.agent.run(self.tape):
                if not event.step:
                    continue
                self.tape = self.tape.append(event.step)
                if isinstance(event.step, Thought) and not isinstance(event.step, ControlFlow):
                    thoughts.append(event.step)
                    logger.info(f"Thought: {event.step.llm_view()}")
                elif isinstance(event.step, Action) and not action:  # we use first action only
                    action = event.step
                    logger.info(f"Action: {action.llm_view()}")
                else:
                    # there could be control flow steps for switching nodes and if clauses
                    logger.info(f"Other step: {type(event.step)}")
        logger.info(f"Tape after run: ({len(self.tape)}) {[type(s).__name__ for s in self.tape]}")
        think_str = "\n".join([t.llm_view() for t in thoughts])
        return (action, {"thoughts": thoughts, "think": think_str})

    @property
    def final_tape(self) -> Tape:
        truncated = not any([isinstance(s, StopStep) for s in self.tape.steps])
        self.tape.metadata = ExtendedMetadata(author=self.agent.name, truncated=truncated)
        return self.tape
