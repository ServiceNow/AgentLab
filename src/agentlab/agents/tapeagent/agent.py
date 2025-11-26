import logging
from dataclasses import dataclass
from typing import Literal

import bgym
import hydra
from omegaconf import DictConfig
from pydantic import Field
from tapeagents.agent import Agent
from tapeagents.core import Action, Observation, StopStep, TapeMetadata, Thought
from tapeagents.core import Tape as BaseTape

from agentlab.agents.agent_args import AgentArgs

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


class Tape(BaseTape):
    metadata: ExtendedMetadata = Field(default_factory=ExtendedMetadata)  # type: ignore


def load_config(config_name: str) -> DictConfig:
    with hydra.initialize(config_path="conf", version_base="1.1"):
        config = hydra.compose(config_name=config_name)
    return config


@dataclass
class TapeAgentArgs(AgentArgs):
    config: DictConfig = None  # type: ignore

    def make_agent(self) -> bgym.Agent:
        agent: Agent = hydra.utils.instantiate(self.config.agent)
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


class TapeAgent(bgym.Agent):
    agent: Agent
    tape: Tape

    def __init__(self, agent: Agent):
        super().__init__()
        self.agent = agent
        self.tape = Tape(steps=[])

    def obs_preprocessor(self, obs: Observation | list[Observation]) -> list[Observation]:
        if isinstance(obs, Observation):
            obs = [obs]
        assert isinstance(obs, list), f"Expected list of Observations, got {type(obs)}"
        logger.info(f"Observations: {[type(o).__name__ for o in obs]}")
        return obs

    def get_action(self, obs: Observation | list[Observation]) -> tuple[Action, TapeAgentInfo]:
        self.tape += obs  # type: ignore
        thoughts: list[Thought] = []
        action = None
        while not action:
            for event in self.agent.run(self.tape):
                if not event.step:
                    continue
                self.tape = self.tape.append(event.step)
                if isinstance(event.step, Thought):
                    thoughts.append(event.step)
                    logger.info(f"Thought: {event.step.llm_view()}")
                elif isinstance(event.step, Action) and not action:  # we use first action only
                    action = event.step
                    logger.info(f"Action: {action.llm_view()}")
                else:
                    # there could be control flow steps for switching nodes and if clauses
                    logger.info(f"Other step: {type(event.step)}")
        logger.info(f"Tape after run: ({len(self.tape)}) {[type(s).__name__ for s in self.tape]}")
        return (action, TapeAgentInfo(thoughts=thoughts))

    @property
    def final_tape(self) -> Tape:
        truncated = not any([isinstance(s, StopStep) for s in self.tape.steps])
        self.tape.metadata = ExtendedMetadata(author=self.agent.name, truncated=truncated)
        return self.tape
