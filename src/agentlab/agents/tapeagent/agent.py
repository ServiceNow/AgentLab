import logging
from dataclasses import dataclass
from typing import Any

import bgym
import hydra
from tapeagents.agent import Agent
from tapeagents.core import Action, Observation, Tape, Thought

from agentlab.agents.agent_args import AgentArgs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class TapeAgentArgs(AgentArgs):
    agent_name: str

    def make_agent(self) -> bgym.Agent:
        with hydra.initialize(config_path="conf", version_base="1.1"):
            config = hydra.compose(config_name=self.agent_name)
        agent: Agent = hydra.utils.instantiate(config)
        return TapeAgent(agent=agent, tape=Tape(steps=[]))


@dataclass
class TapeAgentInfo(bgym.AgentInfo):
    thoughts: list[Thought] = None


class TapeAgent(bgym.Agent):
    agent: Agent
    tape: Tape

    def __init__(self, agent: Agent, tape: Tape):
        super().__init__()
        self.agent = agent
        self.tape = tape

    def obs_preprocessor(self, obs: Any) -> Any:
        logger.info(f"Observation: {obs}")
        return obs

    def get_action(self, obs: Observation | list[Observation]) -> tuple[str, TapeAgentInfo]:
        if isinstance(obs, Observation):
            obs = [obs]
        for observation in obs:
            logger.info(f"Add observation: {type(observation)}")
            self.tape = self.tape.append(observation)
        thoughts = []
        action = None
        while not action:
            for event in self.agent.run(self.tape):
                if event.final_tape:
                    logger.info(
                        f"agent run final tape state: {[type(s).__name__ for s in self.tape]}"
                    )
                if not event.step:
                    continue
                self.tape = self.tape.append(event.step)
                if isinstance(event.step, Thought):
                    thoughts.append(event.step.llm_dict())
                    logger.info(f"Thought: {event.step.llm_view()}")
                elif isinstance(event.step, Action) and not action:
                    action = event.step
                    logger.info(f"Action: {action}")
                    # we stop at the first action
                else:
                    logger.info(f"Other step: {type(event.step)}")
        logger.info(f"Tape state: {[type(s).__name__ for s in self.tape]}")
        return (action, TapeAgentInfo(thoughts=thoughts))
