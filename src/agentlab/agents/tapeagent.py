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
        with hydra.initialize(config_path="../../../conf"):
            config = hydra.compose(config_name=self.agent_name)
        agent: Agent = hydra.utils.instantiate(config)
        return TapeAgent(agent=agent, tape=Tape(steps=[]))


class TapeAgent(bgym.Agent):
    agent: Agent
    tape: Tape

    def __init__(self, agent: Agent, tape: Tape):
        super().__init__()
        self.agent = agent
        self.tape = tape

    def obs_preprocessor(self, obs: dict) -> Any:
        logger.info(f"Preprocessing observation: {obs}")
        return obs

    def get_action(self, obs: Observation) -> tuple[str, bgym.AgentInfo]:
        self.tape = self.tape.append(obs)
        thoughts = []
        for event in self.agent.run(self.tape):
            if not event.step:
                continue
            self.tape = self.tape.append(event.step)
            if isinstance(event.step, Thought):
                thoughts.append(event.step.llm_view())
                logger.info(f"Thought: {event.step.llm_view()}")
            elif isinstance(event.step, Action):
                action = event.step.llm_view()
                logger.info(f"Action: {action}")
                break  # we stop at the first action
        return (action, bgym.AgentInfo(think="\n".join(thoughts), stats={}))
