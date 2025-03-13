from typing import Annotated, Union

from pydantic import Field, TypeAdapter
from tapeagents.core import Action, Observation, Tape
from tapeagents.environment import ToolCollectionEnvironment
from tapeagents.tools.base import StatefulTool, Tool

from agentlab.benchmarks.abstract_env import AbstractEnv

EnvTape = Tape[None, Action | Observation]


class MultiToolGym(AbstractEnv):
    def __init__(self, tools: list[Tool | StatefulTool]):
        self._env = ToolCollectionEnvironment(tools)
        self._actions = self._env.actions()
        self._actions_parser: TypeAdapter = TypeAdapter(
            Annotated[Union[self._actions], Field(discriminator="kind")]
        )

    def reset(self):
        self._env.reset()

    def step(self, action: str) -> tuple[Observation, float, bool, bool, dict]:
        try:
            action_step = self._actions_parser.validate_json(action)
        except Exception:
            raise ValueError("Action must be a valid JSON dict")
        assert isinstance(action_step, Action), "{action_step.kind} is not an Action"
        observation = self._env.step(action_step)
        reward = self.calculate_reward()
        terminated = False
        truncated = False
        env_info = {"step_metadata": observation.metadata}
        return observation, reward, terminated, truncated, env_info

    def calculate_reward(self) -> float:
        return 0.0

    def close(self):
        self._env.close()
