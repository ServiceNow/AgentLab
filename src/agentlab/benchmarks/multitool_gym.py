import logging
import time

from tapeagents.core import Action, Observation, StopStep, Tape
from tapeagents.environment import ToolCollectionEnvironment
from tapeagents.tools.base import StatefulTool, Tool

from agentlab.benchmarks.abstract_env import AbstractEnv

logger = logging.getLogger(__name__)
EnvTape = Tape[None, Action | Observation]


class MultiToolGym(AbstractEnv):
    def __init__(self, tools: list[Tool | StatefulTool]):
        self._env = ToolCollectionEnvironment(tools)
        self._actions = self._env.actions()

    def reset(self):
        self._env.reset()

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        logger.info(f"Gym {self.__class__.__name__} step called with action {type(action)}")
        assert isinstance(action, Action)

        action_exec_start = time.time()
        terminated = isinstance(action, StopStep)
        if terminated:
            observation = Observation()  # empty observation
        else:
            observation = self._env.step(action)
        action_exec_stop = time.time()

        reward = self.calculate_reward(action)

        truncated = False

        env_info = {
            "step_metadata": observation.metadata,
            "action_exec_start": action_exec_start,
            "action_exec_stop": action_exec_stop,
            "action_exec_timeout": 0.0,
        }
        obs_view = observation.short_view() if isinstance(observation, Observation) else observation
        logger.info(f"Gym {self.__class__.__name__} observation: {obs_view}")
        return observation, reward, terminated, truncated, env_info

    def calculate_reward(self, action: Action) -> float:
        logger.warning("Reward calculation is not implemented, returning 0")
        return 0.0

    def close(self):
        self._env.close()
