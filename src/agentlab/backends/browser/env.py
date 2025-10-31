import logging
import time
from typing import Any, Literal

from tapeagents.core import Action, Observation, StopStep

from agentlab.backends.browser.base import BrowserBackend
from agentlab.benchmarks.abstract_env import AbstractEnv
from agentlab.benchmarks.miniwob.task import AbstractWebTask

logger = logging.getLogger(__name__)


class PageObservation(Observation):
    kind: Literal["page_observation"] = "page_observation"
    content: str

class BrowserAction(Action):
    kind: Literal["browser_action"] = "browser_action"
    name: str
    arguments: dict[str, Any]


class BrowserEnv(AbstractEnv):
    def __init__(self, task_name: str, task: AbstractWebTask, backend: BrowserBackend, seed: int = 0):
        self.task_name = task_name
        self.task = task
        self.seed = seed
        self.backend = backend
        self._turns = 0

    def reset(self, seed: int):
        self.seed = seed
        setup_js = self.task.get_setup_js()
        if setup_js:
            js_result_str = self.backend.run_js(setup_js)
            logger.info(f"Task reset result: {js_result_str}")

    def step(self, action: BrowserAction) -> tuple[Observation, float, bool, bool, dict]:
        logger.info(f"BrowserEnv.step() called with action {type(action)}")

        action_exec_start = time.time()
        finished = isinstance(action, StopStep)
        if finished:
            observation = Observation()  # empty observation
        else:
            observation = self._step(action)
        action_exec_stop = time.time()
        self._turns += 1

        truncated = self._turns >= self.max_turns

        if self.task.validate_per_step or finished or truncated:
            reward = self.calculate_reward(action, observation)
        else:
            reward = None

        env_info = {
            "step_metadata": observation.metadata,
            "action_exec_start": action_exec_start,
            "action_exec_stop": action_exec_stop,
            "action_exec_timeout": 0.0,
        }
        obs_view = observation.short_view() if isinstance(observation, Observation) else observation
        logger.info(f"Action result in observation: {obs_view}")
        return observation, reward, finished, truncated, env_info

    def _step(self, action: Action) -> PageObservation:
        tool_result = self.backend.call_tool(action.name, action.arguments)
        return PageObservation(content=tool_result)

    def calculate_reward(self, action: Action, observation: PageObservation) -> float:
        validate_js = self.task.get_step_validate_js()
        validate_result = self.backend.run_js(validate_js)
        reward, other = self.task.parse_validation_result(validate_result)
        return reward

    def close(self):
        teardown_js = self.task.get_teardown_js()
        if teardown_js:
            js_result_str = self.backend.run_js(teardown_js)
            logger.info(f"Task teardown result: {js_result_str}")
