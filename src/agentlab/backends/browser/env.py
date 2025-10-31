import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tapeagents.core import Action, Observation, StopStep
from tapeagents.tool_calling import ToolCallAction, ToolSpec

from agentlab.backends.browser.base import BrowserBackend
from agentlab.benchmarks.abstract_env import AbstractEnv, AbstractEnvArgs
from agentlab.benchmarks.miniwob.task import AbstractWebTask

logger = logging.getLogger(__name__)

class GoalObservation(Observation):
    kind: Literal["goal_observation"] = "goal_observation"
    goal: str

class PageObservation(Observation):
    kind: Literal["page_observation"] = "page_observation"
    content: str


class BrowserEnv(AbstractEnv):
    def __init__(self, task_name: str, task: AbstractWebTask, backend: BrowserBackend, seed: int = 0):
        self.task_name = task_name
        self.task = task
        self.seed = seed
        self._turns = 0
        self.backend = backend
        self.backend.initialize()

    def reset(self, seed: int):
        self.seed = seed
        logger.info(f"Open task URL: {self.task.url}")
        page_content = self.backend.goto(self.task.url)
        setup_js = self.task.get_setup_js()
        if setup_js:
            js_result_str = self.backend.run_js(setup_js)
            logger.info(f"Task reset result: {js_result_str}")
        return [GoalObservation(goal=js_result_str), PageObservation(content=page_content)], {}

    def step(self, action: ToolCallAction) -> tuple[Observation, float, bool, bool, dict]:
        logger.info(f"BrowserEnv.step() called with action {action.function.name}")

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

    def _step(self, action: ToolCallAction) -> PageObservation:
        tool_result = self.backend.step(action)
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

    def actions(self) -> list[ToolSpec]:
        all_actions = self.backend.actions()
        filtered_actions = self.task.filter_actions(all_actions)
        logger.info(f"Filtered {len(filtered_actions)} actions out of {len(all_actions)} for task {self.task.dataset}")
        return filtered_actions


@dataclass
class BrowserEnvArgs(AbstractEnvArgs):
    task: AbstractWebTask
    task_seed: int
    task_name: str
    backend: BrowserBackend

    def __init__(self, task_name: str, task: AbstractWebTask, backend: BrowserBackend, task_seed: int = 0):
        self.task_name = task_name
        self.task = task
        self.task_seed = task_seed
        self.backend = backend

    def make_env(self, exp_dir: Path) -> BrowserEnv:
        env = BrowserEnv(task_name=self.task_name, task=self.task, backend=self.backend, seed=self.task_seed)
        return env

