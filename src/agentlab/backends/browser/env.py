import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tapeagents.core import Action, Observation, StopStep
from tapeagents.tool_calling import ToolCallAction, ToolSpec

from agentlab.actions import ToolsActionSet
from agentlab.backends.browser.base import BrowserBackend
from agentlab.benchmarks.abstract_env import AbstractEnv, AbstractEnvArgs
from agentlab.benchmarks.web_task import AbstractWebTask

logger = logging.getLogger(__name__)


class GoalObservation(Observation):
    kind: Literal["goal_observation"] = "goal_observation"
    goal: str


class PageObservation(Observation):
    kind: Literal["page_observation"] = "page_observation"
    content: str


class BrowserEnv(AbstractEnv):
    def __init__(
        self, task_name: str, task: AbstractWebTask, backend: BrowserBackend, seed: int = 0
    ):
        self.task_name = task_name
        self.task = task
        self.seed = seed
        self._turns = 0
        self.max_turns = task.max_turns
        self.backend = backend
        self.backend.initialize()
        self.goal = ""

    def reset(self, seed: int):
        self.seed = seed
        logger.info(f"Open task URL: {self.task.url}")
        self.backend.goto(self.task.url)
        setup_js = self.task.get_setup_js()
        if setup_js:
            js_out = self.backend.run_js(setup_js)
            out_dict = json.loads(js_out)
            logger.info(f"Task setup result: {out_dict}")
            goal = out_dict["goal"]
            done = out_dict["done"]
            task_start_time = out_dict["task_start_time"]
            logger.info(f"Task start time: {task_start_time}")
            if done:
                raise ValueError("Task is already done")
            self.goal = goal
            logger.info(f"Task goal: {self.goal}")
        page_content = self.backend.page_snapshot()
        logger.info(f"Initial obs: {page_content}")
        return {
            "goal_object": [{"type": "text", "text": self.goal}],
            "pruned_html": page_content,
            "axtree_txt": "",
            "last_action_error": "",
            "focused_element_bid": "none",
        }, {}

    def step(self, action: ToolCallAction | str) -> tuple[Observation, float, bool, bool, dict]:
        if isinstance(action, str):
            action = ToolsActionSet.parse_action(action)
        logger.info(f"BrowserEnv.step() called with action {action}")

        action_exec_start = time.time()
        finished = isinstance(action, StopStep)
        if finished:
            observation = {
                "goal_object": [{"type": "text", "text": self.goal}],
                "pruned_html": "Task finished",
                "axtree_txt": "",
                "last_action_error": "",
                "focused_element_bid": "none",
            }
        else:
            observation = self._step(action)
        action_exec_stop = time.time()
        self._turns += 1
        logger.info(f"Obs:\n{observation['pruned_html']}")

        truncated = self._turns >= self.max_turns

        if self.task.validate_per_step or finished or truncated:
            reward, other = self.calculate_reward(action, observation)
            if other.get("done", False):
                finished = True
        else:
            reward = 0.0
            other = {}

        env_info = {
            "action_exec_start": action_exec_start,
            "action_exec_stop": action_exec_stop,
            "action_exec_timeout": 0.0,
        } | other
        obs_view = observation.short_view() if isinstance(observation, Observation) else observation
        logger.info(f"Action result in observation: {obs_view}")
        return observation, reward, finished, truncated, env_info

    def _step(self, action: ToolCallAction) -> dict:
        tool_result = self.backend.step(action)
        return {
            "goal_object": [{"type": "text", "text": self.goal}],
            "pruned_html": tool_result,
            "axtree_txt": "",
            "last_action_error": "",
            "focused_element_bid": "none",
        }

    def calculate_reward(self, action: Action, observation: PageObservation) -> tuple[float, dict]:
        validate_js = self.task.get_step_validate_js()
        validate_result = self.backend.run_js(validate_js)
        reward, other = self.task.parse_validation_result(validate_result)
        return reward, other

    def close(self):
        teardown_js = self.task.get_teardown_js()
        if teardown_js:
            js_result_str = self.backend.run_js(teardown_js)
            logger.info(f"Task teardown result: {js_result_str}")
        self.backend.close()

    def actions(self) -> list[ToolSpec]:
        all_actions = self.backend.actions()
        filtered_actions = self.task.filter_actions(all_actions)
        logger.info(
            f"Filtered {len(filtered_actions)} actions out of {len(all_actions)} for task {self.task.dataset}"
        )
        return filtered_actions


@dataclass
class BrowserEnvArgs(AbstractEnvArgs):
    task: AbstractWebTask
    task_seed: int
    task_name: str
    backend: BrowserBackend

    def __init__(
        self, task_name: str, task: AbstractWebTask, backend: BrowserBackend, task_seed: int = 0
    ):
        self.task_name = task_name
        self.task = task
        self.task_seed = task_seed
        self.backend = backend

    def make_env(self, exp_dir: Path) -> BrowserEnv:
        env = BrowserEnv(
            task_name=self.task_name, task=self.task, backend=self.backend, seed=self.task_seed
        )
        return env
