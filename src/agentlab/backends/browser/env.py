import logging
import time
from dataclasses import dataclass
from pathlib import Path

from agentlab.actions import ToolsActionSet
from agentlab.backends.browser.base import BrowserBackend, ToolCallAction, ToolSpec
from agentlab.benchmarks.abstract_env import AbstractEnv, AbstractEnvArgs
from agentlab.benchmarks.web_task import AbstractWebTask

logger = logging.getLogger(__name__)


def final_step():
    """
    Finish the task execution.
    """
    pass

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
            self.goal = self.task.parse_setup_result(self.backend.run_js(setup_js))
            logger.info(f"Task goal: {self.goal}")
        html = self.backend.page_html()
        screenshot = self.backend.page_screenshot()
        axtree = self.backend.page_axtree()
        obs = {
            "goal_object": [{"type": "text", "text": self.goal}],
            "pruned_html": html,
            "axtree_txt": axtree,
            "screenshot": screenshot,
            "last_action_error": "",
            "focused_element_bid": "none",
        }
        obs = self.task.obs_postprocess(obs)
        logger.info(f"Initial obs: {obs}")
        return obs, {}

    def step(self, action: ToolCallAction | str) -> tuple[dict, float, bool, bool, dict]:
        if isinstance(action, str):
            action = ToolsActionSet.parse_action(action)
        logger.info(f"BrowserEnv.step() called with action {action}")

        action_exec_start = time.time()
        finished = action.function.name == "final_step"
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
        observation = self.task.obs_postprocess(observation)

        action_exec_stop = time.time()
        self._turns += 1
        logger.info(f"Obs: {observation}")

        truncated = self._turns >= self.max_turns

        if self.task.validate_per_step or finished or truncated:
            reward, other = self.validate_task(action, observation)
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
        logger.info(f"Action result in observation: {observation}")
        return observation, reward, finished, truncated, env_info

    def _step(self, action: ToolCallAction) -> dict:
        obs_dict = self.backend.step(action)
        if "goal_object" not in obs_dict:
            obs_dict["goal_object"] = [{"type": "text", "text": self.goal}]
        if "last_action_error" not in obs_dict:
            obs_dict["last_action_error"] = ""
        if "focused_element_bid" not in obs_dict:
            obs_dict["focused_element_bid"] = "none"
        return obs_dict

    def validate_task(self, action: ToolCallAction, observation: dict) -> tuple[float, dict]:
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
        final_step_action = ToolSpec.from_function(final_step)
        filtered_actions.append(final_step_action)
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
