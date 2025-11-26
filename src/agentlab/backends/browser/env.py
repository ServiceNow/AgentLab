import logging
import time
from dataclasses import dataclass
from pathlib import Path

from browsergym.core.task import AbstractBrowserTask

from agentlab.actions import ToolCall, ToolsActionSet, ToolSpec
from agentlab.backends.browser.base import BrowserBackend
from agentlab.benchmarks.abstract_env import AbstractEnv, AbstractEnvArgs
from agentlab.benchmarks.web_task import AbstractWebTask

logger = logging.getLogger(__name__)


def final_step():
    """
    Finish the task execution.
    """
    return {
        "pruned_html": "Task finished",
        "axtree_txt": "",
        "last_action_error": "",
        "focused_element_bid": "",
    }


class BrowserEnv(AbstractEnv):
    def __init__(
        self, task_name: str, task: AbstractWebTask | AbstractBrowserTask, backend: BrowserBackend, seed: int = 0
    ):
        self.task_name = task_name
        self.task = task
        self.seed = seed
        self._turns = 0
        self.backend = backend
        self.backend.initialize()
        self.goal = ""
        if isinstance(self.task, AbstractBrowserTask) and not self.backend.has_pw_page:
            raise ValueError(
                "Legacy task requires a backend with direct playwright page access."
            )

    def reset(self, seed: int):
        self.seed = seed
        if isinstance(self.task, AbstractBrowserTask):
            self.goal, task_info = self.task.setup(page=self.backend.page)
            obs = self._get_obs()
        else:
            self.goal, task_info = self.task.setup(backend=self.backend) 
            obs = self._get_obs()
            obs = self.task.obs_postprocess(obs)
        return obs, task_info

    def _get_obs(self) -> dict:
        html = self.backend.page_html()
        screenshot = self.backend.page_screenshot()
        axtree = self.backend.page_axtree()
        obs = {
            "goal_object": [{"type": "text", "text": self.goal}],
            "html": html,
            "axtree_txt": axtree,
            "screenshot": screenshot,
            "last_action_error": "",
            "focused_element_bid": "",
        }
        return obs

    def step(self, action: ToolCall | str) -> tuple[dict, float, bool, bool, dict]:
        if isinstance(action, str):
            action = ToolsActionSet.parse_action(action)
        logger.info(f"BrowserEnv.step() called with action {action}")

        action_exec_start = time.time()
        done = action.name == "final_step"
        if done:
            observation = final_step()
        else:
            observation = self.backend.step(action)
        action_exec_stop = time.time()
        self._turns += 1
        if isinstance(self.task, AbstractWebTask):
            truncated = self._turns >= self.task.max_turns
        else:
            truncated = False

        observation = self.obs_postprocess(observation)

        if isinstance(self.task, AbstractBrowserTask):
            reward, done, _, info = self.task.validate(page=self.backend.page, chat_messages=[])
        elif self.task.validate_per_step or done or truncated:
            reward, info = self.task.validate()
            if info.get("done", False):
                done = True
        else:
            reward = 0.0
            info = {}

        env_info = {
            **info,
            "action_exec_start": action_exec_start,
            "action_exec_stop": action_exec_stop,
            "action_exec_timeout": 0.0
        }
        logger.info(f"Action result in observation: {observation}")
        return observation, reward, done, truncated, env_info

    def obs_postprocess(self, obs: dict) -> dict:
        if "goal_object" not in obs:
            obs["goal_object"] = [{"type": "text", "text": self.goal}]
        if "last_action_error" not in obs:
            obs["last_action_error"] = ""
        if "focused_element_bid" not in obs:
            obs["focused_element_bid"] = ""
        if isinstance(self.task, AbstractWebTask):
            obs = self.task.obs_postprocess(obs)
        return obs

    def close(self):
        self.task.teardown()

    def actions(self) -> list[ToolSpec]:
        all_actions = self.backend.actions()
        if isinstance(self.task, AbstractWebTask):
            filtered_actions = self.task.filter_actions(all_actions)
            logger.info(
                f"Filtered {len(filtered_actions)} actions out of {len(all_actions)} for dataset {self.task.dataset}"
            )
        else:
            filtered_actions = all_actions
        final_step_action = ToolSpec.from_function(final_step)
        return filtered_actions + [final_step_action]


@dataclass
class BrowserEnvArgs(AbstractEnvArgs):
    task: AbstractWebTask
    task_seed: int
    task_name: str
    backend_cls: type[BrowserBackend]

    def __init__(
        self,
        task: AbstractWebTask,
        backend_cls: type[BrowserBackend],
        task_seed: int = 0,
    ):
        self.task_name = f"{task.dataset}.{task.task_id}"
        self.task = task
        self.task_seed = task_seed
        self.backend_cls = backend_cls

    def make_env(self, exp_dir: Path) -> BrowserEnv:
        backend = self.backend_cls()
        env = BrowserEnv(
            task_name=self.task_name, task=self.task, backend=backend, seed=self.task_seed
        )
        return env
