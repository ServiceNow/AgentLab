import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from bgym import AbstractActionSet
from dataclasses_json import DataClassJsonMixin
from desktop_env.desktop_env import DesktopEnv

from agentlab.benchmarks.abstract_env import AbstractBenchmark, AbstractEnv, AbstractEnvArgs

logger = logging.getLogger(__name__)


class OsworldGym(AbstractEnv):
    def __init__(
        self,
        task: dict,
        provider_name: str,
        region: str | None,
        path_to_vm: str | None,
        snapshot_name: str,
        action_space: str,
        cache_dir: str,
        screen_size: tuple[int, int],
        headless: bool,
        require_a11y_tree: bool,
        require_terminal: bool,
        os_type: str,
        enable_proxy: bool,
    ):
        self.task = task
        self.env_info = {
            "provider_name": provider_name,
            "region": region,
            "path_to_vm": path_to_vm,
            "snapshot_name": snapshot_name,
            "action_space": action_space,
            "cache_dir": cache_dir,
            "screen_size": screen_size,
            "headless": headless,
            "require_a11y_tree": require_a11y_tree,
            "require_terminal": require_terminal,
            "os_type": os_type,
            "enable_proxy": enable_proxy,
        }
        self.env = DesktopEnv(
            action_space=action_space,
            provider_name=provider_name,
            region=region,  # type: ignore
            path_to_vm=path_to_vm,  # type: ignore
            snapshot_name=snapshot_name,
            cache_dir=cache_dir,
            screen_size=screen_size,  # type: ignore
            headless=headless,
            require_a11y_tree=require_a11y_tree,
            require_terminal=require_terminal,
            os_type=os_type,
        )

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        obs = self.env.reset(task_config=self.task, seed=seed)
        return obs, self.env_info

    def step(self, action: str):
        raw_obs, reward, done, info = self.env.step(action)
        truncated = False
        obs = self.convert_observation(raw_obs)
        return obs, reward, done, truncated, info

    def convert_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Convert the observation to a format suitable for the generic agent."""
        # TODO: Implement conversion logic
        return obs

    def close(self):
        return self.env.close()


class OSWorldActionSet(AbstractActionSet):
    def __init__(self, action_space: Literal["computer_13", "pyautogui"]):
        self.action_space = action_space

    def describe(self, with_long_description: bool = True, with_examples: bool = True) -> str:
        # TODO: Implement a detailed description of the action set.
        return ""

    def example_action(self, abstract: bool) -> str:
        # TODO: Provide an example action
        return "example_action"

    def to_python_code(self, action) -> str:
        # TODO: Convert the given action to browsergym-compatible python code.
        return "pass"

    def to_tool_descriptor(self) -> list[dict]:
        # TODO: Convert the action set to a tool descriptor.
        return [{}]


@dataclass
class OSWorldActionSetArgs(DataClassJsonMixin):
    action_space: Literal["computer_13", "pyautogui"] = "computer_13"

    def make_action_set(self):
        logger.info(f"Creating OSWorld Action Set with action space: {self.action_space}")
        return OSWorldActionSet(action_space=self.action_space)


@dataclass
class OsworldEnvArgs(AbstractEnvArgs):
    task: dict[str, Any]
    task_seed: int = 0
    task_name: str | None = None
    path_to_vm: str | None = None  # path to .vmx file
    provider_name: str = "docker"
    region: str = "us-east-1"  # AWS specific, does not apply to all providers
    snapshot_name: str = "init_state"  # snapshot name to revert to
    action_space: Literal["computer_13", "pyautogui"] = "computer_13"
    cache_dir: str = "cache"
    screen_size: tuple[int, int] = (1920, 1080)
    headless: bool = False
    require_a11y_tree: bool = True
    require_terminal: bool = False
    os_type: str = "Ubuntu"
    enable_proxy: bool = False

    def make_env(self, exp_dir: Path, action_mapping=None, use_raw_page_output: bool = False) -> OsworldGym:
        logger.info(f"Creating OSWorld Gym with task: {self.task}")
        gym = OsworldGym(
            task=self.task,
            provider_name=self.provider_name,
            region=self.region,
            path_to_vm=self.path_to_vm,
            snapshot_name=self.snapshot_name,
            action_space=self.action_space,
            cache_dir=self.cache_dir,
            screen_size=self.screen_size,
            headless=self.headless,
            require_a11y_tree=self.require_a11y_tree,
            require_terminal=self.require_terminal,
            os_type=self.os_type,
            enable_proxy=self.enable_proxy,
        )
        return gym


class OsworldBenchmark(AbstractBenchmark):
    name: str = "osworld"
    is_multi_tab: bool = False
    high_level_action_set_args: OSWorldActionSetArgs = None  # type: ignore
    test_set_path: str = "OSWorld/evaluation_examples"
    test_set_name: str = "test_all.json"
    domain: str = "all"
    env_args: OsworldEnvArgs = None  # type: ignore # basic env configuration for all tasks
    env_args_list: list[OsworldEnvArgs] = None  # type: ignore

    def model_post_init(self, __context: Any) -> None:
        self.env_args_list = []
        if not self.env_args:
            self.env_args = OsworldEnvArgs(task={})
        self.high_level_action_set_args = OSWorldActionSetArgs(action_space=self.env_args.action_space)
        with open(os.path.join(self.test_set_path, self.test_set_name)) as f:
            tasks = json.load(f)
        if self.domain != "all":
            tasks = {self.domain: tasks[self.domain]}

        for domain in tasks:
            for task_id in tasks[domain]:
                task_file = os.path.join(self.test_set_path, f"examples/{domain}/{task_id}.json")
                with open(task_file) as f:
                    task = json.load(f)
                name = f"{self.name}.{task['id']}"
                task_env_args = deepcopy(self.env_args)
                task_env_args.task = task
                task_env_args.task_name = name
                self.env_args_list.append(task_env_args)
        logger.info(f"Loaded {len(self.env_args_list)} tasks from domain '{self.domain}'")
