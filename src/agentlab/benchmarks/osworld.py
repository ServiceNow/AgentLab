import json
import logging
import os
from dataclasses import dataclass
from typing import Any

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
        obs, reward, done, info = self.env.step(action)
        truncated = False
        return obs, reward, done, truncated, info

    def close(self):
        return self.env.close()


@dataclass
class OsworldEnvArgs(AbstractEnvArgs):
    task: dict[str, Any]
    task_seed:int = 0
    task_name: str | None = None
    path_to_vm: str | None = None
    provider_name: str = "vmware"  # path to .vmx file
    region: str = "us-east-1"  # AWS specific, does not apply to all providers
    snapshot_name: str = "init_state"  # snapshot name to revert to
    action_space: str = "computer_13"  # "computer_13" | "pyautogui"
    cache_dir: str = "cache"
    screen_size: tuple[int, int] = (1920, 1080)
    headless: bool = False
    require_a11y_tree: bool = True
    require_terminal: bool = False
    os_type: str = "Ubuntu"
    enable_proxy: bool = False

    def make_env(self) -> OsworldGym:
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
    high_level_action_set_args: dict = {}
    test_set_path: str = "OSWorld/evaluation_examples"
    test_set_name: str = "test_all.json"
    domain: str = "all"
    env_args: OsworldEnvArgs = None  # type: ignore # basic env configuration for all tasks
    env_args_list: list[OsworldEnvArgs] = None  # type: ignore

    def model_post_init(self, __context: Any) -> None:
        self.env_args_list = []
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
                if self.env_args:
                    env_args = self.env_args.copy()
                    env_args.task = task
                    env_args.task_name = name
                else:
                    env_args = OsworldEnvArgs(task=task, task_name=name)
                self.env_args_list.append(env_args)
        logger.info(f"Loaded {len(self.env_args_list)} tasks from domain '{self.domain}'")
