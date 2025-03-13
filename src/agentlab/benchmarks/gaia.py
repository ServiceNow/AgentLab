import os
from typing import Literal

import datasets
from tapeagents.environment import ContainerExecutor
from tapeagents.tools.browser import Browser
from tapeagents.tools.code_executor import CodeExecutor
from tapeagents.tools.container_executor import init_code_sandbox
from tapeagents.tools.media_reader import VideoReader
from tapeagents.tools.web_search import WebSearch

from agentlab.benchmarks.abstract_env import AbstractEnvArgs
from agentlab.benchmarks.multitool_gym import MultiToolGym


class GaiaGym(MultiToolGym):
    task: dict
    exp_dir: str


class GaiaGymArgs(AbstractEnvArgs):
    task_id: str
    split: Literal["test", "validation"]
    exp_dir: str
    viewport_chars: int = 64000

    def make_env(self) -> GaiaGym:
        init_code_sandbox(self.exp_dir)
        dataset = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all")
        tasks_by_id = {task["task_id"]: task for task in dataset[self.split]}
        task = tasks_by_id[self.task_id]
        tools = [
            WebSearch(),
            VideoReader(self.exp_dir),
            Browser(self.exp_dir, viewport_chars=self.viewport_chars),
            CodeExecutor(self.exp_dir),
        ]
        env = GaiaGym(tools=tools, task=task, exp_dir=self.exp_dir)
        return env

    def init_code_sandbox(self) -> None:
        code_path = os.path.join(self.exp_dir, "code")
        os.makedirs(code_path, exist_ok=True)
        container_name = self.exp_dir.replace("/", "-")
        ContainerExecutor(
            work_dir=code_path,
            container_name=container_name,
            restart_if_exists=False,
            stop_container=False,
            no_deps=True,
        )
