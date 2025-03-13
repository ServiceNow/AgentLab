import os
from typing import Any, Literal

import bgym
import datasets
from tapeagents.environment import ContainerExecutor
from tapeagents.tools.browser import Browser
from tapeagents.tools.code_executor import CodeExecutor
from tapeagents.tools.media_reader import VideoReader
from tapeagents.tools.web_search import WebSearch

from agentlab.benchmarks.abstract_env import AbstractEnvArgs
from agentlab.benchmarks.multitool_gym import MultiToolGym


class GaiaBenchmark(bgym.Benchmark):
    name = "gaia"
    split: Literal["test", "validation"]
    exp_dir: str

    high_level_action_set_args = None
    is_multi_tab = False
    supports_parallel_seeds = False
    backends = ["gaia"]
    env_args_list = None
    task_metadata = None

    def __post_init__(self):
        super().__post_init__()
        self.env_args_list = []
        dataset = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all")[self.split]
        for task in dataset:
            task_dir = os.path.join(self.name, task["task_id"])
            env_args = GaiaGymArgs(task=task, exp_dir=task_dir)
            self.env_args_list.append(env_args)


class GaiaGym(MultiToolGym):
    task: dict
    exp_dir: str


class GaiaGymArgs(AbstractEnvArgs):
    task: dict[str, Any]
    split: Literal["test", "validation"]
    exp_dir: str
    viewport_chars: int = 64000

    def make_env(self) -> GaiaGym:
        self.init_code_sandbox()
        tools = [
            WebSearch(),
            VideoReader(self.exp_dir),
            Browser(self.exp_dir, viewport_chars=self.viewport_chars),
            CodeExecutor(self.exp_dir),
        ]
        env = GaiaGym(tools=tools, task=self.task, exp_dir=self.exp_dir)
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
