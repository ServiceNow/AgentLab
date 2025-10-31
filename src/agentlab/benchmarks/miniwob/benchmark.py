import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ConfigDict

from agentlab.backends.browser.base import BrowserBackend
from agentlab.backends.browser.env import BrowserEnv
from agentlab.benchmarks.abstract_env import AbstractBenchmark, AbstractEnvArgs
from agentlab.benchmarks.miniwob.task import MiniWobTask, get_miniwob_tasks

logger = logging.getLogger(__name__)


@dataclass
class MiniwobArgs(AbstractEnvArgs):
    task: MiniWobTask
    task_seed: int
    task_name: str
    backend: BrowserBackend

    def __init__(self, task_name: str, task: MiniWobTask, backend: BrowserBackend, task_seed: int = 0):
        self.task_name = task_name
        self.task = task
        self.task_seed = task_seed
        self.backend = backend

    def make_env(self, exp_dir: Path, action_mapping=None) -> BrowserEnv:
        env = BrowserEnv(task_name=self.task_name, task=self.task, backend=self.backend, seed=self.task_seed)
        return env


class MiniWobBenchmark(AbstractBenchmark):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: BrowserBackend
    name: str = "miniwob"
    env_args_list: list[MiniwobArgs] = None  # type: ignore
    dataset: list[MiniWobTask] = None  # type: ignore

    def model_post_init(self, __context: Any) -> None:
        self.env_args_list = []
        if self.dataset is None:
            self.dataset = get_miniwob_tasks()
        for task in self.dataset:
            name = f"miniwob.{task.task_id}"
            env_args = MiniwobArgs(task_name=name, task=task, backend=self.backend)
            self.env_args_list.append(env_args)
        logger.info(f"Loaded {len(self.env_args_list)} miniwob tasks")
