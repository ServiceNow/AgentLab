import logging
from typing import Any

from agentlab.backends.browser.base import BrowserBackend
from agentlab.backends.browser.env import BrowserEnvArgs
from agentlab.benchmarks.abstract_env import AbstractBenchmark
from agentlab.benchmarks.miniwob.task import MiniWobTask, get_miniwob_tasks

logger = logging.getLogger(__name__)


class MiniWobBenchmark(AbstractBenchmark):
    backend: BrowserBackend
    name: str = "miniwob"
    env_args_list: list[BrowserEnvArgs] = None  # type: ignore
    dataset: list[MiniWobTask] = None  # type: ignore

    def model_post_init(self, __context: Any) -> None:
        self.env_args_list = []
        if self.dataset is None:
            self.dataset = get_miniwob_tasks()
        for task in self.dataset:
            name = f"miniwob.{task.task_id}"
            env_args = BrowserEnvArgs(task_name=name, task=task, backend=self.backend)
            self.env_args_list.append(env_args)
        logger.info(f"Loaded {len(self.env_args_list)} miniwob tasks")
