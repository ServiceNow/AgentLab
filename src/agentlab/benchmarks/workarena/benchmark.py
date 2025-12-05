import logging
from typing import Any

from browsergym.workarena import get_all_tasks_agents
from browsergym.workarena.instance import SNowInstance
from pydantic import ConfigDict

from agentlab.actions import ToolsActionSet
from agentlab.backends.browser.base import BrowserBackend
from agentlab.backends.browser.env import BrowserEnvArgs
from agentlab.benchmarks.abstract_env import AbstractBenchmark

from .task import WorkarenaTask

logger = logging.getLogger(__name__)


class WorkArenaBenchmark(AbstractBenchmark):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend_cls: type[BrowserBackend]
    name: str = "workarena"
    level: str = "l1"
    n_seeds: int = 1
    env_args_list: list[BrowserEnvArgs] = None  # type: ignore
    dataset: list[WorkarenaTask] = None  # type: ignore
    is_multi_tab: bool = False
    high_level_action_set_args: ToolsActionSet = None  # type: ignore
    _snow_instance: SNowInstance = None  # type: ignore

    def model_post_init(self, __context: Any) -> None:
        self.name = f"workarena_{self.level}_{self.backend_cls.__name__.lower()}"
        self._snow_instance = SNowInstance()
        self.env_args_list = []
        if self.dataset is None:
            self.dataset = self.load_tasks(self.level)
        for task in self.dataset:
            env_args = BrowserEnvArgs(task=task, backend_cls=self.backend_cls)
            self.env_args_list.append(env_args)
        logger.info(f"Loaded {len(self.env_args_list)} workarena tasks")

    def load_tasks(self, level: str) -> list[WorkarenaTask]:
        task_seed_tuples = get_all_tasks_agents(filter=self.level, n_seed_l1=self.n_seeds)
        tasks = []
        for task_cls, seed in task_seed_tuples:
            task = WorkarenaTask(
                url="",
                task_id=task_cls.get_task_id(),
                instance=self._snow_instance,
                task_cls=task_cls,
                level=level,
                seed=seed,
            )
            tasks.append(task)
        logger.info(f"Loaded {len(tasks)} tasks for level {level}")
        return tasks