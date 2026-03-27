import logging
from typing import Any

from pydantic import ConfigDict

from agentlab.actions import ToolsActionSet
from agentlab.backends.browser.base import BrowserBackend
from agentlab.backends.browser.env import BrowserEnvArgs
from agentlab.benchmarks.abstract_env import AbstractBenchmark
from agentlab.benchmarks.miniwob.task import MiniWobTask, get_miniwob_tasks

logger = logging.getLogger(__name__)


class MiniWobBenchmark(AbstractBenchmark):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend_cls: type[BrowserBackend]
    name: str = "miniwob"
    env_args_list: list[BrowserEnvArgs] = None  # type: ignore
    dataset: list[MiniWobTask] = None  # type: ignore
    is_multi_tab: bool = False
    high_level_action_set_args: ToolsActionSet = None  # type: ignore

    def model_post_init(self, __context: Any) -> None:
        self.name = f"miniwob_{self.backend_cls.__name__.lower()}"
        self.env_args_list = []
        if self.dataset is None:
            self.dataset = get_miniwob_tasks()
        for task in self.dataset:
            env_args = BrowserEnvArgs(task=task, backend_cls=self.backend_cls)
            self.env_args_list.append(env_args)
        logger.info(f"Loaded {len(self.env_args_list)} miniwob tasks")
