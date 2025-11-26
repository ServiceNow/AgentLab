from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel

from agentlab.actions import ToolSpec
from agentlab.backends.browser import BrowserBackend


class AbstractWebTask(BaseModel, ABC):
    dataset: str
    task_id: str
    url: str
    validate_per_step: bool = False
    actions_whitelist: ClassVar[list[str]] = []
    max_turns: int = 100
    _backend: BrowserBackend = None # type: ignore

    def get_task_id(self) -> str:
        return self.task_id

    @abstractmethod
    def setup(self, backend: BrowserBackend) -> tuple[str, dict]:
        """
        Set up everything needed to execute the task.

        Args:
            page: the active playwright page.

        Returns:
            goal: str, goal of the task.
            info: dict, custom information from the task.
        """

    @abstractmethod
    def teardown(self):
        """
        Tear down the task, clean up resources if needed.

        Args:
            page: the active playwright page.
        """

    @abstractmethod
    def validate(self) -> tuple[float, dict]:
        """
        Validate the task, either per step or at the end.

        Returns:
            reward: float, the reward obtained.
            info: dict, custom information from the validation.
        """

    @abstractmethod
    def cheat(self):
        """
        Solve the task using a pre-defined solution (optional).
        """

    @classmethod
    def filter_actions(cls, actions: list[ToolSpec]) -> list[ToolSpec]:
        filtered_actions = [action for action in actions if action.function.name in cls.actions_whitelist]
        return  filtered_actions

    def obs_postprocess(self, obs: dict) -> dict:
        return obs