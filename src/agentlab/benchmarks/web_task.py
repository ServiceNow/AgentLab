from typing import ClassVar

from pydantic import BaseModel

from agentlab.backends.browser.base import ToolSpec


class AbstractWebTask(BaseModel):
    dataset: str
    url: str
    validate_per_step: bool = False
    actions_whitelist: ClassVar[list[str]] = []
    max_turns: int = 100

    @classmethod
    def filter_actions(cls, actions: list[ToolSpec]) -> list[str]:
        return [action for action in actions if action.function.name in cls.actions_whitelist]

    def get_setup_js(self) -> str:
        raise NotImplementedError

    def parse_setup_result(self, setup_result: str | dict | list) -> str:
        raise NotImplementedError

    def get_teardown_js(self) -> str:
        raise NotImplementedError

    def get_task_validate_js(self) -> str:
        raise NotImplementedError

    def get_step_validate_js(self) -> str:
        raise NotImplementedError

    def parse_validation_result(self, validate_result: str) -> tuple[float, dict]:
        raise NotImplementedError

    def obs_postprocess(self, obs: dict) -> dict:
        return obs
