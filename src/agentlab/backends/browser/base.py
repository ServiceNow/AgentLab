import logging
from abc import ABC, abstractmethod

from PIL import Image
from pydantic import BaseModel

from agentlab.actions import ToolCall, ToolSpec

logger = logging.getLogger(__name__)


class BrowserBackend(BaseModel, ABC):
    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def run_js(self, js: str):
        pass

    @abstractmethod
    def goto(self, url: str) -> str:
        pass

    @abstractmethod
    def page_html(self) -> str:
        pass

    @abstractmethod
    def page_screenshot(self) -> Image:
        pass

    @abstractmethod
    def page_axtree(self) -> str:
        pass

    @abstractmethod
    def step(self, action: ToolCall) -> dict:
        pass

    @abstractmethod
    def actions(self) -> tuple[ToolSpec]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
