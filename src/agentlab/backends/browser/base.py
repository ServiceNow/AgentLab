import logging
from abc import ABC, abstractmethod

from PIL import Image
from pydantic import BaseModel

from agentlab.actions import ToolCall, ToolSpec

logger = logging.getLogger(__name__)


class BrowserBackend(BaseModel, ABC):
    has_pw_page: bool = False

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def evaluate_js(self, js: str) -> str | dict | list:
        return ""

    @abstractmethod
    def goto(self, url: str) -> str:
        pass

    @abstractmethod
    def page_html(self) -> str:
        pass

    @abstractmethod
    def page_screenshot(self) -> Image.Image:
        pass

    @abstractmethod
    def page_axtree(self) -> str:
        pass

    @abstractmethod
    def step(self, action: ToolCall) -> dict:
        pass

    @abstractmethod
    def actions(self) -> list[ToolSpec]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @property
    def page(self):
        raise NotImplementedError("Direct access to the playwright page is not supported.")


class AsyncBrowserBackend(BaseModel):
    """Abstract base class for async browser backends."""

    has_pw_page: bool = False

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def evaluate_js(self, js: str) -> str | dict | list:
        pass

    @abstractmethod
    async def goto(self, url: str) -> None:
        pass

    @abstractmethod
    async def page_html(self) -> str:
        pass

    @abstractmethod
    async def page_screenshot(self) -> Image.Image:
        pass

    @abstractmethod
    async def page_axtree(self) -> str:
        pass

    @abstractmethod
    async def step(self, action: ToolCall) -> dict:
        pass

    @abstractmethod
    def actions(self) -> list[ToolSpec]:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    @property
    def page(self):
        raise NotImplementedError("Direct access to the playwright page is not supported.")
