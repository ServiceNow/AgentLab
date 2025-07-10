from abc import ABC, abstractmethod
from agentlab.llm.llm_utils import HumanMessage
from PIL import Image
from typing import Callable, Optional, Union
import numpy as np


class VLPromptPart(ABC):
    @property
    @abstractmethod
    def message_content(self) -> list[dict]:
        raise NotImplementedError


class VLPrompt(ABC):
    @property
    @abstractmethod
    def message(self) -> HumanMessage:
        raise NotImplementedError

    @abstractmethod
    def parse_answer(self, answer_content: list[dict]) -> dict:
        raise NotImplementedError


class VLPromptArgs(ABC):
    @abstractmethod
    def make_main_prompt(
        self,
        obs: dict,
        screenshot_history: list[Union[Image.Image, np.ndarray]],
        think_history: list[str],
        action_history: list[str],
        action_set_description: str,
        action_validator: Callable,
        extra_info: Optional[dict] = None,
    ) -> VLPrompt:
        raise NotImplementedError

    @abstractmethod
    def make_auxiliary_prompt(
        self,
        obs: dict,
        screenshot_history: list[Union[Image.Image, np.ndarray]],
        location_adapter: Callable,
        extra_info: dict,
    ) -> VLPrompt:
        raise NotImplementedError
