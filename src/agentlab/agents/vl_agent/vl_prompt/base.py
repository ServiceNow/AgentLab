from abc import ABC, abstractmethod
from agentlab.llm.llm_utils import HumanMessage
from browsergym.core.action.highlevel import HighLevelActionSet
from PIL import Image
from typing import Optional, Union
import numpy as np


class VLPromptPart(ABC):
    @abstractmethod
    def get_message_content(self) -> list[dict]:
        raise NotImplementedError


class VLPrompt(ABC):
    @abstractmethod
    def get_message(self) -> HumanMessage:
        raise NotImplementedError

    @abstractmethod
    def parse_answer(self, answer_content: list[dict]) -> dict:
        raise NotImplementedError


class VLPromptArgs(ABC):
    @abstractmethod
    def make_prompt(
        self,
        obs: dict,
        screenshot_history: list[Union[Image.Image, np.ndarray]],
        thought_history: list[str],
        action_history: list[str],
        action_set: HighLevelActionSet,
        extra_info: Optional[dict] = None,
    ) -> VLPrompt:
        raise NotImplementedError
