from abc import ABC, abstractmethod
from agentlab.llm.llm_utils import Discussion
from browsergym.core.action.highlevel import HighLevelActionSet
from typing import Optional


class VLPromptPart(ABC):
    @abstractmethod
    def get_message_content(self) -> list[dict]:
        raise NotImplementedError


class VLPrompt(ABC):
    @abstractmethod
    def get_messages(self) -> Discussion:
        raise NotImplementedError

    @abstractmethod
    def parse_answer(self, answer_content: list[dict]) -> dict:
        raise NotImplementedError


class VLPromptArgs(ABC):
    @abstractmethod
    def make_prompt(
        self,
        obs: dict,
        thoughts: list[str],
        actions: list[str],
        action_set: HighLevelActionSet,
        extra_info: Optional[dict] = None,
    ) -> VLPrompt:
        raise NotImplementedError
