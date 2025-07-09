from abc import ABC, abstractmethod
from agentlab.llm.llm_utils import AIMessage, Discussion
from PIL import Image


class VLModel(ABC):
    @abstractmethod
    def __call__(self, messages: Discussion) -> AIMessage:
        raise NotImplementedError

    @abstractmethod
    def adapt_location(self, image: Image.Image, x: int, y: int) -> tuple[int, int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def stats(self) -> dict:
        raise NotImplementedError


class VLModelArgs(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def make_model(self) -> VLModel:
        raise NotImplementedError

    @abstractmethod
    def prepare(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    @abstractmethod
    def set_reproducibility_mode(self):
        raise NotImplementedError
