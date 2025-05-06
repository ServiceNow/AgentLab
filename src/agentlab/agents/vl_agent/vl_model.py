from abc import ABC, abstractmethod
from dataclasses import dataclass


class VLModel(ABC):
    @abstractmethod
    def __call__(self, messages: list[dict]) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_stats(self):
        raise NotImplementedError


@dataclass
class VLModelArgs(ABC):
    model_name: str
    max_total_tokens: int = None
    max_input_tokens: int = None
    max_new_tokens: int = None
    temperature: float = 0.1
    vision_support: bool = False
    log_probs: bool = False

    @abstractmethod
    def make_model(self) -> VLModel:
        raise NotImplementedError

    @abstractmethod
    def prepare(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError
