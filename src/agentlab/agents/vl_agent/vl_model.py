from abc import ABC, abstractmethod
from dataclasses import dataclass


class VLModel(ABC):
    @abstractmethod
    def __call__(self, messages: list[dict]) -> dict:
        pass

    def get_stats(self):
        return {}


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
        pass

    def prepare(self):
        pass

    def close(self):
        pass
