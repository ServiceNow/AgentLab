from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


class AbstractChatModel(ABC):
    @abstractmethod
    def __call__(self, messages: list[dict]) -> dict:
        pass

    def get_stats(self):
        return {}


@dataclass
class BaseModelArgsVLLM(ABC):
    """Base class for all model arguments."""
 
    model_name: str
    max_total_tokens: int = None
    max_input_tokens: int = None
    max_new_tokens: int = None
    temperature: float = 0.1
    vision_support: bool = False
    log_probs: bool = False
    top_p: float = 1.0 # Added top_p
    stop_sequences: Optional[List[str]] = None # Added stop_sequences
 
    @abstractmethod
    def make_model(self) -> AbstractChatModel:
        pass
 
    def prepare_server(self):
        pass
 
    def close_server(self):
        pass


@dataclass
class BaseModelArgs(ABC):
    """Base class for all model arguments."""

    model_name: str
    max_total_tokens: int = None
    max_input_tokens: int = None
    max_new_tokens: int = None
    temperature: float = 0.1
    vision_support: bool = False
    log_probs: bool = False

    @abstractmethod
    def make_model(self) -> AbstractChatModel:
        pass

    def prepare_server(self):
        pass

    def close_server(self):
        pass
