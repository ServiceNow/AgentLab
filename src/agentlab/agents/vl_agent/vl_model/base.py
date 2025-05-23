from abc import ABC, abstractmethod
from agentlab.llm.llm_utils import AIMessage, Discussion
from torch.nn import Module


class VLModel(ABC, Module):
    @abstractmethod
    def __call__(self, messages: Discussion) -> AIMessage:
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> dict:
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
