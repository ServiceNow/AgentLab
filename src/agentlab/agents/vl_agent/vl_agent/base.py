from abc import ABC, abstractmethod
from browsergym.experiments.benchmark import Benchmark
from dataclasses import dataclass


class VLAgent(ABC):
    @abstractmethod
    def get_action(self, obs: dict) -> tuple[str, dict]:
        raise NotImplementedError

    @abstractmethod
    def obs_preprocessor(self, obs: dict) -> dict:
        raise NotImplementedError


@dataclass
class VLAgentArgs(ABC):
    @property
    @abstractmethod
    def agent_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def make_agent(self) -> VLAgent:
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

    @abstractmethod
    def set_benchmark(self, benchmark: Benchmark, demo_mode: bool):
        raise NotImplementedError
