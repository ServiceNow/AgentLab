from abc import ABC, abstractmethod

import gymnasium as gym
from pydantic import BaseModel


class AbstractEnvArgs(BaseModel):
    """Easily serialiazable class to store the arguments of an environment"""

    @abstractmethod
    def make_env(self, action_mapping, exp_dir, exp_task_kwargs) -> "AbstractEnv":
        """Create an instance of the environment with the arguments stored in this object.

        Args:
            action_mapping (dict[str,str]): mapping from the agent's action space to the environment's action space
                see AbstractActionSet.to_python_code from BrowserGym for an example
            exp_dir (str): directory where the experiment is stored
            exp_task_kwargs (dict[str,Any]): additional arguments for the environment

        Returns:
            env (AbstractEnv): instance of the environment.
        """


class AbstractBenchmark(BaseModel):
    name: str
    env_args_list: list[AbstractEnvArgs]

    def get_version(self) -> int:
        return "1"

    def prepare_backends(self):
        pass

    def dependency_graph_over_tasks(self) -> dict[str, list[str]]:
        return {}


class AbstractEnv(gym.Env, ABC):
    @abstractmethod
    def reset(self, seed: int = None) -> tuple[dict[str, any], dict[str, any]]:
        """Reset the environment to the initial state, ready for an agent to start a new episode.

        Args:
            seed (int): seed to be used for the environment's random number generator. Some task may
                be deterministic and not require a seed.

        Returns:
            obs (dict[str,Any]): dictionary containing the observations
            env_info (dict[str,Any]): additional information about the environment (see step's docstring)
        """

    @abstractmethod
    def step(self, action: str):
        """Exection action in the environment and return the next observations

        Args:
            action (str): action to be executed in the environment, as a string

        Returns:
            obs (dict[str,Any]): dictionary containing the observations
            reward (float): reward obtained after executing the action
            terminated (bool): whether the episode is terminated. The MDP reached a terminal state
            truncated (bool): whether the episode is truncated. The episode was truncated due to external reasons
            env_info (dict[str,Any]): additional information about the environment
                task_info (str): Some potential debugging information about the task, not intended for the agent
                action_exec_start (float): time when the action execution started
                action_exec_stop (float): time when the action execution ended
                action_exec_timeout (float): TODO I don't remember exactly what this is
        """

    @abstractmethod
    def close(self):
        """Close any resources used by the environment"""

    @abstractmethod
    def calculate_reward(self) -> float:
        """Calculate the reward obtained in the last step"""
