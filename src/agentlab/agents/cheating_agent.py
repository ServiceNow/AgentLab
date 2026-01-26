from dataclasses import dataclass
from typing import Any, Iterable

import bgym
from browsergym.experiments.agent import Agent, AgentInfo

from agentlab.agents.agent_args import AgentArgs


@dataclass
class CheatingAgentArgs(AgentArgs):
    """Agent that executes oracle actions from task.cheat()."""

    cheat_method: str = "cheat"
    stop_on_exhausted: bool = True

    def __post_init__(self):
        try:
            self.agent_name = "CheatingAgent"
        except AttributeError:
            pass

    def set_benchmark(self, benchmark: bgym.Benchmark, demo_mode: bool):
        self.action_set_args = benchmark.high_level_action_set_args

    def make_agent(self):
        return CheatingAgent(
            action_set_args=self.action_set_args,
            cheat_method=self.cheat_method,
            stop_on_exhausted=self.stop_on_exhausted,
        )


class CheatingAgent(Agent):
    def __init__(self, action_set_args, cheat_method: str = "cheat", stop_on_exhausted: bool = True):
        self.action_set = action_set_args.make_action_set()
        self._cheat_method = cheat_method
        self._stop_on_exhausted = stop_on_exhausted
        self._env = None
        self._task = None
        self._oracle_actions = None
        self._oracle_index = 0

    def set_env(self, env):
        self._env = env
        self._task = getattr(getattr(env, "unwrapped", env), "task", None)

    def _extract_oracle_actions(self, oracle: Any) -> list[str]:
        if oracle is None:
            return []
        if isinstance(oracle, dict):
            if "actions" in oracle:
                oracle = oracle["actions"]
            elif "trajectory" in oracle:
                oracle = oracle["trajectory"]
        if isinstance(oracle, tuple) and oracle and isinstance(oracle[0], (list, tuple)):
            oracle = oracle[0]
        if isinstance(oracle, str):
            return [oracle]
        if isinstance(oracle, Iterable):
            return list(oracle)
        raise TypeError(f"Unsupported oracle type: {type(oracle)}")

    def _init_oracle(self, obs):
        if self._oracle_actions is not None:
            return

        task = self._task
        if task is None and self._env is not None:
            task = getattr(getattr(self._env, "unwrapped", self._env), "task", None)

        if task is None:
            raise RuntimeError(
                "CheatingAgent needs access to env.task. Ensure the experiment loop "
                "calls agent.set_env(env) after env creation."
            )

        cheat_fn = getattr(task, self._cheat_method, None)
        if cheat_fn is None:
            raise RuntimeError(
                f"Task {type(task).__name__} has no {self._cheat_method}() method."
            )

        try:
            oracle = cheat_fn()
        except TypeError:
            oracle = cheat_fn(obs)

        self._oracle_actions = self._extract_oracle_actions(oracle)
        self._oracle_index = 0

    def get_action(self, obs):
        self._init_oracle(obs)

        if self._oracle_index >= len(self._oracle_actions):
            action = None if self._stop_on_exhausted else ""
        else:
            action = self._oracle_actions[self._oracle_index]

        agent_info = AgentInfo(
            think="oracle",
            chat_messages=[],
            stats={"oracle_step": self._oracle_index, "oracle_len": len(self._oracle_actions)},
        )
        self._oracle_index += 1
        return action, agent_info


CHEATING_AGENT = CheatingAgentArgs()
