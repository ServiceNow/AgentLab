from dataclasses import dataclass
from typing import Any, Iterable
import logging

import bgym
from browsergym.experiments.agent import Agent, AgentInfo

from agentlab.agents.agent_args import AgentArgs


@dataclass
class CheatingAgentArgs(AgentArgs):
    """Agent that executes oracle actions from task.cheat()."""

    cheat_method: str = "cheat"
    stop_on_exhausted: bool = True
    fail_fast: bool = True

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
            fail_fast=self.fail_fast,
        )


class CheatingAgent(Agent):
    def __init__(
        self,
        action_set_args,
        cheat_method: str = "cheat",
        stop_on_exhausted: bool = True,
        fail_fast: bool = True,
    ):
        self.action_set = action_set_args.make_action_set()
        self._cheat_method = cheat_method
        self._stop_on_exhausted = stop_on_exhausted
        self._fail_fast = fail_fast
        self._env = None
        self._task = None
        self._oracle_actions = None
        self._oracle_index = 0
        self._logger = logging.getLogger(__name__)

    def set_env(self, env):
        self._env = env
        self._task = getattr(getattr(env, "unwrapped", env), "task", None)

    def _get_chat_messages(self):
        env = self._env
        if env is None:
            return None
        chat = getattr(getattr(env, "unwrapped", env), "chat", None)
        if chat is None:
            return None
        if hasattr(chat, "messages"):
            return chat.messages
        if hasattr(chat, "get_messages"):
            try:
                return chat.get_messages()
            except TypeError:
                return None
        return None

    def _get_page(self):
        env = self._env
        if env is None:
            return None
        unwrapped = getattr(env, "unwrapped", env)
        for attr in ("page", "_page", "pw_page"):
            page = getattr(unwrapped, attr, None)
            if page is not None:
                return page
        return None

    def _call_cheat(self, cheat_fn, obs):
        page = self._get_page()
        chat_messages = self._get_chat_messages()

        self._logger.debug(
            "Calling cheat() with page=%s chat_messages=%s obs_keys=%s",
            "yes" if page is not None else "no",
            "yes" if chat_messages is not None else "no",
            list(obs.keys()) if isinstance(obs, dict) else type(obs).__name__,
        )

        if page is not None and chat_messages is not None:
            try:
                return cheat_fn(page, chat_messages)
            except TypeError:
                pass
        if page is not None:
            try:
                return cheat_fn(page)
            except TypeError:
                pass
        if chat_messages is not None:
            try:
                return cheat_fn(chat_messages)
            except TypeError:
                pass
        try:
            return cheat_fn(obs, chat_messages)
        except TypeError:
            pass
        try:
            return cheat_fn(obs)
        except TypeError:
            pass
        return cheat_fn()

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

        oracle = self._call_cheat(cheat_fn, obs)

        self._logger.debug(
            "cheat() returned type=%s value_preview=%s",
            type(oracle).__name__,
            repr(oracle)[:200],
        )

        self._oracle_actions = self._extract_oracle_actions(oracle)
        self._oracle_index = 0

        if self._fail_fast and len(self._oracle_actions) == 0:
            raise RuntimeError("cheat() returned no actions; cannot proceed.")

        self._logger.debug("oracle_actions_len=%d", len(self._oracle_actions))

    def get_action(self, obs):
        self._init_oracle(obs)

        if self._oracle_index >= len(self._oracle_actions):
            action = None if self._stop_on_exhausted else ""
        else:
            action = self._oracle_actions[self._oracle_index]

        if self._fail_fast and (action is None or action == ""):
            raise RuntimeError("Oracle produced empty action; failing fast.")

        agent_info = AgentInfo(
            think="oracle",
            chat_messages=[],
            stats={"oracle_step": self._oracle_index, "oracle_len": len(self._oracle_actions)},
        )
        self._oracle_index += 1
        return action, agent_info


CHEATING_AGENT = CheatingAgentArgs()
