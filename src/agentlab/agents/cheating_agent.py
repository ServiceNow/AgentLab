from dataclasses import dataclass
from typing import Any, Iterable
import inspect
import logging

import bgym
from browsergym.experiments.agent import Agent, AgentInfo

from agentlab.agents.agent_args import AgentArgs
from agentlab.cheat_custom import ensure_cheat_custom


@dataclass
class CheatingAgentArgs(AgentArgs):
    """Agent that executes oracle actions from task.cheat_custom() or task.cheat()."""

    cheat_method: str = "cheat_custom"
    stop_on_exhausted: bool = True
    fail_fast: bool = True
    fallback_action: str = "scroll(0, 0)"
    snow_browser_timeout_ms: int | None = None

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
            fallback_action=self.fallback_action,
            snow_browser_timeout_ms=self.snow_browser_timeout_ms,
        )


class CheatingAgent(Agent):
    def __init__(
        self,
        action_set_args,
        cheat_method: str = "cheat_custom",
        stop_on_exhausted: bool = True,
        fail_fast: bool = True,
        fallback_action: str = "scroll(0, 0)",
        snow_browser_timeout_ms: int | None = None,
    ):
        self.action_set = action_set_args.make_action_set()
        self._cheat_method = cheat_method
        self._cheat_custom_only = cheat_method == "cheat_custom"
        self._stop_on_exhausted = stop_on_exhausted
        self._fail_fast = fail_fast
        self._fallback_action = fallback_action
        self._snow_browser_timeout_ms = snow_browser_timeout_ms
        self._env = None
        self._task = None
        self._oracle_actions = None
        self._oracle_index = 0
        self._mode = None  # "actions", "single", or "compositional"
        self._cheat_executed = False
        self._subtask_idx = 0
        self._subtask_count = None
        self._logger = logging.getLogger(__name__)

    def set_env(self, env):
        self._env = env
        self._task = getattr(getattr(env, "unwrapped", env), "task", None)
        if self._cheat_custom_only:
            ensure_cheat_custom(self._task)
        if self._snow_browser_timeout_ms is not None:
            try:
                import browsergym.workarena.config as wa_cfg

                wa_cfg.SNOW_BROWSER_TIMEOUT = int(self._snow_browser_timeout_ms)
                self._logger.info(
                    "Set WorkArena SNOW_BROWSER_TIMEOUT=%sms", wa_cfg.SNOW_BROWSER_TIMEOUT
                )
            except Exception as e:
                self._logger.warning("Could not set WorkArena timeout: %s", e)

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

    def _get_context(self):
        page = self._get_page()
        if page is None:
            return None
        return getattr(page, "context", None)

    def _call_cheat(self, cheat_fn, obs, subtask_idx: int | None = None):
        page = self._get_page()
        chat_messages = self._get_chat_messages()

        if page is not None and self._snow_browser_timeout_ms is not None:
            try:
                page.set_default_timeout(int(self._snow_browser_timeout_ms))
                self._logger.info(
                    "Set page default timeout to %sms", int(self._snow_browser_timeout_ms)
                )
            except Exception as e:
                self._logger.warning("Failed to set page timeout: %s", e)
        if self._snow_browser_timeout_ms is not None:
            context = self._get_context()
            if context is not None:
                try:
                    context.set_default_timeout(int(self._snow_browser_timeout_ms))
                    self._logger.info(
                        "Set context default timeout to %sms", int(self._snow_browser_timeout_ms)
                    )
                except Exception as e:
                    self._logger.warning("Failed to set context timeout: %s", e)

        self._logger.debug(
            "Calling cheat() with page=%s chat_messages=%s subtask_idx=%s obs_keys=%s",
            "yes" if page is not None else "no",
            "yes" if chat_messages is not None else "no",
            subtask_idx,
            list(obs.keys()) if isinstance(obs, dict) else type(obs).__name__,
        )

        if subtask_idx is not None:
            try:
                sig = inspect.signature(cheat_fn)
                if "subtask_idx" in sig.parameters and page is not None and chat_messages is not None:
                    return cheat_fn(page, chat_messages, subtask_idx)
            except (ValueError, TypeError):
                pass

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

    def _normalize_cheat_custom_actions(self, oracle: Any, obs) -> list[str]:
        actions = self._extract_oracle_actions(oracle)
        if not actions:
            raise RuntimeError(
                "cheat_custom() returned no actions; cannot proceed. "
                f"oracle_type={type(oracle).__name__} oracle_preview={repr(oracle)[:200]} "
                f"obs_keys={list(obs.keys()) if isinstance(obs, dict) else type(obs).__name__}"
            )
        bad = [a for a in actions if not isinstance(a, str) or a == ""]
        if bad:
            raise RuntimeError(
                f"cheat_custom() returned invalid actions: {bad[:3]} (total={len(bad)})"
            )
        return actions

    def _extract_oracle_actions(self, oracle: Any) -> list[str]:
        if oracle is None:
            return []
        if isinstance(oracle, dict):
            for key in (
                "actions",
                "action",
                "trajectory",
                "steps",
                "actions_list",
                "oracle_actions",
            ):
                if key in oracle:
                    oracle = oracle[key]
                    break
            else:
                if len(oracle) == 1:
                    oracle = next(iter(oracle.values()))
                else:
                    return []
        if isinstance(oracle, tuple) and oracle and isinstance(oracle[0], (list, tuple)):
            oracle = oracle[0]
        if isinstance(oracle, str):
            return [oracle]
        if isinstance(oracle, list):
            if len(oracle) == 0:
                return []
            if all(isinstance(x, dict) and "action" in x for x in oracle):
                return [x["action"] for x in oracle]
            if all(isinstance(x, (list, tuple)) for x in oracle):
                flat = []
                for sub in oracle:
                    flat.extend(list(sub))
                return flat
            return list(oracle)
        if isinstance(oracle, Iterable):
            return list(oracle)
        raise TypeError(f"Unsupported oracle type: {type(oracle)}")

    def _get_task_or_error(self):
        task = self._task
        if task is None and self._env is not None:
            task = getattr(getattr(self._env, "unwrapped", self._env), "task", None)
        if task is None:
            raise RuntimeError(
                "CheatingAgent needs access to env.task. Ensure the experiment loop "
                "calls agent.set_env(env) after env creation."
            )
        return task

    def _is_compositional_task(self, task) -> bool:
        return hasattr(task, "subtasks") or hasattr(task, "valid_index")

    def _get_subtask_count(self, task) -> int:
        try:
            return len(task)
        except Exception:
            pass
        if hasattr(task, "subtasks"):
            try:
                return len(task.subtasks)
            except Exception:
                pass
        return 0

    def _handle_oracle_return(self, oracle, obs):
        self._logger.info(
            "cheat() returned type=%s value_preview=%s",
            type(oracle).__name__,
            repr(oracle)[:200],
        )

        if oracle is None:
            self._logger.info(
                "cheat() returned None; using fallback action: %s", self._fallback_action
            )
            chat_messages = self._get_chat_messages()
            if isinstance(chat_messages, list) and chat_messages:
                last = chat_messages[-1]
                self._logger.info("last_chat_message=%s", repr(last)[:200])
            return False

        actions = self._extract_oracle_actions(oracle)
        if self._fail_fast and len(actions) == 0:
            page = self._get_page()
            chat_messages = self._get_chat_messages()
            page_type = type(page).__name__ if page is not None else "None"
            page_url = getattr(page, "url", None) if page is not None else None
            if isinstance(chat_messages, list):
                chat_summary = f"list(len={len(chat_messages)})"
            elif chat_messages is None:
                chat_summary = "None"
            else:
                chat_summary = type(chat_messages).__name__
            obs_keys = list(obs.keys()) if isinstance(obs, dict) else type(obs).__name__
            raise RuntimeError(
                "cheat() returned no actions; cannot proceed. "
                f"oracle_type={type(oracle).__name__} oracle_preview={repr(oracle)[:200]} "
                f"page_type={page_type} page_url={page_url} chat_messages={chat_summary} "
                f"obs_keys={obs_keys}"
            )

        if len(actions) > 0:
            self._oracle_actions = actions
            self._oracle_index = 0
            self._mode = "actions"
            self._logger.debug("oracle_actions_len=%d", len(self._oracle_actions))
            return True

        return False

    def get_action(self, obs):
        task = self._get_task_or_error()
        if self._cheat_custom_only:
            ensure_cheat_custom(task)
        cheat_fn = getattr(task, self._cheat_method, None)
        if cheat_fn is None:
            raise RuntimeError(
                f"Task {type(task).__name__} has no {self._cheat_method}() method."
            )

        if self._cheat_custom_only:
            return self._get_action_cheat_custom(obs, task, cheat_fn)

        if self._mode is None:
            self._mode = "compositional" if self._is_compositional_task(task) else "single"
            self._logger.debug("CheatingAgent mode=%s", self._mode)

        if self._mode == "actions":
            if self._oracle_index >= len(self._oracle_actions):
                action = None if self._stop_on_exhausted else ""
            else:
                action = self._oracle_actions[self._oracle_index]
            if self._fail_fast and (action is None or action == ""):
                raise RuntimeError("Oracle produced empty action; failing fast.")
            self._oracle_index += 1
        elif self._mode == "compositional":
            if self._subtask_count is None:
                self._subtask_count = self._get_subtask_count(task)
                self._logger.info("Compositional task with %s subtasks", self._subtask_count)
            if self._subtask_idx < (self._subtask_count or 0):
                self._logger.info("Cheating subtask %s/%s", self._subtask_idx + 1, self._subtask_count)
                oracle = self._call_cheat(cheat_fn, obs, subtask_idx=self._subtask_idx)
                self._subtask_idx += 1
                switched = self._handle_oracle_return(oracle, obs)
                if switched:
                    action = self._oracle_actions[self._oracle_index]
                    self._oracle_index += 1
                else:
                    action = self._fallback_action
            else:
                action = self._fallback_action
        else:
            if not self._cheat_executed:
                oracle = self._call_cheat(cheat_fn, obs)
                switched = self._handle_oracle_return(oracle, obs)
                self._cheat_executed = True
                if switched:
                    action = self._oracle_actions[self._oracle_index]
                    self._oracle_index += 1
                else:
                    action = self._fallback_action
            else:
                action = self._fallback_action

        agent_info = AgentInfo(
            think="oracle",
            chat_messages=[],
            stats={
                "oracle_step": self._oracle_index,
                "oracle_len": len(self._oracle_actions) if self._oracle_actions else 0,
            },
        )
        return action, agent_info

    def _get_action_cheat_custom(self, obs, task, cheat_fn):
        if self._oracle_actions is None or self._oracle_index >= len(self._oracle_actions):
            if self._is_compositional_task(task):
                if self._subtask_count is None:
                    self._subtask_count = self._get_subtask_count(task)
                    self._logger.info(
                        "Compositional task with %s subtasks", self._subtask_count
                    )
                if self._subtask_idx >= (self._subtask_count or 0):
                    raise RuntimeError("cheat_custom() exhausted all subtasks.")
                self._logger.info(
                    "cheat_custom subtask %s/%s",
                    self._subtask_idx + 1,
                    self._subtask_count,
                )
                oracle = self._call_cheat(cheat_fn, obs, subtask_idx=self._subtask_idx)
                self._subtask_idx += 1
            else:
                oracle = self._call_cheat(cheat_fn, obs)

            self._oracle_actions = self._normalize_cheat_custom_actions(oracle, obs)
            self._oracle_index = 0

        action = self._oracle_actions[self._oracle_index]
        if not isinstance(action, str) or action == "":
            raise RuntimeError(f"cheat_custom() produced invalid action: {action!r}")
        self._oracle_index += 1

        agent_info = AgentInfo(
            think="oracle",
            chat_messages=[],
            stats={
                "oracle_step": self._oracle_index,
                "oracle_len": len(self._oracle_actions) if self._oracle_actions else 0,
            },
        )
        return action, agent_info


CHEATING_AGENT = CheatingAgentArgs()
