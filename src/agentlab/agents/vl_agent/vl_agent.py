from abc import ABC, abstractmethod
from agentlab.agents import dynamic_prompting as dp
from agentlab.llm.llm_utils import Discussion, ParseError, retry, SystemMessage
from agentlab.llm.tracking import cost_tracker_decorator
from browsergym.core.action.base import AbstractActionSet
from browsergym.experiments.agent import AgentInfo
from browsergym.experiments.benchmark import Benchmark
from copy import deepcopy
from dataclasses import asdict, dataclass
from .vl_model import VLModelArgs
from .vl_prompt import VLPrompt, VLPromptFlags


@dataclass
class VLAgentArgs(ABC):
    agent_name: str

    @abstractmethod
    def make_agent(self):
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


@dataclass
class VLAgent(ABC):
    action_set: AbstractActionSet

    @abstractmethod
    def get_action(self, obs: dict):
        raise NotImplementedError

    @abstractmethod
    def obs_preprocessor(self, obs: dict):
        raise NotImplementedError


@dataclass
class UIAgentArgs(VLAgentArgs):
    general_model_args: VLModelArgs
    grounding_model_args: VLModelArgs
    prompt_flags: VLPromptFlags
    max_retry: int

    def __post_init__(self):
        self.agent_name = (
            f"ui_agent-{self.general_model_args.model_name}-{self.grounding_model_args.model_name}"
        )

    def make_agent(self):
        return UIAgent(
            general_model_args=self.general_model_args,
            grounding_model_args=self.grounding_model_args,
            prompt_flags=self.prompt_flags,
            max_retry=self.max_retry,
        )

    def prepare(self):
        self.general_model_args.prepare()
        self.grounding_model_args.prepare()

    def close(self):
        self.general_model_args.close()
        self.grounding_model_args.close()

    def set_reproducibility_mode(self):
        self.general_model_args.set_reproducibility_mode()
        self.grounding_model_args.set_reproducibility_mode()

    def set_benchmark(self, benchmark: Benchmark, demo_mode: bool):
        self.prompt_flags.obs_flags.use_tabs = benchmark.is_multi_tab
        self.prompt_flags.action_flags.action_set = deepcopy(benchmark.high_level_action_set_args)
        if demo_mode:
            self.prompt_flags.action_flags.action_set.demo_mode = "all_blue"


class UIAgent(VLAgent):
    def __init__(
        self,
        general_model_args: VLModelArgs,
        grounding_model_args: VLModelArgs,
        prompt_flags: VLPromptFlags,
        max_retry: int,
    ):
        self.general_model_args = general_model_args
        self.grounding_model_args = grounding_model_args
        self.prompt_flags = prompt_flags
        self.max_retry = max_retry
        self.general_model = general_model_args.make_model()
        self.grounding_model = grounding_model_args.make_model()
        self.action_set = prompt_flags.action_flags.action_set.make_action_set()
        self._obs_preprocessor = dp.make_obs_preprocessor(prompt_flags.obs_flags)
        self.obs_history = []
        self.actions = []
        self.thoughts = []

    @cost_tracker_decorator
    def get_action(self, obs: dict):
        self.obs_history.append(obs)
        vl_prompt = VLPrompt(
            prompt_flags=self.prompt_flags,
            action_set=self.action_set,
            obs_history=self.obs_history,
            actions=self.actions,
            thoughts=self.thoughts,
        )
        system_prompt = SystemMessage(dp.SystemPrompt().prompt)
        try:
            chat_messages = Discussion([system_prompt, vl_prompt.prompt])
            answer = retry(
                self.vl_model,
                chat_messages,
                n_retry=self.max_retry,
                parser=vl_prompt.parse_answer,
            )
            answer["busted_retry"] = 0
            answer["n_retry"] = (len(chat_messages) - 3) / 2
        except ParseError:
            answer = dict(
                action=None,
                n_retry=self.max_retry + 1,
                busted_retry=1,
            )
        stats = self.vl_model.get_stats()
        stats["n_retry"] = answer["n_retry"]
        stats["busted_retry"] = answer["busted_retry"]
        self.actions.append(answer["action"])
        self.thoughts.append(answer.get("think", None))
        agent_info = AgentInfo(
            think=answer.get("think", None),
            chat_messages=chat_messages,
            stats=stats,
            extra_info={"vl_model_args": asdict(self.vl_model_args)},
        )
        return answer["action"], agent_info

    def obs_preprocessor(self, obs: dict):
        return self._obs_preprocessor(obs)
