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
class VLAgent(ABC):
    action_set: AbstractActionSet

    @abstractmethod
    def get_action(self, obs: dict) -> tuple[str, dict]:
        raise NotImplementedError

    @abstractmethod
    def obs_preprocessor(self, obs: dict) -> dict:
        raise NotImplementedError


@dataclass
class VLAgentArgs(ABC):
    agent_name: str

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


class UIAgent(VLAgent):
    def __init__(
        self,
        general_vl_model_args: VLModelArgs,
        grounding_vl_model_args: VLModelArgs,
        vl_prompt_flags: VLPromptFlags,
        max_retry: int,
    ):
        self.general_vl_model = general_vl_model_args.make_model()
        self.grounding_vl_model = grounding_vl_model_args.make_model()
        self.action_set = vl_prompt_flags.action_flags.action_set.make_action_set()
        self._obs_preprocessor = dp.make_obs_preprocessor(vl_prompt_flags.obs_flags)
        self.vl_prompt_flags = vl_prompt_flags
        self.max_retry = max_retry
        self.obs_history = []
        self.actions = []
        self.thoughts = []

    @cost_tracker_decorator
    def get_action(self, obs: dict) -> tuple[str, dict]:
        self.obs_history.append(obs)
        vl_prompt = VLPrompt(
            vl_prompt_flags=self.vl_prompt_flags,
            action_set=self.action_set,
            obs_history=self.obs_history,
            actions=self.actions,
            thoughts=self.thoughts,
        )
        system_prompt = SystemMessage(dp.SystemPrompt().prompt)
        try:
            messages = Discussion([system_prompt, vl_prompt.prompt])
            answer = retry(
                chat=self.general_vl_model,
                messages=messages,
                n_retry=self.max_retry,
                parser=vl_prompt.parse_answer,
            )
            num_tries = (len(messages) - 3) / 2
            num_busted_tries = 0
        except ParseError:
            answer = {"action": None, "think": None}
            num_tries = self.max_retry + 1
            num_busted_tries = 1
        self.actions.append(answer["action"])
        self.thoughts.append(answer["think"])
        stats = {"num_tries": num_tries, "num_busted_tries": num_busted_tries}
        stats.update(self.general_vl_model.get_stats())
        stats.update(self.grounding_vl_model.get_stats())
        agent_info = AgentInfo(think=answer["think"], chat_messages=messages, stats=stats)
        return answer["action"], asdict(agent_info)

    def obs_preprocessor(self, obs: dict) -> dict:
        return self._obs_preprocessor(obs)


@dataclass
class UIAgentArgs(VLAgentArgs):
    general_vl_model_args: VLModelArgs
    grounding_vl_model_args: VLModelArgs
    vl_prompt_flags: VLPromptFlags
    max_retry: int

    def __post_init__(self):
        self.agent_name = f"ui_agent-{self.general_vl_model_args.model_name}-{self.grounding_vl_model_args.model_name}"

    def make_agent(self) -> UIAgent:
        return UIAgent(
            general_vl_model_args=self.general_vl_model_args,
            grounding_vl_model_args=self.grounding_vl_model_args,
            vl_prompt_flags=self.vl_prompt_flags,
            max_retry=self.max_retry,
        )

    def prepare(self):
        self.general_vl_model_args.prepare()
        self.grounding_vl_model_args.prepare()

    def close(self):
        self.general_vl_model_args.close()
        self.grounding_vl_model_args.close()

    def set_reproducibility_mode(self):
        self.general_vl_model_args.set_reproducibility_mode()
        self.grounding_vl_model_args.set_reproducibility_mode()

    def set_benchmark(self, benchmark: Benchmark, demo_mode: bool):
        self.vl_prompt_flags.obs_flags.use_tabs = benchmark.is_multi_tab
        self.vl_prompt_flags.action_flags.action_set = deepcopy(
            benchmark.high_level_action_set_args
        )
        if demo_mode:
            self.vl_prompt_flags.action_flags.action_set.demo_mode = "all_blue"
