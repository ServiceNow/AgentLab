from abc import ABC, abstractmethod
from agentlab.agents import dynamic_prompting as dp
from agentlab.llm.llm_utils import Discussion, ParseError, retry, SystemMessage
from agentlab.llm.tracking import cost_tracker_decorator
from browsergym.experiments.agent import AgentInfo
from browsergym.experiments.benchmark import Benchmark
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Optional
from .vl_model import VLModelArgs
from .vl_prompt import UIPromptArgs


class VLAgent(ABC):
    @abstractmethod
    def get_action(self, obs: dict) -> tuple[str, dict]:
        raise NotImplementedError

    @property
    @abstractmethod
    def obs_preprocessor(self) -> callable:
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
        main_vl_model_args: VLModelArgs,
        auxiliary_vl_model_args: Optional[VLModelArgs],
        ui_prompt_args: UIPromptArgs,
        max_retry: int,
    ):
        self.main_vl_model = main_vl_model_args.make_model()
        if auxiliary_vl_model_args is None:
            self.auxiliary_vl_model = None
        else:
            self.auxiliary_vl_model = auxiliary_vl_model_args.make_model()
        self.ui_prompt_args = ui_prompt_args
        self.max_retry = max_retry
        self.obs_history = []
        self.actions = []
        self.thoughts = []

    @cost_tracker_decorator
    def get_action(self, obs: dict) -> tuple[str, dict]:
        self.obs_history.append(obs)
        ui_prompt = self.ui_prompt_args.make_prompt(self.obs_history, self.actions, self.thoughts)
        try:
            messages = Discussion(
                [SystemMessage(dp.SystemPrompt().prompt), ui_prompt.get_message()]
            )
            answer = retry(
                chat=self.main_vl_model,
                messages=messages,
                n_retry=self.max_retry,
                parser=ui_prompt.answer_parser,
            )
            stats = {"num_main_retries": (len(messages) - 3) // 2}
        except ParseError:
            answer = {"think": None, "action": None}
            stats = {"num_main_retries": self.max_retry}
        stats.update(self.main_vl_model.get_stats())
        if self.auxiliary_vl_model is not None:
            try:
                messages = Discussion(
                    [SystemMessage(dp.SystemPrompt().prompt), ui_prompt.get_message()]
                )
                messages.add_text(f"{answer['think']}\n{answer['action']}\n")
                answer = retry(
                    chat=self.auxiliary_vl_model,
                    messages=messages,
                    n_retry=self.max_retry,
                    parser=ui_prompt.answer_parser,
                )
                stats["num_auxiliary_retries"] = (len(messages) - 3) // 2
            except ParseError:
                answer = {"action": None, "think": None}
                stats["num_auxiliary_retries"] = self.max_retry
            stats.update(self.auxiliary_vl_model.get_stats())
        self.thoughts.append(answer["think"])
        self.actions.append(answer["action"])
        agent_info = AgentInfo(think=answer["think"], chat_messages=messages, stats=stats)
        return answer["action"], asdict(agent_info)

    @property
    def obs_preprocessor(self) -> callable:
        return dp.make_obs_preprocessor(self.ui_prompt_args.obs_flags)


@dataclass
class UIAgentArgs(VLAgentArgs):
    main_vl_model_args: VLModelArgs
    auxiliary_vl_model_args: VLModelArgs
    ui_prompt_args: UIPromptArgs
    max_retry: int

    def make_agent(self) -> UIAgent:
        return UIAgent(
            main_vl_model_args=self.main_vl_model_args,
            auxiliary_vl_model_args=self.auxiliary_vl_model_args,
            ui_prompt_args=self.ui_prompt_args,
            max_retry=self.max_retry,
        )

    def prepare(self):
        self.main_vl_model_args.prepare()
        if self.auxiliary_vl_model_args is not None:
            self.auxiliary_vl_model_args.prepare()

    def close(self):
        self.main_vl_model_args.close()
        if self.auxiliary_vl_model_args is not None:
            self.auxiliary_vl_model_args.close()

    def set_reproducibility_mode(self):
        self.main_vl_model_args.set_reproducibility_mode()
        if self.auxiliary_vl_model_args is not None:
            self.auxiliary_vl_model_args.set_reproducibility_mode()

    def set_benchmark(self, benchmark: Benchmark, demo_mode: bool):
        self.ui_prompt_args.obs_flags.use_tabs = benchmark.is_multi_tab
        self.ui_prompt_args.action_flags.action_set = deepcopy(benchmark.high_level_action_set_args)
        if demo_mode:
            self.ui_prompt_args.action_flags.action_set.demo_mode = "all_blue"
