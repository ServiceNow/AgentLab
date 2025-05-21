from abc import ABC, abstractmethod
from agentlab.llm.llm_utils import ParseError, retry
from agentlab.llm.tracking import cost_tracker_decorator
from browsergym.experiments.agent import AgentInfo
from browsergym.experiments.benchmark import Benchmark
from browsergym.utils.obs import overlay_som
from copy import copy, deepcopy
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
        self.actions = []
        self.thoughts = []

    @cost_tracker_decorator
    def get_action(self, obs: dict) -> tuple[str, dict]:
        ui_prompt = self.ui_prompt_args.make_prompt(
            obs=obs,
            actions=self.actions,
            thoughts=self.thoughts,
            extra_instructions=None,
            preliminary_answer=None,
        )
        try:
            messages = ui_prompt.get_messages()
            answer = retry(
                chat=self.main_vl_model,
                messages=messages,
                n_retry=self.max_retry,
                parser=ui_prompt.parse_answer,
            )
            stats = {"num_main_retries": (len(messages) - 3) // 2}
        except ParseError:
            answer = {"thought": None, "action": None}
            stats = {"num_main_retries": self.max_retry}
        stats.update(self.main_vl_model.get_stats())
        if self.auxiliary_vl_model is not None:
            preliminary_answer = answer
            ui_prompt = self.ui_prompt_args.make_prompt(
                obs=obs,
                actions=self.actions,
                thoughts=self.thoughts,
                extra_instructions=None,
                preliminary_answer=preliminary_answer,
            )
            try:
                messages = ui_prompt.get_messages()
                answer = retry(
                    chat=self.auxiliary_vl_model,
                    messages=messages,
                    n_retry=self.max_retry,
                    parser=ui_prompt.parse_answer,
                )
                stats["num_auxiliary_retries"] = (len(messages) - 3) // 2
            except ParseError:
                answer = {"thought": None, "action": None}
                stats["num_auxiliary_retries"] = self.max_retry
            stats.update(self.auxiliary_vl_model.get_stats())
        else:
            preliminary_answer = None
        self.thoughts.append(str(answer["thought"]))
        self.actions.append(str(answer["action"]))
        agent_info = AgentInfo(
            think=str(answer["thought"]), stats=stats, extra_info=preliminary_answer
        )
        return answer["action"], asdict(agent_info)

    @property
    def obs_preprocessor(self, obs: dict) -> dict:
        obs = copy(obs)
        if self.ui_prompt_args.use_screenshot and self.ui_prompt_args.use_screenshot_som:
            obs["screenshot"] = overlay_som(
                obs["screenshot"], extra_properties=obs["extra_element_properties"]
            )
        return obs


@dataclass
class UIAgentArgs(VLAgentArgs):
    main_vl_model_args: VLModelArgs
    auxiliary_vl_model_args: VLModelArgs
    ui_prompt_args: UIPromptArgs
    max_retry: int

    @property
    def agent_name(self) -> str:
        return f"UIAgent-{self.main_vl_model_args.model_name}-{self.auxiliary_vl_model_args.model_name}"

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
        self.ui_prompt_args.use_tabs = benchmark.is_multi_tab
        self.ui_prompt_args.action_set_args = deepcopy(benchmark.high_level_action_set_args)
        if demo_mode:
            self.ui_prompt_args.action_set_args.demo_mode = "all_blue"
