from agentlab.llm.llm_utils import ParseError, retry
from agentlab.llm.tracking import cost_tracker_decorator
from browsergym.experiments.agent import AgentInfo
from browsergym.experiments.benchmark import Benchmark
from browsergym.experiments.benchmark.base import HighLevelActionSetArgs
from browsergym.utils.obs import overlay_som
from copy import copy, deepcopy
from dataclasses import asdict, dataclass
from typing import Optional
from .base import VLAgent, VLAgentArgs
from ..vl_model.base import VLModelArgs
from ..vl_prompt.ui_prompt import UIPromptArgs


class UIAgent(VLAgent):
    def __init__(
        self,
        main_vl_model_args: VLModelArgs,
        auxiliary_vl_model_args: Optional[VLModelArgs],
        action_set_args: HighLevelActionSetArgs,
        ui_prompt_args: UIPromptArgs,
        max_retry: int,
    ):
        self.main_vl_model = main_vl_model_args.make_model()
        if auxiliary_vl_model_args is None:
            self.auxiliary_vl_model = None
        else:
            self.auxiliary_vl_model = auxiliary_vl_model_args.make_model()
        self.action_set = action_set_args.make_action_set()
        self.ui_prompt_args = ui_prompt_args
        self.max_retry = max_retry
        self.thoughts = []
        self.actions = []

    @cost_tracker_decorator
    def get_action(self, obs: dict) -> tuple[str, dict]:
        ui_prompt = self.ui_prompt_args.make_prompt(
            obs=obs, thoughts=self.thoughts, actions=self.actions, action_set=self.action_set
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
                thoughts=self.thoughts,
                actions=self.actions,
                action_set=self.action_set,
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
    auxiliary_vl_model_args: Optional[VLModelArgs]
    action_set_args: HighLevelActionSetArgs
    ui_prompt_args: UIPromptArgs
    max_retry: int

    @property
    def agent_name(self) -> str:
        if self.auxiliary_vl_model_args is None:
            return f"UIAgent-{self.main_vl_model_args.model_name}"
        else:
            return f"UIAgent-{self.main_vl_model_args.model_name}-{self.auxiliary_vl_model_args.model_name}"

    def make_agent(self) -> UIAgent:
        return UIAgent(
            main_vl_model_args=self.main_vl_model_args,
            auxiliary_vl_model_args=self.auxiliary_vl_model_args,
            action_set_args=self.action_set_args,
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
        self.action_set_args = deepcopy(benchmark.high_level_action_set_args)
        if demo_mode:
            self.action_set_args.demo_mode = "all_blue"
