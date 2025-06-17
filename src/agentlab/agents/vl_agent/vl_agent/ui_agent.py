from agentlab.llm.llm_utils import Discussion, ParseError, retry
from agentlab.llm.tracking import cost_tracker_decorator
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.experiments.agent import AgentInfo
from browsergym.experiments.benchmark import Benchmark
from browsergym.experiments.benchmark.base import HighLevelActionSetArgs
from copy import copy, deepcopy
from dataclasses import asdict, dataclass
from .base import VLAgent, VLAgentArgs
from ..vl_model.base import VLModelArgs
from ..vl_prompt.ui_prompt import UIPromptArgs


class UIAgent(VLAgent):
    def __init__(
        self,
        main_vl_model_args: VLModelArgs,
        auxiliary_vl_model_args: VLModelArgs,
        ui_prompt_args: UIPromptArgs,
        action_set_args: HighLevelActionSetArgs,
        max_num_retries: int,
    ):
        self.main_vl_model = main_vl_model_args.make_model()
        self.auxiliary_vl_model = auxiliary_vl_model_args.make_model()
        self.ui_prompt_args = ui_prompt_args
        self._action_set = action_set_args.make_action_set()
        self.max_num_retries = max_num_retries
        self.screenshot_history = []
        self.thought_history = []
        self.action_history = []

    @property
    def action_set(self) -> HighLevelActionSet:
        return self._action_set

    @cost_tracker_decorator
    def get_action(self, obs: dict) -> tuple[str, dict]:
        stats = {}
        preliminary_main_ui_prompt = self.ui_prompt_args.make_prompt(
            obs,
            screenshot_history=self.screenshot_history,
            thought_history=self.thought_history,
            action_history=self.action_history,
            action_set=self.action_set,
        )
        try:
            messages = Discussion([preliminary_main_ui_prompt.message])
            preliminary_answer = retry(
                self.main_vl_model,
                messages=messages,
                n_retry=self.max_num_retries,
                parser=preliminary_main_ui_prompt.parse_answer,
            )
            stats["preliminary_main_num_retries"] = (len(messages) - 2) // 2
        except ParseError:
            preliminary_answer = {"thought": None, "location": None}
            stats["preliminary_main_num_retries"] = self.max_num_retries
        auxiliary_ui_prompt = self.ui_prompt_args.make_prompt(
            obs,
            screenshot_history=self.screenshot_history,
            thought_history=self.thought_history,
            action_history=self.action_history,
            action_set=self.action_set,
            extra_info=preliminary_answer,
        )
        try:
            messages = Discussion([auxiliary_ui_prompt.message])
            auxiliary_answer = retry(
                self.auxiliary_vl_model,
                messages=messages,
                n_retry=self.max_num_retries,
                parser=auxiliary_ui_prompt.parse_answer,
            )
            stats["auxiliary_num_retries"] = (len(messages) - 2) // 2
        except ParseError:
            auxiliary_answer = {"coordinates": None}
            stats["auxiliary_num_retries"] = self.max_num_retries
        preliminary_answer.update(auxiliary_answer)
        final_main_ui_prompt = self.ui_prompt_args.make_prompt(
            obs,
            screenshot_history=self.screenshot_history,
            thought_history=self.thought_history,
            action_history=self.action_history,
            action_set=self.action_set,
            extra_info=preliminary_answer,
        )
        try:
            messages = Discussion([final_main_ui_prompt.message])
            final_answer = retry(
                self.main_vl_model,
                messages=messages,
                n_retry=self.max_num_retries,
                parser=final_main_ui_prompt.parse_answer,
            )
            stats["final_main_num_retries"] = (len(messages) - 2) // 2
        except ParseError:
            final_answer = {"action": None}
            stats["final_main_num_retries"] = self.max_num_retries
        stats.update(self.main_vl_model.get_stats())
        stats.update(self.auxiliary_vl_model.get_stats())
        self.screenshot_history.append(obs["screenshot"])
        self.thought_history.append(str(preliminary_answer["thought"]))
        self.action_history.append(str(final_answer["action"]))
        agent_info = AgentInfo(stats=stats, extra_info=preliminary_answer)
        return final_answer["action"], asdict(agent_info)

    def obs_preprocessor(self, obs: dict) -> dict:
        obs = copy(obs)
        return obs


@dataclass
class UIAgentArgs(VLAgentArgs):
    main_vl_model_args: VLModelArgs
    auxiliary_vl_model_args: VLModelArgs
    ui_prompt_args: UIPromptArgs
    action_set_args: HighLevelActionSetArgs
    max_num_retries: int

    @property
    def agent_name(self) -> str:
        return f"UIAgent-{self.main_vl_model_args.model_name}-{self.auxiliary_vl_model_args.model_name}"

    def make_agent(self) -> UIAgent:
        return UIAgent(
            main_vl_model_args=self.main_vl_model_args,
            auxiliary_vl_model_args=self.auxiliary_vl_model_args,
            ui_prompt_args=self.ui_prompt_args,
            action_set_args=self.action_set_args,
            max_num_retries=self.max_num_retries,
        )

    def prepare(self):
        self.main_vl_model_args.prepare()
        self.auxiliary_vl_model_args.prepare()

    def close(self):
        self.main_vl_model_args.close()
        self.auxiliary_vl_model_args.close()

    def set_reproducibility_mode(self):
        self.main_vl_model_args.set_reproducibility_mode()
        self.auxiliary_vl_model_args.set_reproducibility_mode()

    def set_benchmark(self, benchmark: Benchmark, demo_mode: bool):
        self.ui_prompt_args.use_tabs = benchmark.is_multi_tab
        self.action_set_args = deepcopy(benchmark.high_level_action_set_args)
        if demo_mode:
            self.action_set_args.demo_mode = "all_blue"
