from agentlab.llm.llm_utils import Discussion, ParseError, retry
from agentlab.llm.tracking import cost_tracker_decorator
from browsergym.experiments.agent import AgentInfo
from browsergym.experiments.benchmark import Benchmark
from browsergym.experiments.benchmark.base import HighLevelActionSetArgs
from copy import deepcopy
from dataclasses import asdict, dataclass
from .base import VLAgent, VLAgentArgs
from ..vl_model.base import VLModelArgs
from ..vl_prompt.ui_prompt import AuxiliaryUIPromptArgs, MainUIPromptArgs


class UIAgent(VLAgent):
    def __init__(
        self,
        main_vl_model_args: VLModelArgs,
        auxiliary_vl_model_args: VLModelArgs,
        main_ui_prompt_args: MainUIPromptArgs,
        auxiliary_ui_prompt_args: AuxiliaryUIPromptArgs,
        action_set_args: HighLevelActionSetArgs,
        max_num_retries: int,
    ):
        self.main_vl_model = main_vl_model_args.make_model()
        self.auxiliary_vl_model = auxiliary_vl_model_args.make_model()
        self.main_ui_prompt_args = main_ui_prompt_args
        self.auxiliary_ui_prompt_args = auxiliary_ui_prompt_args
        self.action_set = action_set_args.make_action_set()
        self.max_num_retries = max_num_retries
        self.screenshot_history = []
        self.think_history = []
        self.action_history = []

    @cost_tracker_decorator
    def get_action(self, obs: dict) -> tuple[str, dict]:
        answers = {}
        stats = {}
        preliminary_main_ui_prompt = self.main_ui_prompt_args.make_prompt(
            obs,
            screenshot_history=self.screenshot_history,
            think_history=self.think_history,
            action_history=self.action_history,
            action_set=self.action_set,
        )
        try:
            preliminary_main_messages = Discussion([preliminary_main_ui_prompt.message])
            preliminary_answer = retry(
                self.main_vl_model,
                messages=preliminary_main_messages,
                n_retry=self.max_num_retries,
                parser=preliminary_main_ui_prompt.parse_answer,
            )
            stats["preliminary_main_num_retries"] = (len(preliminary_main_messages) - 2) // 2
        except ParseError:
            preliminary_answer = {"main_think": None, "main_location": None}
            stats["preliminary_main_num_retries"] = self.max_num_retries
        answers.update(preliminary_answer)
        auxiliary_ui_prompt = self.auxiliary_ui_prompt_args.make_prompt(
            obs,
            screenshot_history=self.screenshot_history,
            extra_info=answers,
        )
        try:
            auxiliary_messages = Discussion([auxiliary_ui_prompt.message])
            auxiliary_answer = retry(
                self.auxiliary_vl_model,
                messages=auxiliary_messages,
                n_retry=self.max_num_retries,
                parser=auxiliary_ui_prompt.parse_answer,
            )
            stats["auxiliary_num_retries"] = (len(auxiliary_messages) - 2) // 2
        except ParseError:
            auxiliary_answer = {"auxiliary_think": None, "auxiliary_location": None}
            stats["auxiliary_num_retries"] = self.max_num_retries
        answers.update(auxiliary_answer)
        final_main_ui_prompt = self.main_ui_prompt_args.make_prompt(
            obs,
            screenshot_history=self.screenshot_history,
            think_history=self.think_history,
            action_history=self.action_history,
            action_set=self.action_set,
            extra_info=answers,
        )
        try:
            final_main_messages = Discussion([final_main_ui_prompt.message])
            final_answer = retry(
                self.main_vl_model,
                messages=final_main_messages,
                n_retry=self.max_num_retries,
                parser=final_main_ui_prompt.parse_answer,
            )
            stats["final_main_num_retries"] = (len(final_main_messages) - 2) // 2
        except ParseError:
            final_answer = {"action": None}
            stats["final_main_num_retries"] = self.max_num_retries
        answers.update(final_answer)
        stats.update(self.main_vl_model.stats)
        stats.update(self.auxiliary_vl_model.stats)
        self.screenshot_history.append(obs["screenshot"])
        self.think_history.append(str(answers["main_think"]))
        self.action_history.append(str(answers["action"]))
        agent_info = AgentInfo(think=str(answers), stats=stats)
        return answers["action"], asdict(agent_info)

    def obs_preprocessor(self, obs: dict) -> dict:
        return obs


@dataclass
class UIAgentArgs(VLAgentArgs):
    main_vl_model_args: VLModelArgs = None
    auxiliary_vl_model_args: VLModelArgs = None
    main_ui_prompt_args: MainUIPromptArgs = None
    auxiliary_ui_prompt_args: AuxiliaryUIPromptArgs = None
    action_set_args: HighLevelActionSetArgs = None
    max_num_retries: int = None

    def __post_init__(self):
        self.agent_name = f"UIAgent-{self.main_vl_model_args.model_name}-{self.auxiliary_vl_model_args.model_name}"

    def make_agent(self) -> UIAgent:
        return UIAgent(
            main_vl_model_args=self.main_vl_model_args,
            auxiliary_vl_model_args=self.auxiliary_vl_model_args,
            main_ui_prompt_args=self.main_ui_prompt_args,
            auxiliary_ui_prompt_args=self.auxiliary_ui_prompt_args,
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
        self.main_ui_prompt_args.use_tabs = benchmark.is_multi_tab
        self.action_set_args = deepcopy(benchmark.high_level_action_set_args)
        if demo_mode:
            self.action_set_args.demo_mode = "all_blue"
