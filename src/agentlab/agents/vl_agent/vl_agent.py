from dataclasses import asdict, dataclass
from browsergym.experiments.agent import Agent, AgentInfo
from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.llm_utils import Discussion, ParseError, SystemMessage, retry
from agentlab.llm.tracking import cost_tracker_decorator
from .vl_model import VLModelArgs
from .vl_prompt import VLPromptFlags, VLPrompt
import bgym


@dataclass
class VLAgentArgs(AgentArgs):
    vl_model_args: VLModelArgs = None
    vl_prompt_flags: VLPromptFlags = None
    max_retry: int = None

    def __post_init__(self):
        self.agent_name = f"VLAgent-{self.vl_model_args.model_name}".replace("/", "_")

    def set_benchmark(self, benchmark: bgym.Benchmark, demo_mode):
        self.vl_prompt_flags.obs.use_tabs = benchmark.is_multi_tab

    def set_reproducibility_mode(self):
        self.vl_model_args.temperature = 0

    def prepare(self):
        return self.vl_model_args.prepare()

    def close(self):
        return self.vl_model_args.close()

    def make_agent(self):
        return VLAgent(
            vl_model_args=self.vl_model_args,
            vl_prompt_flags=self.vl_prompt_flags,
            max_retry=self.max_retry,
        )


class VLAgent(Agent):
    def __init__(self, vl_model_args: VLModelArgs, vl_prompt_flags: VLPromptFlags, max_retry: int):
        self.vl_model_args = vl_model_args
        self.vl_prompt_flags = vl_prompt_flags
        self.max_retry = max_retry
        self.vl_model = vl_model_args.make_model()
        self.action_set = vl_prompt_flags.action_flags.action_set.make_action_set()
        self._obs_preprocessor = dp.make_obs_preprocessor(vl_prompt_flags.obs_flags)
        self.reset(seed=None)

    def obs_preprocessor(self, obs: dict) -> dict:
        return self._obs_preprocessor(obs)

    @cost_tracker_decorator
    def get_action(self, obs):
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

    def reset(self, seed=None):
        self.seed = seed
        self.obs_history = []
        self.actions = []
        self.thoughts = []
