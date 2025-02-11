# TODO: functional
# 1. The llm for different modules should be different
# 2. Now there's an additonal call of the "actor"
# 3. Support passed in `memory`, `state`, and `action_set`
# 4. fast_reward() is dummy now
# TODO: refactor
# 1. Cluster actions prompt is now in utils.py, should be stored separately
# 2. Can we using AgentLab native llm for token counting?

import logging
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

from browsergym.core.action.highlevel import HighLevelActionSet
# AgentLab & BrowserGym imports
from browsergym.experiments.agent import Agent, AgentInfo
from browsergym.experiments.benchmark import Benchmark, HighLevelActionSetArgs
from reasoners.algorithm.mcts import MCTSNode
from reasoners.lm.openai_model_w_parser import OpenAIModel
from reasoners.visualization import visualize

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.agent_args import AgentArgs
from agentlab.agents.generic_agent.generic_agent import (GenericAgent,
                                                         GenericAgentArgs)
from agentlab.llm.chat_api import (BaseModelArgs, make_system_message,
                                   make_user_message)
from agentlab.llm.llm_utils import Discussion, ParseError, SystemMessage, retry
from agentlab.llm.tracking import cost_tracker_decorator

from .agent_model.agent_prompts import \
    encoder_prompt_template as DefaultStateEncoderPromptTemplate
# Reasoners imports
from .agent_model.modules import PromptedEncoder
from .agent_model.variables import (AgentInstructionEnvironmentIdentity,
                                    BrowserActionSpace,
                                    BrowserGymObservationSpace,
                                    StepKeyValueMemory)
from .planner import Planner, PlannerArgs
# Adaptive AgentLab imports
from .prompts import ActorPrompt, EncoderPrompt, PlanAgentPromptFlags
from .utils import image_to_jpg_base64_url, llm_response_parser

logger = logging.getLogger(__name__)


# TODO: move to planner class or some other file
# TODO: add parameter for visualization
def node_data_factory(x: MCTSNode):
    if not x.state:
        return {}

    return {
        "step_idx": int(x.state["step_idx"]),
        "action_history": x.state["action_history"],
        "summary_state": x.state["summary_state"],
    }


def edge_data_factory(x: MCTSNode):
    if not x.state:
        return {}

    return {
        "step_idx": int(x.state["step_idx"]),
        "action_history": x.state["action_history"],
        "summary_state": x.state["summary_state"],
    }


@dataclass
class PlanAgentArgs(GenericAgentArgs):
    flags: PlanAgentPromptFlags = None
    planner_args: PlannerArgs = None
    action_set_args: HighLevelActionSetArgs = None
    max_retry: int = 1

    def __post_init__(self):
        logger.info(f"PlanAgentArgs.__post_init__: {self.action_set_args}")
        try:  # some attributes might be temporarily args.CrossProd for hyperparameter generation
            self.agent_name = f"PlanAgent-{self.chat_model_args.model_name}".replace("/", "_")
        except AttributeError:
            pass

    def make_agent(self):
        return PlanAgent(
            chat_model_args=self.chat_model_args,
            planner_args=self.planner_args,
            action_set_args=self.action_set_args,
            flags=self.flags,
            max_retry=self.max_retry,
        )


class PlanAgent(GenericAgent):

    def __init__(
        self,
        chat_model_args: BaseModelArgs,
        planner_args: PlannerArgs,
        action_set_args: HighLevelActionSetArgs,
        flags: PlanAgentPromptFlags,
        max_retry: int = 4,
    ):
        super().__init__(chat_model_args, flags, max_retry)
        self.planner_args = planner_args
        self.action_set = action_set_args.make_action_set()
        self._make_misc()
        self.planner = Planner.from_args(args=planner_args, identity=self.agent_identity)

    def _make_misc(self):
        """Misc init; should be components imported from llm-reasoners, currently under review."""
        # TODO: Contain overlap concept with AgentLab, e.g., `memory`, `agent_identity`(i.e., obs space+action space). Can they be compatible?
        self.observation_space = BrowserGymObservationSpace()
        agent_name = "Web Browsing Agent"
        agent_description = "An information and automation assistant who responds to \
user instructions by browsing the internet. The assistant strives to answer each question \
accurately, thoroughly, efficiently, and politely, and to be forthright when it is \
impossible to answer the question or carry out the instruction. The assistant will \
end the task once it sends a message to the user."
        self.agent_identity = AgentInstructionEnvironmentIdentity(
            agent_name=agent_name,
            agent_description=agent_description,
            observation_space=self.observation_space,
            action_space=self.action_set.describe(with_long_description=True, with_examples=True),
        )
        if self.flags.use_intent_only_memory:
            self.memory = StepKeyValueMemory(["intent"])
        else:
            self.memory = StepKeyValueMemory(["state", "intent"])

        if self.flags.use_state_encoder:
            self.state_encoder = PromptedEncoder(
                self.agent_identity,
                OpenAIModel(
                    model=self.planner_args.llm_args.model,
                    backend=self.planner_args.llm_args.backend,
                    is_instruct_model=self.planner_args.llm_args.is_instruct_model,
                    max_tokens=self.planner_args.llm_args.max_tokens,
                    temperature=self.planner_args.llm_args.temperature,
                    additional_prompt=self.planner_args.llm_args.additional_prompt,
                ),
                prompt_template=DefaultStateEncoderPromptTemplate,
                parser=partial(llm_response_parser, keys=["state"]),
            )

    def obs_preprocessor(self, obs: dict) -> Any:
        # Optionally override this method to customize observation preprocessing
        # The output of this method will be fed to the get_action method and also saved on disk.
        return super().obs_preprocessor(obs)

    @cost_tracker_decorator
    def get_action(self, obs):
        # Processing obs
        self.obs_history.append(obs)
        obs = self.obs_preprocessor(obs)

        # Update the agent identity with the user instruction
        self.agent_identity.update(user_instruction=obs["goal_object"][0]["text"])

        # Encode (summarize) the obs into a textual state, to be used by the planner
        # TODO: The state encoder might not be a necessity.
        if self.flags.use_state_encoder:
            summary_state = self.state_encoder(
                obs["axtree_txt"], image_to_jpg_base64_url(obs["screenshot_som"]), self.memory
            )["state"]
        else:
            raise ValueError("State encoder is not enabled. To be implemented.")

        logger.debug(f"Summarizing state done. Here is the state: \n{summary_state}")

        output = self.planner(summary_state, self.memory)

        action_plan = output["action_plan"]
        plan_full_result = output["plan_full_result"]

        action_plan = "\n".join([f"{i}. {step}" for i, step in enumerate(action_plan, start=1)])

        logger.info(f"Planning done. Here is the action plan: \n{action_plan}")

        # Built prompts for the actor, which generates the actual next action
        max_prompt_tokens, max_trunc_itr = self._get_maxes()
        system_prompt = SystemMessage(dp.SystemPrompt().prompt)
        actor_prompt = ActorPrompt(
            action_set=self.action_set,
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            previous_plan=self.plan,
            step=self.plan_step,
            flags=self.flags,
            action_plan=action_plan,  # added
        )
        actor_human_prompt = dp.fit_tokens(
            shrinkable=actor_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.chat_model_args.model_name,
            max_iterations=max_trunc_itr,
            additional_prompts=system_prompt,
        )

        logger.debug(f"Actor Human Prompt (actor): \n{actor_human_prompt}")

        try:
            actor_chat_messages = Discussion([system_prompt, actor_human_prompt])
            actor_ans_dict = retry(
                self.chat_llm,
                actor_chat_messages,
                n_retry=self.max_retry,
                parser=actor_prompt._parse_answer,
            )
            actor_ans_dict["busted_retry"] = 0
            actor_ans_dict["n_retry"] = (len(actor_chat_messages) - 3) / 2
        except ParseError as e:
            actor_ans_dict = dict(
                action=None,
                n_retry=self.max_retry + 1,
                busted_retry=1,
            )
        stats = self.chat_llm.get_stats()
        stats["n_retry"] = actor_ans_dict["n_retry"]
        stats["busted_retry"] = actor_ans_dict["busted_retry"]

        self.plan = action_plan
        # self.plan_step = actor_ans_dict.get("step", self.plan_step)  # seems redundant in this context
        self.actions.append(actor_ans_dict.get("action", None))
        self.memories.append(actor_ans_dict.get("memory", None))
        self.thoughts.append(actor_ans_dict.get("think", None))

        logger.debug(f"Thoughts: {self.thoughts}")
        logger.debug(f"Actions: {self.actions}")
        logger.debug(f"Memories: {self.memories}")
        logger.debug(f"Plan: {self.plan}")
        logger.debug(f"Plan step: {self.plan_step}")

        agent_info = AgentInfo(
            think=actor_ans_dict.get("think", None),
            chat_messages=actor_chat_messages,
            stats=stats,
            extra_info={
                "chat_model_args": asdict(self.chat_model_args),
                "visualizer": visualize(
                    plan_full_result,
                    open_browser=False,
                    node_data_factory=node_data_factory,
                    edge_data_factory=edge_data_factory,
                ),
            },
        )
        return actor_ans_dict["action"], agent_info
