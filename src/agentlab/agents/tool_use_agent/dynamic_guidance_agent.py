import logging
from copy import deepcopy
from typing import Any
import bgym
from agentlab.agents.tool_use_agent.cua_like_agent import (
    ADDITIONAL_ACTION_INSTRUCTIONS,
    GeneralHints,
    Goal,
    Obs,
    PromptConfig,
    StructuredDiscussion,
    Summarizer,
    TaskHint,
    action_from_generalized_bgym_action_tool,
    simple_bgym_action_tool,
)
from agentlab.benchmarks.abstract_env import AbstractBenchmark as AgentLabBenchmark
from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.litellm_api import LiteLLMModelArgs
from agentlab.llm.response_api import APIPayload, LLMOutput
from bgym import Benchmark as BgymBenchmark
from browsergym.core.observation import extract_screenshot

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

USER_AGENT_SUMMARY_SYS_MSG = """You are an agent in a multi-agent system for the ServiceNow platform. You are impersonate a real end user who is navigating ServiceNow and asking for help. Your task is to navigate the ServiceNow platform by asking guidance from the **Dynamic Guidance** agent.

For each navigation step, you receive:
- a screenshot of the *previous* UI state,
- a screenshot of the *current* UI state,
- and a short description of the *last action* taken.

Using this context:
- Briefly explain how the last action changed the interface or workflow (if at all).  
- Then ask—conversationally and in the first person—what you should do next. Keep the request generic and concise. Do not specify any implied goal in your ask, just ask what is the next step.

Write your response as a single concise paragraph inside `<summary>...</summary>` tags. Keep it natural, like you're chatting with someone helping you. Keep it under 40 words."""

USER_AGENT_ACTION_SYS_MSG = """You are an agent in a multi-agent system for the ServiceNow platform. You are impersonate a real end user who is navigating ServiceNow. Your role at this stage is to execute an action based on guidance you've received.

You are provided:
- a screenshot showing the current UI state, and
- an instruction from the **Dynamic Guidance** agent.

Using the screenshot to understand what is visible and actionable on the interface, interpret the instruction and carry out the corresponding action. If using tools is required, invoke them appropriately based on the instruction and what the screenshot allows.

Respond only with the action you perform—no explanation."""

DYNAMIC_GUIDANCE_AGENT_PLANNING_SYS_MSG = """You are the **Dynamic Guidance** agent in a multi-agent ServiceNow system. 
Your role is to help an end-user achieve their goal by analyzing what is currently visible in the UI and deciding what they should do next.

Using the current screenshot and the goal:
- Create a short plan describing the steps needed to achieve the goal.
- Suggest the next action the user should take right now.

Be factual, avoid assumptions beyond what is visible, and provide only relevant guidance.
Your output must use the following structure:
<plan>...</plan>
<guidance>...</guidance>
"""

DYNAMIC_GUIDANCE_AGENT_GUIDANCE_SYS_MSG = """You are the **Dynamic Guidance** agent in a multi-agent ServiceNow system. 
Your role is to help an end-user achieve their goal by analyzing what is currently visible in the UI and deciding what they should do next.

Using the current screenshot and the user query, suggest the next action the user should take right now.

Be factual, avoid assumptions beyond what is visible, and provide only relevant guidance.
Your output must use the following structure:
<guidance>...</guidance>
"""

class DynamicGuidanceAgentArgs(AgentArgs):
    model_args: BaseModelArgs = None
    config: PromptConfig = None
    use_raw_page_output: bool = False  # This attribute is used in loop.py to setup the env.
    action_set: bgym.AbstractActionSet | None = None

    def __post_init__(self):
        try:
            self.agent_name = f"DG-{self.model_args.model_name}".replace("/", "_")
        except AttributeError:
            pass

    def make_agent(self) -> bgym.Agent:
        if self.config is None:
            self.config = DEFAULT_PROMPT_CONFIG
        return DynamicGuidanceAgent(
            model_args=self.model_args,  # type: ignore
            config=self.config,
            action_set=self.action_set,
        )

    def prepare(self):
        return self.model_args.prepare_server()

    def close(self):
        return self.model_args.close_server()

    def set_benchmark(self, benchmark: AgentLabBenchmark | BgymBenchmark, demo_mode: bool):
        """Set benchmark specific flags."""
        benchmark_name = benchmark.name
        if benchmark_name == "osworld":
            self.config.obs.skip_preprocessing = True

        self.config.obs.use_tabs = benchmark.is_multi_tab
        benchmark_action_set = (
            deepcopy(benchmark.high_level_action_set_args).make_action_set().action_set
        )
        # these actions are added based on the benchmark action set
        if "send_msg_to_user" in benchmark_action_set:
            self.config.action_subsets += ("chat",)
        if "report_infeasible" in benchmark_action_set:
            self.config.action_subsets += ("infeas",)

class DynamicGuidanceAgent(bgym.Agent):
    def __init__(
        self,
        model_args: LiteLLMModelArgs,
        config: PromptConfig = None,
        action_set: bgym.AbstractActionSet | None = None,
    ):
        self.model_args = model_args
        self.config = config
        self.action_set: bgym.AbstractActionSet = action_set or bgym.HighLevelActionSet(
            self.config.action_subsets,
            multiaction=self.config.multiaction,  # type: ignore
        )
        if self.config.use_generalized_bgym_action_tool:
            self.tools = [simple_bgym_action_tool]
        else:
            self.tools = self.action_set.to_tool_description(api=model_args.api)

        self.call_ids = []

        self.llm = model_args.make_model()
        self.msg_builder = model_args.get_message_builder()
        self.llm.msg = self.msg_builder

        self.task_hint = self.config.task_hint.make()
        self.obs_block = self.config.obs.make()

        self.discussion = StructuredDiscussion(self.config.keep_last_n_obs)
        self.last_response: LLMOutput = LLMOutput()
        self._responses: list[LLMOutput] = []

        self.is_goal_set = False

    def obs_preprocessor(self, obs):
        obs = deepcopy(obs)
        if self.config.obs.skip_preprocessing:
            return obs
        page = obs.pop("page", None)
        if page is not None:
            obs["screenshot"] = extract_screenshot(page)
        else:
            raise Exception("No page found in observation")
        return obs

    def set_task_name(self, task_name: str):
        self.task_name = task_name

    @cost_tracker_decorator
    def get_action(self, obs: Any) -> float:
        self.llm.reset_stats()

        if not self.is_goal_set:
            # 0. First LLM call if goal is not set is to prompt the dynamic guidance
            #    agent given the current state and the goal.
            dynamic_guidance_messages = [
                {"role": "system", "content": DYNAMIC_GUIDANCE_AGENT_GUIDANCE_SYS_MSG},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "[SCREENSHOT]",
                        },
                        {"type": "input_image", "image_url": f"data:image/png;base64,{after_image_base64}"},
                        {
                            "type": "input_text",
                            "text": f"[GOAL]\n{self.config.goal}",
                        },
                    ],
                },
            ]

        else:
            # 1: based on observation and the last user llm action,
            #    the user llm asks something like "what should I do next?"
            summary_messages = [
                {"role": "system", "content": USER_AGENT_SUMMARY_SYS_MSG},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "[PREVIOUS SCREENSHOT]",
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{before_image_base64}",
                        },
                        {
                            "type": "input_text",
                            "text": "[CURRENT SCREENSHOT]",
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{after_image_base64}",
                        },
                        {
                            "type": "input_text",
                            "text": f"[LAST_ACTION]\n{last_action}",
                        },
                    ],
                },
            ]

            summary_response = self.llm(
                APIPayload(
                    messages=summary_messages,
                    reasoning_effort="low",
                )
            )
            summary_response_text = summary_response.output_text

            # 2: based on the summary response, the agent llm instructs the user llm to perform a given action (in natural language)
            dynamic_guidance_messages = [
                {"role": "system", "content": DYNAMIC_GUIDANCE_AGENT_GUIDANCE_SYS_MSG},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "[SCREENSHOT]",
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{after_image_base64}",
                        },
                        {
                            "type": "input_text",
                            "text": f"[QUERY]\n{summary_response_text}",
                        },
                    ],
                },
            ]

        # 2: using the screenshot of the current page, and given its original goal
        #    the agent llm instructs the user llm to perform a given action (in natural language)
        dynamic_guidance_response: LLMOutput = self.llm(
            APIPayload(
                messages=dynamic_guidance_messages,
                cache_tool_definition=True,
                cache_complete_prompt=False,
                use_cache_breakpoints=True,
                reasoning_effort="low",
            )
        )

        dynamic_guidance_response_text = dynamic_guidance_response.output_text

        # 3: based on the current screenshot (for grounding) and the agent llm instruction ONLY,
        #    the user llm performs one of the provided actions
        user_action_messages = [
            {"role": "system", "content": USER_AGENT_ACTION_SYS_MSG},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "[SCREENSHOT]",
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{after_image_base64}",
                    },
                    {
                        "type": "input_text",
                        "text": f"[INSTRUCTION]\n{dynamic_guidance_response_text}",
                    },
                ],
            },
        ]

        user_action_response: LLMOutput = self.llm(
            APIPayload(
                messages=user_action_messages,
                tools=self.tools,
                tool_choice="auto",  # auto must be enabled to get reasoning for tool calls.
                reasoning_effort="low",
            )
        )

        agent_info = bgym.AgentInfo(
            think=think,
            chat_messages=messages,
            stats=self.llm.stats.stats_dict,
        )
        return action, agent_info


DYNAMIC_GUIDANCE_PROMPT_CONFIG = PromptConfig(
    tag_screenshot=True,
    goal=Goal(goal_as_system_msg=True),
    obs=Obs(
        use_last_error=True,
        use_screenshot=True,
        use_axtree=False,
        use_dom=False,
        use_som=False,
        use_tabs=False,
        overlay_mouse_action=True,
    ),
    summarizer=Summarizer(do_summary=False),
    general_hints=GeneralHints(use_hints=False),
    task_hint=TaskHint(use_task_hint=False),
    action_subsets=("coord",),
    keep_last_n_obs=5,  # max 20 no more than 20 screenshots for claude
)


def get_dynamic_guidance_agent_config(model_name: str) -> DynamicGuidanceAgentArgs:

    return DynamicGuidanceAgentArgs(
        model_args=LiteLLMModelArgs(
            model_name=model_name,
            max_new_tokens=2000,
            temperature=None,
        ),
        config=DYNAMIC_GUIDANCE_PROMPT_CONFIG,
    )
