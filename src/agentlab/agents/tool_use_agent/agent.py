import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import bgym
import numpy as np
from browsergym.core.observation import extract_screenshot
from PIL import Image, ImageDraw

from agentlab.agents import agent_utils
from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.response_api import (
    BaseModelArgs,
    ClaudeResponseModelArgs,
    LLMOutput,
    Message,
    OpenAIChatModelArgs,
    OpenAIResponseModelArgs,
    OpenRouterModelArgs,
    VLLMModelArgs,
)
from agentlab.llm.tracking import cost_tracker_decorator


@dataclass
class ToolUseAgentFlags:
    use_first_obs: bool = True
    tag_screenshot: bool = True
    add_thoughts_to_history: bool = True


@dataclass
class ToolUseAgentArgs(AgentArgs):
    model_args: BaseModelArgs = None
    flags: ToolUseAgentFlags = ToolUseAgentFlags
    use_raw_page_output: bool = True # This attribute is used in loop.py to setup the env.

    def __post_init__(self):
        try:
            self.agent_name = f"ToolUse-{self.model_args.model_name}".replace("/", "_")
        except AttributeError:
            pass

    def make_agent(self) -> bgym.Agent:
        return ToolUseAgent(
            model_args=self.model_args,
            flags=self.flags,
        )

    def prepare(self):
        return self.model_args.prepare_server()

    def close(self):
        return self.model_args.close_server()


class ToolUseAgent(bgym.Agent):
    def __init__(
        self,
        model_args: OpenAIResponseModelArgs,
        flags: ToolUseAgentFlags = ToolUseAgentFlags,
    ):
        self.model_args = model_args
        self.flags = flags
        self.action_set = bgym.HighLevelActionSet(["coord"], multiaction=False)
        self.tools = self.action_set.to_tool_description(api=model_args.api)

        self.call_ids = []
        self.llm = model_args.make_model(
            tools=self.tools
        )  #  Passing tools like may need changing if we want on-demand tools for agents
        self.msg_builder = model_args.get_message_builder()
        self.messages: list[Message] = []
        self.all_responses = []

    def obs_preprocessor(self, obs):
        page = obs.pop("page", None)
        if page is not None:
            obs["screenshot"] = extract_screenshot(page)
            if self.flags.tag_screenshot:
                screenshot = Image.fromarray(obs["screenshot"])
                screenshot = agent_utils.tag_screenshot_with_action(screenshot, obs["last_action"])
                obs["screenshot_tag"] = np.array(screenshot)
        else:
            raise ValueError("No page found in the observation.")

        return obs



    @cost_tracker_decorator
    def get_action(self, obs: Any) -> float:

        if len(self.messages) == 0:
            self.messages += self.get_initalize_messages(obs)
            self.messages += [
                self.msg_builder.user().add_text("Remember, You can call only one tool at a time.")
            ]
        else:

            if obs["last_action_error"] != "":
                self.messages.append(
                    self.msg_builder.user().add_text(f"Last Action Error:{obs['last_action_error']}")
                )

            if self.flags.add_thoughts_to_history:
                self.messages.append(self.msg_builder.assistant().add_text(f"{self.last_llm_output.think}\n"))
            # noop() is the default action in the absence of tool calls.
            if self.last_llm_output.tool_calls is not None:
                # Handle tool calls
                self.messages += [self.last_llm_output.tool_calls]
                self.messages += [
                    self.msg_builder.tool(self.last_llm_output.raw_response).add_text(
                        "See observation below."
                    )
                ]
            screenshot_key = "screenshot_tag" if self.flags.tag_screenshot else "screenshot"
            self.messages += [self.msg_builder.user().add_image_url(obs[screenshot_key])]

        llm_output: "LLMOutput" = self.llm(messages=self.messages)

        self.last_llm_output = llm_output
        self.all_responses.append(llm_output.raw_response)
        action = llm_output.action
        think = llm_output.think

        return (
            action,
            bgym.AgentInfo(
                think=think,
                chat_messages=self.messages,
                stats={},
            ),
        )

    def get_initalize_messages(self, obs: Any) -> None:
        sys_msg = self.msg_builder.system().add_text(
            "You are an web-agent. Based on the observation, you will decide which action to take to accomplish your goal."
        )

        goal_message = self.msg_builder.user()
        for content in obs["goal_object"]:
            if content["type"] == "text":
                goal_message.add_text(content["text"])
            elif content["type"] == "image_url":
                goal_message.add_image(content["image_url"])

        extra_info = []
        extra_info.append(
            """Use ControlOrMeta instead of Control and Meta for keyboard shortcuts, to be cross-platform compatible. E.g. use ControlOrMeta for mutliple selection in lists.\n"""
        )

        extra_info_msg = self.msg_builder.user().add_text("\n".join(extra_info))

        first_obs_msg = []
        if self.flags.use_first_obs:
            msg = "Here is the first observation."
            screenshot_key = "screenshot_tag" if self.flags.tag_screenshot else "screenshot"
            if self.flags.tag_screenshot:
                msg += " A blue dot on screenshots indicate the previous click action."

            first_obs_msg = self.msg_builder.user().add_text(msg).add_image_url(obs[screenshot_key])

        return [sys_msg, goal_message, extra_info_msg, first_obs_msg]


OPENAI_MODEL_CONFIG = OpenAIResponseModelArgs(
    model_name="gpt-4.1",
    max_total_tokens=200_000,
    max_input_tokens=200_000,
    max_new_tokens=2_000,
    temperature=0.1,
    vision_support=True,
)

OPENAI_CHATAPI_MODEL_CONFIG = OpenAIChatModelArgs(
    model_name="gpt-4o-2024-08-06",
    max_total_tokens=200_000,
    max_input_tokens=200_000,
    max_new_tokens=2_000,
    temperature=0.1,
    vision_support=True,
)

CLAUDE_MODEL_CONFIG = ClaudeResponseModelArgs(
    model_name="claude-3-7-sonnet-20250219",
    max_total_tokens=200_000,
    max_input_tokens=200_000,
    max_new_tokens=2_000,
    temperature=0.1,
    vision_support=True,
)


AGENT_CONFIG = ToolUseAgentArgs(
    model_args=CLAUDE_MODEL_CONFIG,
)

CHATAPI_AGENT_CONFIG = ToolUseAgentArgs(
    model_args=OpenAIChatModelArgs(
        model_name="gpt-4o-2024-11-20",
        max_total_tokens=200_000,
        max_input_tokens=200_000,
        max_new_tokens=2_000,
        temperature=0.7,
        vision_support=True,
    ),
)


OAI_CHAT_TOOl_AGENT = ToolUseAgentArgs(
    model_args=OpenAIChatModelArgs(model_name="gpt-4o-2024-08-06")
)


PROVIDER_FACTORY_MAP = {
    "openai": {"chatcompletion": OpenAIChatModelArgs, "response": OpenAIResponseModelArgs},
    "openrouter": OpenRouterModelArgs,
    "vllm": VLLMModelArgs,
    "anthropic": ClaudeResponseModelArgs,
}


def get_tool_use_agent(
    api_provider: str,
    model_args: "BaseModelArgs",
    tool_use_agent_args: dict = None,
    api_provider_spec=None,
) -> ToolUseAgentArgs:

    if api_provider == "openai":
        assert (
            api_provider_spec is not None
        ), "Endpoint specification is required for OpenAI provider. Choose between 'chatcompletion' and 'response'."

    model_args_factory = (
        PROVIDER_FACTORY_MAP[api_provider]
        if api_provider_spec is None
        else PROVIDER_FACTORY_MAP[api_provider][api_provider_spec]
    )

    # Create the agent with model arguments from the factory
    agent = ToolUseAgentArgs(
        model_args=model_args_factory(**model_args), **(tool_use_agent_args or {})
    )
    return agent


