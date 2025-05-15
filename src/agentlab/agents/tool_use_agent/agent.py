import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import bgym
import numpy as np
from PIL import Image, ImageDraw

from agentlab.agents import agent_utils
from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.llm_utils import image_to_png_base64_url
from agentlab.llm.response_api import (
    ClaudeResponseModelArgs,
    MessageBuilder,
    OpenAIChatModelArgs,
    OpenAIResponseModelArgs,
    OpenRouterModelArgs,
    ResponseLLMOutput,
)
from agentlab.llm.tracking import cost_tracker_decorator
from browsergym.core.observation import extract_screenshot

if TYPE_CHECKING:
    from openai.types.responses import Response


@dataclass
class ToolUseAgentArgs(AgentArgs):
    model_args: OpenAIResponseModelArgs = None
    use_first_obs: bool = True
    tag_screenshot: bool = True
    use_raw_page_output: bool = True

    def __post_init__(self):
        try:
            self.agent_name = f"ToolUse-{self.model_args.model_name}".replace("/", "_")
        except AttributeError:
            pass

    def make_agent(self) -> bgym.Agent:
        return ToolUseAgent(
            model_args=self.model_args,
            use_first_obs=self.use_first_obs,
            tag_screenshot=self.tag_screenshot,
        )

    def prepare(self):
        return self.model_args.prepare_server()

    def close(self):
        return self.model_args.close_server()


class ToolUseAgent(bgym.Agent):
    def __init__(
        self,
        model_args: OpenAIResponseModelArgs,
        use_first_obs: bool = True,
        tag_screenshot: bool = True,
    ):
        self.chat = model_args.make_model()
        self.model_args = model_args
        self.use_first_obs = use_first_obs
        self.tag_screenshot = tag_screenshot
        self.action_set = bgym.HighLevelActionSet(["coord"], multiaction=False)
        self.tools = self.action_set.to_tool_description(api=model_args.api)

        self.call_ids = []

        # self.tools.append(
        #     {
        #         "type": "function",
        #         "name": "chain_of_thought",
        #         "description": "A tool that allows the agent to think step by step. Every other action must ALWAYS be preceeded by a call to this tool.",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "thoughts": {
        #                     "type": "string",
        #                     "description": "The agent's reasoning process.",
        #                 },
        #             },
        #             "required": ["thoughts"],
        #         },
        #     }
        # )

        self.llm = model_args.make_model(extra_kwargs={"tools": self.tools})
        self.msg_builder = model_args.get_message_builder()
        self.messages: list[MessageBuilder] = []

    def obs_preprocessor(self, obs):
        page = obs.pop("page", None)
        if page is not None:
            obs["screenshot"] = extract_screenshot(page)
            if self.tag_screenshot:
                screenshot = Image.fromarray(obs["screenshot"])
                screenshot = agent_utils.tag_screenshot_with_action(screenshot, obs["last_action"])
                obs["screenshot_tag"] = np.array(screenshot)
        else:
            raise ValueError("No page found in the observation.")

        return obs

    @cost_tracker_decorator
    def get_action(self, obs: Any) -> float:
        if len(self.messages) == 0:
            self.initalize_messages(obs)
        else:
            if obs["last_action_error"] == "":  # Check No error in the last action
                screenshot_key = "screenshot_tag" if self.tag_screenshot else "screenshot"
                tool_message = self.msg_builder.tool().add_image(
                    image_to_png_base64_url(obs[screenshot_key])
                )
                tool_message.update_last_raw_response(self.last_response)
                tool_message.add_tool_id(self.previous_call_id)
                self.messages.append(tool_message)
            else:
                tool_message = self.msg_builder.tool().add_text(
                    f"Function call failed: {obs['last_action_error']}"
                )
                tool_message.add_tool_id(self.previous_call_id)
                tool_message.update_last_raw_response(self.last_response)
                self.messages.append(tool_message)

        response: ResponseLLMOutput = self.llm(messages=self.messages)

        action = response.action
        think = response.think
        self.last_response = response
        self.previous_call_id = response.last_computer_call_id
        self.messages.append(response.assistant_message)  # this is tool call

        return (
            action,
            bgym.AgentInfo(
                think=think,
                chat_messages=self.messages,
                stats={},
            ),
        )

    def initalize_messages(self, obs: Any) -> None:
        system_message = self.msg_builder.system().add_text(
            "You are an agent. Based on the observation, you will decide which action to take to accomplish your goal."
        )
        self.messages.append(system_message)

        goal_message = self.msg_builder.user()
        for content in obs["goal_object"]:
            if content["type"] == "text":
                goal_message.add_text(content["text"])
            elif content["type"] == "image_url":
                goal_message.add_image(content["image_url"])
        self.messages.append(goal_message)

        extra_info = []

        extra_info.append(
            """Use ControlOrMeta instead of Control and Meta for keyboard shortcuts, to be cross-platform compatible. E.g. use ControlOrMeta for mutliple selection in lists.\n"""
        )

        self.messages.append(self.msg_builder.user().add_text("\n".join(extra_info)))

        if self.use_first_obs:
            msg = "Here is the first observation."
            screenshot_key = "screenshot_tag" if self.tag_screenshot else "screenshot"
            if self.tag_screenshot:
                msg += " A red dot on screenshots indicate the previous click action."
            message = self.msg_builder.user().add_text(msg)
            message.add_image(image_to_png_base64_url(obs[screenshot_key]))
            self.messages.append(message)


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


def supports_tool_calling(model_name: str) -> bool:
    """
    Check if the model supports tool calling.

    Args:
        model_name (str): The name of the model.

    Returns:
        bool: True if the model supports tool calling, False otherwise.
    """
    import os

    import openai

    client = openai.Client(
        api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1"
    )
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Call the test tool"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "dummy_tool",
                        "description": "Just a test tool",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                }
            ],
            tool_choice="required",
        )
        response = response.to_dict()
        return "tool_calls" in response["choices"][0]["message"]
    except Exception as e:
        print(f"Model '{model_name}' error: {e}")
        return False


def get_openrouter_model(model_name: str, **open_router_args) -> OpenRouterModelArgs:
    default_model_args = {
        "max_total_tokens": 200_000,
        "max_input_tokens": 180_000,
        "max_new_tokens": 2_000,
        "temperature": 0.1,
        "vision_support": True,
    }
    merged_args = {**default_model_args, **open_router_args}

    return OpenRouterModelArgs(model_name=model_name, **merged_args)


def get_openrouter_tool_use_agent(
    model_name: str,
    model_args: dict = {},
    use_first_obs=True,
    tag_screenshot=True,
    use_raw_page_output=True,
) -> ToolUseAgentArgs:
    #To Do : Check if OpenRouter endpoint specific args are working
    if not supports_tool_calling(model_name):
        raise ValueError(f"Model {model_name} does not support tool calling.")

    model_args = get_openrouter_model(model_name, **model_args)

    return ToolUseAgentArgs(
        model_args=model_args,
        use_first_obs=use_first_obs,
        tag_screenshot=tag_screenshot,
        use_raw_page_output=use_raw_page_output,
    )


OPENROUTER_MODEL = get_openrouter_tool_use_agent("google/gemini-2.5-pro-preview")


AGENT_CONFIG = ToolUseAgentArgs(
    model_args=CLAUDE_MODEL_CONFIG,
)

MT_TOOL_USE_AGENT = ToolUseAgentArgs(
    model_args=OPENROUTER_MODEL,
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
    model_args=OpenAIChatModelArgs(model_name="gpt-4o-2024-08-06"),
    use_first_obs=False,
    tag_screenshot=False,
    use_raw_page_output=True,
)


## We have three providers that we want to support.
# Anthropic
# OpenAI
# vllm (uses OpenAI API)
