import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import bgym
import numpy as np
from browsergym.core.observation import extract_screenshot
from PIL import Image, ImageDraw

from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.llm_utils import image_to_png_base64_url
from agentlab.llm.response_api import (
    ClaudeResponseModelArgs,
    MessageBuilder,
    OpenAIResponseModelArgs,
    ResponseLLMOutput,
)
from agentlab.llm.tracking import cost_tracker_decorator

if TYPE_CHECKING:
    from openai.types.responses import Response


def tag_screenshot_with_action(screenshot: Image, action: str) -> Image:
    """
    If action is a coordinate action, try to render it on the screenshot.

    e.g. mouse_click(120, 130) -> draw a dot at (120, 130) on the screenshot

    Args:
        screenshot: The screenshot to tag.
        action: The action to tag the screenshot with.

    Returns:
        The tagged screenshot.

    Raises:
        ValueError: If the action parsing fails.
    """
    if action.startswith("mouse_click"):
        try:
            coords = action[action.index("(") + 1 : action.index(")")].split(",")
            coords = [c.strip() for c in coords]
            if len(coords) != 2:
                raise ValueError(f"Invalid coordinate format: {coords}")
            if coords[0].startswith("x="):
                coords[0] = coords[0][2:]
            if coords[1].startswith("y="):
                coords[1] = coords[1][2:]
            x, y = float(coords[0].strip()), float(coords[1].strip())
            draw = ImageDraw.Draw(screenshot)
            radius = 5
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius), fill="red", outline="red"
            )
        except (ValueError, IndexError) as e:
            logging.warning(f"Failed to parse action '{action}': {e}")
    return screenshot


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

        # count tools tokens
        from agentlab.llm.llm_utils import count_tokens

        tool_str = json.dumps(self.tools, indent=2)
        print(f"Tool description: {tool_str}")
        tool_tokens = count_tokens(tool_str, model_args.model_name)
        print(f"Tool tokens: {tool_tokens}")

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

        self.messages: list[MessageBuilder] = []

    def obs_preprocessor(self, obs):
        page = obs.pop("page", None)
        if page is not None:
            obs["screenshot"] = extract_screenshot(page)
            if self.tag_screenshot:
                screenshot = Image.fromarray(obs["screenshot"])
                screenshot = tag_screenshot_with_action(screenshot, obs["last_action"])
                obs["screenshot_tag"] = np.array(screenshot)
        else:
            raise ValueError("No page found in the observation.")

        return obs

    @cost_tracker_decorator
    def get_action(self, obs: Any) -> float:
        if len(self.messages) == 0:
            system_message = MessageBuilder.system().add_text(
                "You are an agent. Based on the observation, you will decide which action to take to accomplish your goal."
            )
            self.messages.append(system_message)

            goal_message = MessageBuilder.user()
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

            self.messages.append(MessageBuilder.user().add_text("\n".join(extra_info)))

            if self.use_first_obs:
                msg = "Here is the first observation."
                screenshot_key = "screenshot_tag" if self.tag_screenshot else "screenshot"
                if self.tag_screenshot:
                    msg += " A red dot on screenshots indicate the previous click action."
                message = MessageBuilder.user().add_text(msg)
                message.add_image(image_to_png_base64_url(obs[screenshot_key]))
                self.messages.append(message)
        else:
            if obs["last_action_error"] == "":
                screenshot_key = "screenshot_tag" if self.tag_screenshot else "screenshot"
                tool_message = MessageBuilder.tool().add_image(
                    image_to_png_base64_url(obs[screenshot_key])
                )
                tool_message.add_tool_id(self.previous_call_id)
                self.messages.append(tool_message)
            else:
                tool_message = MessageBuilder.tool().add_text(
                    f"Function call failed: {obs['last_action_error']}"
                )
                tool_message.add_tool_id(self.previous_call_id)
                self.messages.append(tool_message)

        response: ResponseLLMOutput = self.llm(messages=self.messages)

        action = response.action
        think = response.think
        self.previous_call_id = response.last_computer_call_id
        self.messages.append(response.assistant_message)

        return (
            action,
            bgym.AgentInfo(
                think=think,
                chat_messages=self.messages,
                stats={},
            ),
        )


OPENAI_MODEL_CONFIG = OpenAIResponseModelArgs(
    model_name="gpt-4.1",
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
