from copy import copy
from dataclasses import dataclass
from typing import Any

import bgym
import numpy as np
from browsergym.core.observation import extract_screenshot
from browsergym.utils.obs import (
    flatten_axtree_to_str,
    flatten_dom_to_str,
    overlay_som,
    prune_html,
)
from PIL import Image

from agentlab.agents import agent_utils
from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.llm_utils import image_to_png_base64_url
from agentlab.llm.response_api import (
    BaseModelArgs,
    ClaudeResponseModelArgs,
    LLMOutput,
    MessageBuilder,
    OpenAIChatModelArgs,
    OpenAIResponseModelArgs,
    OpenRouterModelArgs,
    VLLMModelArgs,
)
from agentlab.llm.tracking import cost_tracker_decorator


class Block:

    def make(self):
        return self


@dataclass
class Goal(Block):
    """Block to add the goal to the messages."""

    goal_as_system_msg: bool = True

    def apply(self, llm, messages: list[MessageBuilder], obs: dict) -> dict:
        system_message = llm.msg.system().add_text(
            "You are an agent. Based on the observation, you will decide which action to take to accomplish your goal."
        )
        messages.append(system_message)

        if self.goal_as_system_msg:
            goal_message = llm.msg.system()
        else:
            goal_message = llm.msg.user()

        goal_message.add_text("# Goal:\n")
        for content in obs["goal_object"]:
            if content["type"] == "text":
                goal_message.add_text(content["text"])
            elif content["type"] == "image_url":
                goal_message.add_image(content["image_url"])
        messages.append(goal_message)


AXTREE_NOTE = """
AXTree extracts most of the interactive elements of the DOM in a tree structure. It may also contain information that is not visible in the screenshot.
A line starting with [bid] is a node in the AXTree. It is a unique alpha-numeric identifier to be used when calling tools.
"""


@dataclass
class Obs(Block):
    """Block to add the observation to the messages."""

    use_last_error: bool = True
    use_screenshot: bool = True
    use_axtree: bool = False
    use_dom: bool = False
    use_som: bool = False
    use_tabs: bool = False
    add_mouse_pointer: bool = True

    def apply(
        self, llm, messages: list[MessageBuilder], obs: dict, last_llm_output: LLMOutput
    ) -> dict:

        if last_llm_output.tool_calls is None:
            obs_msg = llm.msg.user()  # type: MessageBuilder
        else:
            messages.append(last_llm_output.tool_calls)  # TODO move else where
            obs_msg = llm.msg.tool(last_llm_output.raw_response)  # type: MessageBuilder

        if self.use_last_error:
            if obs["last_action_error"] != "":
                obs_msg.add_text(f"Last action error:\n{obs['last_action_error']}")

        if self.use_screenshot:

            if self.use_som:
                screenshot = obs["screenshot_som"]
            else:
                screenshot = obs["screenshot"]

            if self.add_mouse_pointer:
                screenshot = np.array(
                    agent_utils.add_mouse_pointer_from_action(
                        Image.fromarray(obs["screenshot"]), obs["last_action"]
                    )
                )

            obs_msg.add_image(image_to_png_base64_url(screenshot))
        if self.use_axtree:
            obs_msg.add_text(f"AXTree:\n{AXTREE_NOTE}\n{obs['axtree_txt']}")
        if self.use_dom:
            obs_msg.add_text(f"DOM:\n{obs['pruned_html']}")
        if self.use_tabs:
            obs_msg.add_text(_format_tabs(obs))

        messages.append(obs_msg)
        return obs_msg


def _format_tabs(obs):
    """Format the open tabs in a llm-readable way."""
    prompt_pieces = ["Currently open tabs:"]
    for page_index, (page_url, page_title) in enumerate(
        zip(obs["open_pages_urls"], obs["open_pages_titles"])
    ):
        active_or_not = " (active tab)" if page_index == obs["active_page_index"] else ""
        prompt_piece = f"""\
Tab {page_index}{active_or_not}:
    Title: {page_title}
    URL: {page_url}
"""
        prompt_pieces.append(prompt_piece)
    return "\n".join(prompt_pieces)


@dataclass
class GeneralHints(Block):

    use_hints: bool = True

    def apply(self, llm, messages: list[MessageBuilder]) -> dict:
        if not self.use_hints:
            return

        hints = []

        hints.append(
            """Use ControlOrMeta instead of Control and Meta for keyboard shortcuts, to be cross-platform compatible. E.g. use ControlOrMeta for mutliple selection in lists.\n"""
        )

        messages.append(llm.msg.user().add_text("\n".join(hints)))


@dataclass
class Summarizer(Block):
    """Block to summarize the last action and the current state of the environment."""

    def apply(self, llm, messages: list[MessageBuilder]) -> dict:
        msg = llm.msg.user().add_text(
            "Summarize the effect of the last action and the current state of the environment."
        )

        messages.append(msg)
        # TODO need to make sure we don't force tool use here
        summary_response = llm(messages=messages)

        summary_msg = llm.msg.assistant().add_text(summary_response.think)
        messages.append(summary_msg)


class ToolCall(Block):

    def __init__(self, tool_server):
        self.tool_server = tool_server

    def apply(self, llm, messages: list[MessageBuilder], obs: dict) -> dict:
        # build the message by adding components to obs
        response: LLMOutput = llm(messages=self.messages)

        messages.append(response.assistant_message)  # this is tool call

        tool_answer = self.tool_server.call_tool(response)
        tool_msg = llm.msg.tool()  # type: MessageBuilder
        tool_msg.add_tool_id(response.last_computer_call_id)
        tool_msg.update_last_raw_response(response)
        tool_msg.add_text(str(tool_answer))
        messages.append(tool_msg)


@dataclass
class PromptConfig:
    tag_screenshot: bool = True  # Whether to tag the screenshot with the last action.
    goal: Goal = None
    obs: Obs = None
    summarizer: Summarizer = None
    general_hints: GeneralHints = None


@dataclass
class ToolUseAgentArgs(AgentArgs):
    model_args: OpenAIResponseModelArgs = None
    config: PromptConfig = None
    use_raw_page_output: bool = False  # This attribute is used in loop.py to setup the env.

    def __post_init__(self):
        try:
            self.agent_name = f"MultiToolUse-{self.model_args.model_name}".replace("/", "_")
        except AttributeError:
            pass

    def make_agent(self) -> bgym.Agent:
        if self.config is None:
            self.config = DEFAULT_PROMPT_CONFIG
        return ToolUseAgent(
            model_args=self.model_args,
            config=self.config,
        )

    def prepare(self):
        return self.model_args.prepare_server()

    def close(self):
        return self.model_args.close_server()


class ToolUseAgent(bgym.Agent):
    def __init__(
        self,
        model_args: OpenAIResponseModelArgs,
        config: PromptConfig = None,
    ):
        self.model_args = model_args
        self.config = config
        self.action_set = bgym.HighLevelActionSet(["coord"], multiaction=False)
        self.tools = self.action_set.to_tool_description(api=model_args.api)

        self.call_ids = []

        self.llm = model_args.make_model(extra_kwargs={"tools": self.tools})
        self.msg_builder = model_args.get_message_builder()
        self.llm.msg = self.msg_builder

        # # blocks
        # self.goal_block = self.config.goal
        # self.obs_block = self.config.obs
        # self.summarizer_block = self.config.summarizer
        # self.general_hints_block = self.config.general_hints

        self.messages: list[MessageBuilder] = []
        self.last_response: LLMOutput = LLMOutput()
        self._responses: list[LLMOutput] = []

    def obs_preprocessor(self, obs):
        obs = copy(obs)

        page = obs.pop("page", None)
        if page is not None:
            obs["screenshot"] = extract_screenshot(page)
        else:
            if self.config.obs.use_dom:
                obs["dom_txt"] = flatten_dom_to_str(
                    obs["dom_object"],
                    extra_properties=obs["extra_element_properties"],
                )
                obs["pruned_html"] = prune_html(obs["dom_txt"])

            if self.config.obs.use_axtree:
                obs["axtree_txt"] = flatten_axtree_to_str(
                    obs["axtree_object"],
                    extra_properties=obs["extra_element_properties"],
                )

            if self.config.obs.use_som:
                obs["screenshot_som"] = overlay_som(
                    obs["screenshot"], extra_properties=obs["extra_element_properties"]
                )

        # if self.config.tag_screenshot:
        #     screenshot = Image.fromarray(obs["screenshot"])
        #     screenshot = agent_utils.tag_screenshot_with_action(screenshot, obs["last_action"])
        #     obs["screenshot"] = np.array(screenshot)

        return obs

    @cost_tracker_decorator
    def get_action(self, obs: Any) -> float:
        if len(self.messages) == 0:
            self.config.goal.apply(self.llm, self.messages, obs)
            self.config.general_hints.apply(self.llm, self.messages)

        self.config.obs.apply(self.llm, self.messages, obs, last_llm_output=self.last_response)
        self.config.summarizer.apply(self.llm, self.messages)

        response: LLMOutput = self.llm(messages=self.messages)

        action = response.action
        think = response.think
        self.last_response = response
        self._responses.append(response)  # may be useful for debugging
        # self.messages.append(response.assistant_message)  # this is tool call

        agent_info = bgym.AgentInfo(think=think, chat_messages=self.messages, stats={})
        return action, agent_info


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


DEFAULT_PROMPT_CONFIG = PromptConfig(
    tag_screenshot=True,
    goal=Goal(goal_as_system_msg=True),
    obs=Obs(
        use_last_error=True,
        use_screenshot=True,
        use_axtree=True,
        use_dom=False,
        use_som=False,
        use_tabs=False,
    ),
    summarizer=Summarizer(),
    general_hints=GeneralHints(use_hints=False),
)

AGENT_CONFIG = ToolUseAgentArgs(
    model_args=CLAUDE_MODEL_CONFIG,
    config=DEFAULT_PROMPT_CONFIG,
)

# MT_TOOL_USE_AGENT = ToolUseAgentArgs(
#     model_args=OPENROUTER_MODEL,
# )
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
