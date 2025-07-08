import fnmatch
import json
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import bgym
import numpy as np
import pandas as pd
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
    ClaudeResponseModelArgs,
    LLMOutput,
    MessageBuilder,
    OpenAIChatModelArgs,
    OpenAIResponseModelArgs,
)
from agentlab.llm.tracking import cost_tracker_decorator


@dataclass
class Block(ABC):

    def _init(self):
        """Initialize the block."""
        pass

    def make(self) -> "Block":
        """Returns a copy so the init can start adding some stuff to `self` without changing the
        original datatclass that should only contain a config.
        The aim is avoid having 2 calss definition for each block, e.g. Block and BlockArgs.

        Returns:
            Block: A copy of the current block instance with initialization applied.
        """
        block = self.__class__(**asdict(self))
        block._init()
        return block

    @abstractmethod
    def apply(self, llm, messages: list[MessageBuilder], **kwargs):
        pass


@dataclass
class MsgGroup:
    name: str = None
    messages: list[MessageBuilder] = field(default_factory=list)
    summary: MessageBuilder = None


class StructuredDiscussion:
    """
    A structured discussion that groups messages into named groups with a potential summary for each group.

    When the discussion is flattened, only the last `keep_last_n_obs` groups are kept in the final list,
    the other groups are replaced by their summaries if they have one.
    """

    def __init__(self, keep_last_n_obs=None):
        self.groups: list[MsgGroup] = []
        self.keep_last_n_obs: int | None = keep_last_n_obs

    def append(self, message: MessageBuilder):
        """Append a message to the last group."""
        self.groups[-1].messages.append(message)

    def new_group(self, name: str = None):
        """Start a new group of messages."""
        if name is None:
            name = f"group_{len(self.groups)}"
        self.groups.append(MsgGroup(name))

    def flatten(self) -> list[MessageBuilder]:
        """Flatten the groups into a single list of messages."""

        keep_last_n_obs = self.keep_last_n_obs or len(self.groups)
        messages = []
        for i, group in enumerate(self.groups):
            is_tail = i >= len(self.groups) - keep_last_n_obs

            if not is_tail and group.summary is not None:
                messages.append(group.summary)
            else:
                messages.extend(group.messages)
            # Mark all summarized messages for caching
            if i == len(self.groups) - keep_last_n_obs:
                messages[i].mark_all_previous_msg_for_caching()
        return messages

    def set_last_summary(self, summary: MessageBuilder):
        # append None to summaries until we reach the current group index
        self.groups[-1].summary = summary

    def get_last_summary(self) -> MessageBuilder | None:
        """Get the last summary message."""
        if len(self.groups) == 0:
            return None
        return self.groups[-1].summary

    def is_goal_set(self) -> bool:
        """Check if the goal is set in the first group."""
        return len(self.groups) > 0


SYS_MSG = """You are a web agent. Based on the observation, you will decide which action to take to accomplish your goal. 
You strive for excellence and need to be as meticulous as possible. Make sure to explore when not sure.
"""


@dataclass
class Goal(Block):
    """Block to add the goal to the messages."""

    goal_as_system_msg: bool = True

    def apply(self, llm, discussion: StructuredDiscussion, obs: dict) -> dict:
        system_message = llm.msg.system().add_text(SYS_MSG)
        discussion.append(system_message)

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
        discussion.append(goal_message)


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
    add_mouse_pointer: bool = False
    use_zoomed_webpage: bool = False

    def apply(
        self, llm, discussion: StructuredDiscussion, obs: dict, last_llm_output: LLMOutput
    ) -> dict:

        if last_llm_output.tool_calls is None:
            obs_msg = llm.msg.user()  # type: MessageBuilder
        else:
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
                # TODO this mouse pointer should be added at the browsergym level
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

        discussion.append(obs_msg)
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

    def apply(self, llm, discussion: StructuredDiscussion) -> dict:
        if not self.use_hints:
            return

        hints = []

        hints.append(
            """Use ControlOrMeta instead of Control and Meta for keyboard shortcuts, to be cross-platform compatible. E.g. use ControlOrMeta for mutliple selection in lists.\n"""
        )

        discussion.append(llm.msg.user().add_text("\n".join(hints)))


@dataclass
class Summarizer(Block):
    """Block to summarize the last action and the current state of the environment."""

    do_summary: bool = False
    high_details: bool = True

    def apply(self, llm, discussion: StructuredDiscussion) -> dict:
        if not self.do_summary:
            return

        msg = llm.msg.user().add_text("""Summarize\n""")

        discussion.append(msg)
        # TODO need to make sure we don't force tool use here
        summary_response = llm(messages=discussion.flatten(), tool_choice="none")

        summary_msg = llm.msg.assistant().add_text(summary_response.think)
        discussion.append(summary_msg)
        discussion.set_last_summary(summary_msg)
        return summary_msg

    def apply_init(self, llm, discussion: StructuredDiscussion) -> dict:
        """Initialize the summarizer block."""
        if not self.do_summary:
            return

        system_msg = llm.msg.system()
        if self.high_details:
            # Add a system message to the LLM to indicate that it should summarize
            system_msg.add_text(
                """# Summarizer instructions:\nWhen asked to summarize, do the following:
1) Summarize the effect of the last action, with attention to details.
2) Give a semantic description of the current state of the environment, with attention to details. If there was a repeating mistake, mention the cause of it.
3) Reason about the overall task at a high level.
4) What hint can be relevant for the next action? Only chose from the hints provided in the task description. Or select none.
5) Reason about the next action to take, based on the current state and the goal.
"""
            )
        else:
            system_msg.add_text(
                """When asked to summarize, give a semantic description of the current state of the environment."""
            )
        discussion.append(system_msg)


@dataclass
class TaskHint(Block):
    use_task_hint: bool = True
    hint_db_rel_path: str = "hint_db.csv"

    def _init(self):
        """Initialize the block."""
        hint_db_path = Path(__file__).parent / self.hint_db_rel_path
        self.hint_db = pd.read_csv(hint_db_path, header=0, index_col=None, dtype=str)

    def apply(self, llm, discussion: StructuredDiscussion, task_name: str) -> dict:
        if not self.use_task_hint:
            return

        task_hints = self.hint_db[
            self.hint_db["task_name"].apply(lambda x: fnmatch.fnmatch(x, task_name))
        ]

        hints = []
        for hint in task_hints["hint"]:
            hint = hint.strip()
            if hint:
                hints.append(f"- {hint}")

        if len(hints) > 0:
            hints_str = (
                "# Hints:\nHere are some hints for the task you are working on:\n"
                + "\n".join(hints)
            )
            msg = llm.msg.user().add_text(hints_str)

            discussion.append(msg)


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
    task_hint: TaskHint = None
    keep_last_n_obs: int = 1
    multiaction: bool = False
    action_subsets: tuple[str] = None


@dataclass
class ToolUseAgentArgs(AgentArgs):
    model_args: OpenAIResponseModelArgs = None
    config: PromptConfig = None
    use_raw_page_output: bool = False  # This attribute is used in loop.py to setup the env.

    def __post_init__(self):
        try:
            self.agent_name = f"ToolUse-{self.model_args.model_name}".replace("/", "_")
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
        self.action_set = bgym.HighLevelActionSet(
            self.config.action_subsets, multiaction=self.config.multiaction
        )
        self.tools = self.action_set.to_tool_description(api=model_args.api)

        self.call_ids = []

        self.llm = model_args.make_model(extra_kwargs={"tools": self.tools})
        self.msg_builder = model_args.get_message_builder()
        self.llm.msg = self.msg_builder

        self.task_hint = self.config.task_hint.make()
        self.obs_block = self.config.obs.make()

        self.discussion = StructuredDiscussion(self.config.keep_last_n_obs)
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
            if self.config.obs.use_zoomed_webpage:
                pass

        return obs

    def set_task_name(self, task_name: str):
        """Cheater function that is supposed to be called by loop.py before callling get_action"""
        self.task_name = task_name

    @cost_tracker_decorator
    def get_action(self, obs: Any) -> float:
        self.llm.reset_stats()
        if not self.discussion.is_goal_set():
            self.discussion.new_group("goal")
            self.config.goal.apply(self.llm, self.discussion, obs)
            self.config.summarizer.apply_init(self.llm, self.discussion)
            self.config.general_hints.apply(self.llm, self.discussion)
            self.task_hint.apply(self.llm, self.discussion, self.task_name)

            self.discussion.new_group()

        self.obs_block.apply(self.llm, self.discussion, obs, last_llm_output=self.last_response)

        self.config.summarizer.apply(self.llm, self.discussion)

        messages = self.discussion.flatten()
        response: LLMOutput = self.llm(
            messages=messages,
            tool_choice="any",
            cache_tool_definition=True,
            cache_complete_prompt=False,
            use_cache_breakpoints=True,
        )

        action = response.action
        think = response.think
        last_summary = self.discussion.get_last_summary()
        if last_summary is not None:
            think = last_summary.content[0]["text"] + "\n" + think

        self.discussion.new_group()
        self.discussion.append(response.tool_calls)

        self.last_response = response
        self._responses.append(response)  # may be useful for debugging
        # self.messages.append(response.assistant_message)  # this is tool call

        tools_str = json.dumps(self.tools, indent=2)
        tools_msg = MessageBuilder("tool_description").add_text(tools_str)

        # Adding these extra messages to visualize in gradio
        messages.insert(0, tools_msg)  # insert at the beginning of the messages
        messages.append(response.tool_calls)

        agent_info = bgym.AgentInfo(
            think=think,
            chat_messages=messages,
            stats=self.llm.stats.stats_dict,
        )
        return action, agent_info


OPENAI_MODEL_CONFIG = OpenAIResponseModelArgs(
    model_name="gpt-4.1",
    max_total_tokens=200_000,
    max_input_tokens=200_000,
    max_new_tokens=2_000,
    temperature=0.1,
    vision_support=True,
)

GPT_4_1_MINI = OpenAIResponseModelArgs(
    model_name="gpt-4.1-mini",
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
    summarizer=Summarizer(do_summary=True),
    general_hints=GeneralHints(use_hints=False),
    task_hint=TaskHint(use_task_hint=True),
    keep_last_n_obs=None,  # keep only the last observation in the discussion
    multiaction=False,  # whether to use multi-action or not
    # action_subsets=("bid",),
    action_subsets=("coord"),
    # action_subsets=("coord", "bid"),
)

AGENT_CONFIG = ToolUseAgentArgs(
    model_args=CLAUDE_MODEL_CONFIG,
    config=DEFAULT_PROMPT_CONFIG,
)
