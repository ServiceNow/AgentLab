import json
import logging
import os
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import bgym
from bgym import Benchmark as BgymBenchmark
from browsergym.core.observation import extract_screenshot
from browsergym.utils.obs import (
    flatten_axtree_to_str,
    flatten_dom_to_str,
    overlay_som,
    prune_html,
)

from agentlab.agents.agent_args import AgentArgs
from agentlab.benchmarks.abstract_env import AbstractBenchmark as AgentLabBenchmark
from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.litellm_api import LiteLLMModelArgs
from agentlab.llm.llm_utils import image_to_png_base64_url
from agentlab.llm.response_api import (
    APIPayload,
    LLMOutput,
    MessageBuilder,
)
from agentlab.llm.tracking import cost_tracker_decorator
from agentlab.utils.hinting import HintsSource


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ADDITIONAL_ACTION_INSTRUCTIONS = """
**Important Rules:**
- Coordinates (x, y) must be NUMBERS, not strings
- Do NOT use named parameters for coordinates unless necessary for clarity
- Button parameter is optional, defaults to 'left'
- String values must be in quotes
- Call send_msg_to_user only with a single number in the answer when sending the final answer for evaluation.

**Correct Examples:**
- mouse_click(347, 192)
- mouse_click(56, 712.56, 'right')
- keyboard_type('hello@example.com')
- keyboard_type('System Diagnostics')
- keyboard_press('ControlOrMeta+v')
- keyboard_press('Escape')
- mouse_drag_and_drop(100, 200, 300, 400)

**WRONG Examples (DO NOT DO THIS):**
- mouse_click(x='347, 192', y=192)  ❌ x is a string with both coords
- mouse_click('347', '192')  ❌ coordinates as strings
- "mouse_click(100, 200)"  ❌ wrapped in quotes
- keyboard_press(Escape)  ❌ string argument missing quotes
- keyboard_type(System Diagnostics)  ❌ text argument missing quotes
"""

simple_bgym_action_tool = {
    "name": "perform_action",
    "type": "function",
    "description": f"""Return a string representation of a Python function call for browsergym actions.
        You must return ONLY the function call string, exactly as it would appear in Python code.""",
    "parameters": {
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": "The agent's internal chain of thought for performing the action.",
            },
            "action": {
                "type": "string",
                "description": "The Python function call string (e.g., 'mouse_click(100, 200)' or 'keyboard_type(\"hello\")')",
            },
        },
        "required": ["thought", "action"],
    },
}


def action_from_generalized_bgym_action_tool(
    response: LLMOutput, tool_name: str = "perform_action"
) -> tuple[str | None, str | None]:
    """Extract the action string from the tool call in the LLM response."""
    action, think = None, None
    if response.tool_calls is not None:
        for tc in response.tool_calls.tool_calls:
            if tc.name == tool_name:
                action = tc.arguments.get("action")
                think = tc.arguments.get("thought")
                break
    return action, think


@dataclass
class Block(ABC):
    def _init(self):
        """Initialize the block."""
        pass

    def make(self) -> "Block":
        """Returns a copy so the init can start adding some stuff to `self` without changing the
        original dataclass that should only contain a config.
        The aim is avoid having 2 class definition for each block, e.g. Block and BlockArgs.

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

    @property
    def tool_summary(self) -> None:
        return [msg for msg in self.messages if msg.role == "tool"]

    @property
    def messages_without_images(self) -> list[MessageBuilder]:
        _messages = deepcopy(self.messages)
        for msg in _messages:
            for content in msg.content:
                if "image" in content:
                    content.pop("image")
                    content["text"] = "[Screenshot Placeholder]"

        return _messages


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

            if not is_tail:
                if group.summary is not None:
                    messages.append(group.summary)
                else:
                    messages.extend(group.messages_without_images)

            else:
                messages.extend(group.messages)
            # Mark all summarized messages for caching
            if i == len(self.groups) - keep_last_n_obs:
                for msg in messages:  # unset previous cache breakpoints
                    msg._cache_breakpoint = False
                # set new cache breakpoint
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

    def apply(
        self, llm, discussion: StructuredDiscussion, obs: dict, sys_msg: str = SYS_MSG
    ) -> dict:
        system_message = llm.msg.system().add_text(sys_msg)
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
A line starting with [bid] is a node in the AXTree. It is a unique alpha-numeric identifier to be used when calling tools, e.g, click(bid="a253"). Make sure to include letters and numbers in the bid.
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
    overlay_mouse_action: bool = False
    use_zoomed_webpage: bool = False
    skip_preprocessing: bool = False

    def _init(self):
        self._last_observation = None

    def apply(
        self, llm, discussion: StructuredDiscussion, obs: dict, last_llm_output: LLMOutput
    ) -> dict:
        obs_msg = llm.msg.user()
        tool_calls = last_llm_output.tool_calls
        # add the tool call response first in the observation
        # to maintain continuity with last response.
        if tool_calls:
            for call in tool_calls:
                call.response_text("See Observation")
            tool_response = llm.msg.add_responded_tool_calls(tool_calls)
            discussion.append(tool_response)

        if self.use_last_error:
            if obs["last_action_error"] != "":
                obs_msg.add_text(f"Last action error:\n{obs['last_action_error']}")

        if self.use_screenshot:
            if self.use_som:
                screenshot = obs["screenshot_som"]
            else:
                screenshot = obs["screenshot"]

            if self.overlay_mouse_action and self._last_observation is not None:
                self.overlay_last_screenshot_with_action(
                    discussion, obs["last_action"], self._last_observation
                )

            obs_msg.add_image(image_to_png_base64_url(screenshot))
        if self.use_axtree:
            obs_msg.add_text(f"AXTree:\n{AXTREE_NOTE}\n{obs['axtree_txt']}")
        if self.use_dom:
            obs_msg.add_text(f"DOM:\n{obs['pruned_html']}")
        if self.use_tabs:
            obs_msg.add_text(_format_tabs(obs))

        discussion.append(obs_msg)
        self._last_observation = deepcopy(obs)
        return obs_msg

    @staticmethod
    def overlay_last_screenshot_with_action(discussion: StructuredDiscussion, action, obs):
        """Update the last image with new_image_base64 overlayed with the action."""
        import base64
        from agentlab.analyze import overlay_utils
        from PIL import Image
        from io import BytesIO

        for msg_groups in reversed(discussion.groups):
            for msg in reversed(msg_groups.messages):
                for content in reversed(msg.content):
                    if "image" in content:
                        data_url = content["image"]
                        header, encoded = data_url.split(",", 1)
                        new_obs_properties = deepcopy(obs["extra_element_properties"])
                        sc = Image.open(BytesIO(base64.b64decode(encoded)))
                        overlay_utils.annotate_action(sc, action, properties=new_obs_properties)
                        new_base64_image = image_to_png_base64_url(sc)
                        content["image"] = new_base64_image
                        return


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
        # simulated a hint.
        # hints.append(
        #     """Remember to submit the form once all the fields are filled out.\n"""
        # )

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

        summary_response = llm(APIPayload(messages=discussion.flatten()))

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
    hint_retrieval_mode: Literal["direct", "llm", "emb"] = "direct"
    top_n: int = 4  # Number of top hints to return when using embedding retrieval
    embedder_model: str = "Qwen/Qwen3-Embedding-0.6B"  # Model for embedding hints
    embedder_server: str = "http://localhost:5000"
    skip_hints_for_current_task: bool = False
    skip_hints_for_current_goal: bool = False

    def _init(self):
        """Initialize the block."""
        if self.use_task_hint:
            self.hints_source = HintsSource(
                hint_db_path=self.hint_db_rel_path,
                hint_retrieval_mode=self.hint_retrieval_mode,
                top_n=self.top_n,
                embedder_model=self.embedder_model,
                embedder_server=self.embedder_server,
                skip_hints_for_current_task=self.skip_hints_for_current_task,
                skip_hints_for_current_goal=self.skip_hints_for_current_goal,
            )

    def apply(self, llm, discussion: StructuredDiscussion, obs: dict, task_name: str) -> dict:
        if not self.use_task_hint:
            return {}

        try:
            goal_text = obs["goal_object"][0]["text"]
        except (KeyError, IndexError):
            Warning("Goal text not found in observation")
            goal_text = ""
        task_hints = self.hints_source.choose_hints(llm, task_name, goal_text)

        hints = []
        for hint in task_hints:
            hint = hint.strip()
            if hint:
                hints.append(f"- {hint}")

        if len(hints) > 0:
            hints_str = (
                "\n# Hints:\nHere are some hints for the task you are working on:\n"
                + "\n".join(hints)
            )
            msg = llm.msg.user().add_text(hints_str)

            discussion.append(msg)


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
    use_noop_as_default_action: bool = False
    use_generalized_bgym_action_tool: bool = True


@dataclass
class ToolUseAgentArgs(AgentArgs):
    model_args: BaseModelArgs = None
    config: PromptConfig = None
    use_raw_page_output: bool = False  # This attribute is used in loop.py to setup the env.
    action_set: bgym.AbstractActionSet | None = None

    def __post_init__(self):
        try:
            self.agent_name = f"CUAv2-{self.model_args.model_name}".replace("/", "_")
            if self.config.task_hint.use_task_hint:
                if self.config.task_hint.hint_retrieval_mode == "direct":
                    self.agent_name += f"-direct-hint"
                if self.config.task_hint.hint_retrieval_mode == "emb":
                    self.agent_name += f"-emb-hint"
                if self.config.task_hint.hint_retrieval_mode == "llm":
                    self.agent_name += f"-llm-hint"

        except AttributeError:
            pass

    def make_agent(self) -> bgym.Agent:
        if self.config is None:
            self.config = DEFAULT_PROMPT_CONFIG
        return ToolUseAgent(
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


class ToolUseAgent(bgym.Agent):
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

    def obs_preprocessor(self, obs):
        obs = copy(obs)
        if self.config.obs.skip_preprocessing:
            return obs
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

            if self.config.multiaction:
                sys_msg = SYS_MSG + "\nYou can take multiple actions in a single step, if needed."
            else:
                sys_msg = SYS_MSG + "\nYou can only take one action at a time."

            sys_msg += (
                "\nAvailable browsergym actions that can be returned with get_action:\n"
                + self.action_set.describe()
            )
            sys_msg += ADDITIONAL_ACTION_INSTRUCTIONS
            self.config.goal.apply(self.llm, self.discussion, obs, sys_msg)

            self.config.summarizer.apply_init(self.llm, self.discussion)
            self.config.general_hints.apply(self.llm, self.discussion)
            self.task_hint.apply(self.llm, self.discussion, obs, self.task_name)

            self.discussion.new_group()

        self.obs_block.apply(self.llm, self.discussion, obs, last_llm_output=self.last_response)

        self.config.summarizer.apply(self.llm, self.discussion)

        messages = self.discussion.flatten()
        response: LLMOutput = self.llm(
            APIPayload(
                messages=messages,
                tools=self.tools,
                tool_choice="any",
                cache_tool_definition=True,
                cache_complete_prompt=False,
                use_cache_breakpoints=True,
            )
        )

        if self.config.use_generalized_bgym_action_tool:
            action, think = action_from_generalized_bgym_action_tool(response)
        else:
            action = response.action
            think = response.think

        if action is None and self.config.use_noop_as_default_action:
            action = "noop()"

        last_summary = self.discussion.get_last_summary()
        if last_summary is not None:
            think = last_summary.content[0]["text"] + "\n" + think
        else:
            # Add the think to the history when use_summarizer is False
            if think is not None:
                self.discussion.append(self.llm.msg.assistant().add_text(think))

        self.discussion.new_group()

        self.last_response = response
        self._responses.append(response)  # may be useful for debugging

        tools_str = json.dumps(self.tools, indent=2)
        tools_msg = MessageBuilder("tool_description").add_text(tools_str)

        # Adding these extra messages to visualize in gradio
        messages.insert(0, tools_msg)  # insert at the beginning of the message
        # This avoids the assertion error with self.llm.user().add_responded_tool_calls(tool_calls)
        msg = self.llm.msg("tool")
        msg.responded_tool_calls = response.tool_calls
        messages.append(msg)

        agent_info = bgym.AgentInfo(
            think=think,
            chat_messages=messages,
            stats=self.llm.stats.stats_dict,
        )
        return action, agent_info


CUA_PROMPT_CONFIG = PromptConfig(
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
    keep_last_n_obs=5,  # no more than 20 screenshots for claude
    multiaction=True,
    use_noop_as_default_action=False,
    use_generalized_bgym_action_tool=True,
)


def get_cua_like_agent_config_with_hint(
    model_name: str,
    hint_db_path: str,
    hint_retrieval_mode: Literal["direct", "llm", "emb"] = "direct",
) -> ToolUseAgentArgs:
    config = deepcopy(CUA_PROMPT_CONFIG)
    config.task_hint.use_task_hint = True
    config.task_hint.hint_db_rel_path = hint_db_path
    config.task_hint.hint_retrieval_mode = hint_retrieval_mode
    return ToolUseAgentArgs(
        model_args=LiteLLMModelArgs(
            model_name=model_name,
            max_new_tokens=2000,
            temperature=None,  # NONE for claude-4-5 to enable reasoning effort.
        ),
        config=config,
    )


def get_cua_like_agent_config_with_hint_skip_for_current_goal(
    model_name: str,
    hint_db_path: str,
    hint_retrieval_mode: Literal["llm", "emb"] = "llm",
) -> ToolUseAgentArgs:
    config = deepcopy(CUA_PROMPT_CONFIG)
    config.task_hint.use_task_hint = True
    config.task_hint.skip_hints_for_current_goal = True
    config.task_hint.hint_db_rel_path = hint_db_path
    config.task_hint.hint_retrieval_mode = hint_retrieval_mode
    return ToolUseAgentArgs(
        model_args=LiteLLMModelArgs(
            model_name=model_name,
            max_new_tokens=2000,
            temperature=None,  # NONE for claude-4-5 to enable reasoning effort.
        ),
        config=config,
    )


def get_cua_like_agent_config(model_name: str) -> ToolUseAgentArgs:

    return ToolUseAgentArgs(
        model_args=LiteLLMModelArgs(
            model_name=model_name,
            max_new_tokens=2000,
            temperature=None,
        ),
        config=CUA_PROMPT_CONFIG,
    )


CUA_LIKE_CLAUDE_4_SONNET = get_cua_like_agent_config("anthropic/claude-sonnet-4-20250514")


if __name__ == "__main__":

    from agentlab.agents.tool_use_agent.cua_like_agent import CUA_LIKE_CLAUDE_4_SONNET
    from agentlab.experiments.study import Study
    import bgym
    import logging

    logging.getLogger().setLevel(logging.INFO)
    os.environ["LITELLM_LOG"] = "WARNING"

    benchmark = "workarena_l1"
    benchmark = bgym.DEFAULT_BENCHMARKS[benchmark](n_repeats=2)
    benchmark = benchmark.subset_from_glob("task_name", "*create*")
    for env_args in benchmark.env_args_list:
        env_args.max_steps = 20  # increase the number of steps for coord agent testing

    agent_args = [CUA_LIKE_CLAUDE_4_SONNET]
    study = Study(agent_args, benchmark, logging_level_stdout=logging.WARNING)
    study.run(
        n_jobs=5,
        parallel_backend="ray",
        strict_reproducibility=False,
        n_relaunch=1,
    )
