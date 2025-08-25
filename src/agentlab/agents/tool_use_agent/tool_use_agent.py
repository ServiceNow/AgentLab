import fnmatch
import json
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import bgym
import numpy as np
import pandas as pd
import requests
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
from agentlab.benchmarks.osworld import OSWorldActionSet
from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.llm_utils import image_to_png_base64_url
from agentlab.llm.response_api import (
    APIPayload,
    ClaudeResponseModelArgs,
    LLMOutput,
    MessageBuilder,
    OpenAIChatModelArgs,
    OpenAIResponseModelArgs,
    OpenRouterModelArgs,
    ToolCalls,
)
from agentlab.llm.tracking import cost_tracker_decorator

logger = logging.getLogger(__name__)


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
    # add_mouse_pointer: bool = False
    use_zoomed_webpage: bool = False
    skip_preprocessing: bool = False

    def apply(
        self, llm, discussion: StructuredDiscussion, obs: dict, last_llm_output: LLMOutput
    ) -> dict:
        obs_msg = llm.msg.user()
        tool_calls = last_llm_output.tool_calls
        if self.use_last_error:
            if obs["last_action_error"] != "":
                obs_msg.add_text(f"Last action error:\n{obs['last_action_error']}")

        if self.use_screenshot:
            if self.use_som:
                screenshot = obs["screenshot_som"]
            else:
                screenshot = obs["screenshot"]

            # if self.add_mouse_pointer:
            #     screenshot = np.array(
            #         agent_utils.add_mouse_pointer_from_action(
            #             Image.fromarray(obs["screenshot"]), obs["last_action"]
            #         )
            #     )

            obs_msg.add_image(image_to_png_base64_url(screenshot))
        if self.use_axtree:
            obs_msg.add_text(f"AXTree:\n{AXTREE_NOTE}\n{obs['axtree_txt']}")
        if self.use_dom:
            obs_msg.add_text(f"DOM:\n{obs['pruned_html']}")
        if self.use_tabs:
            obs_msg.add_text(_format_tabs(obs))

        discussion.append(obs_msg)

        if tool_calls:
            for call in tool_calls:
                call.response_text("See Observation")
            tool_response = llm.msg.add_responded_tool_calls(tool_calls)
            discussion.append(tool_response)

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
    llm_prompt: str = """We're choosing hints to help solve the following task:\n{goal}.\n
You need to choose the most relevant hints topic from the following list:\n\nHint topics:\n{topics}\n
Choose hint topic for the task and return only its number, e.g. 1. If you don't know the answer, return -1."""

    def _init(self):
        """Initialize the block."""
        if Path(self.hint_db_rel_path).is_absolute():
            hint_db_path = Path(self.hint_db_rel_path)
        else:
            hint_db_path = Path(__file__).parent / self.hint_db_rel_path
        self.hint_db = pd.read_csv(hint_db_path, header=0, index_col=None, dtype=str)
        if self.hint_retrieval_mode == "emb":
            self.encode_hints()

    def oai_embed(self, text: str):
        response = self._oai_emb.create(input=text, model="text-embedding-3-small")
        return response.data[0].embedding

    def encode_hints(self):
        self.uniq_hints = self.hint_db.drop_duplicates(subset=["hint"], keep="first")
        logger.info(
            f"Encoding {len(self.uniq_hints)} unique hints with semantic keys using {self.embedder_model} model."
        )
        hints = self.uniq_hints["hint"].tolist()
        semantic_keys = self.uniq_hints["semantic_keys"].tolist()
        lines = [f"{k}: {h}" for h, k in zip(hints, semantic_keys)]
        emb_path = f"{self.hint_db_rel_path}.embs.npy"
        assert os.path.exists(emb_path), f"Embedding file not found: {emb_path}"
        logger.info(f"Loading hint embeddings from: {emb_path}")
        emb_dict = np.load(emb_path, allow_pickle=True).item()
        self.hint_embeddings = np.array([emb_dict[k] for k in lines])
        logger.info(f"Loaded hint embeddings shape: {self.hint_embeddings.shape}")

    def apply(self, llm, discussion: StructuredDiscussion, task_name: str) -> dict:
        if not self.use_task_hint:
            return {}

        goal = "\n".join([c.get("text", "") for c in discussion.groups[0].messages[1].content])
        task_hints = self.choose_hints(llm, task_name, goal)

        hints = []
        for hint in task_hints:
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

    def choose_hints(self, llm, task_name: str, goal: str) -> list[str]:
        """Choose hints based on the task name."""
        if self.hint_retrieval_mode == "llm":
            return self.choose_hints_llm(llm, goal)
        elif self.hint_retrieval_mode == "direct":
            return self.choose_hints_direct(task_name)
        elif self.hint_retrieval_mode == "emb":
            return self.choose_hints_emb(goal)
        else:
            raise ValueError(f"Unknown hint retrieval mode: {self.hint_retrieval_mode}")

    def choose_hints_llm(self, llm, goal: str) -> list[str]:
        """Choose hints using LLM to filter the hints."""
        topic_to_hints = defaultdict(list)
        for i, row in self.hint_db.iterrows():
            topic_to_hints[row["semantic_keys"]].append(i)
        hint_topics = list(topic_to_hints.keys())
        topics = "\n".join([f"{i}. {h}" for i, h in enumerate(hint_topics)])
        prompt = self.llm_prompt.format(goal=goal, topics=topics)
        response = llm(APIPayload(messages=[llm.msg.user().add_text(prompt)]))
        try:
            hint_topic_idx = json.loads(response.think)
            if hint_topic_idx < 0 or hint_topic_idx >= len(hint_topics):
                logger.error(f"Wrong LLM hint id response: {response.think}, no hints")
                return []
            hint_topic = hint_topics[hint_topic_idx]
            hint_indices = topic_to_hints[hint_topic]
            df = self.hint_db.iloc[hint_indices].copy()
            df = df.drop_duplicates(subset=["hint"], keep="first")  # leave only unique hints
            hints = df["hint"].tolist()
            logger.debug(f"LLM hint topic {hint_topic_idx}, chosen hints: {df['hint'].tolist()}")
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM hint id response: {response.think}, no hints")
            hints = []
        return hints

    def choose_hints_emb(self, goal: str) -> list[str]:
        """Choose hints using embeddings to filter the hints."""
        goal_embeddings = self._encode([goal], prompt="task description")
        similarities = self._similarity(goal_embeddings.tolist(), self.hint_embeddings.tolist())
        top_indices = similarities.argsort()[0][-self.top_n :].tolist()
        logger.info(f"Top hint indices based on embedding similarity: {top_indices}")
        hints = self.uniq_hints.iloc[top_indices]
        logger.info(f"Embedding-based hints chosen: {hints}")
        return hints["hint"].tolist()

    def _encode(self, texts: list[str], prompt: str = "", timeout: int = 10, max_retries: int = 5):
        """Call the encode API endpoint with timeout and retries"""
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.embedder_server}/encode",
                    json={"texts": texts, "prompt": prompt},
                    timeout=timeout,
                )
                embs = response.json()["embeddings"]
                return np.asarray(embs)
            except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(random.uniform(1, timeout))
                continue

    def _similarity(
        self, texts1: list[str], texts2: list[str], timeout: int = 2, max_retries: int = 5
    ):
        """Call the similarity API endpoint with timeout and retries"""
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.embedder_server}/similarity",
                    json={"texts1": texts1, "texts2": texts2},
                    timeout=timeout,
                )
                similarities = response.json()["similarities"]
                return np.asarray(similarities)
            except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(random.uniform(1, timeout))
                continue

    def choose_hints_direct(self, task_name: str) -> list[str]:
        hints = self.hint_db[
            self.hint_db["task_name"].apply(lambda x: fnmatch.fnmatch(x, task_name))
        ]
        return hints["hint"].tolist()


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
    model_args: BaseModelArgs = None
    config: PromptConfig = None
    use_raw_page_output: bool = False  # This attribute is used in loop.py to setup the env.
    action_set: bgym.AbstractActionSet | None = None

    def __post_init__(self):
        try:
            self.agent_name = f"ToolUse-{self.model_args.model_name}".replace("/", "_")
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


class ToolUseAgent(bgym.Agent):
    def __init__(
        self,
        model_args: OpenAIResponseModelArgs,
        config: PromptConfig = None,
        action_set: bgym.AbstractActionSet | None = None,
    ):
        self.model_args = model_args
        self.config = config
        self.action_set: bgym.AbstractActionSet = action_set or bgym.HighLevelActionSet(
            self.config.action_subsets,
            multiaction=self.config.multiaction,  # type: ignore
        )
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
            self.config.goal.apply(self.llm, self.discussion, obs, sys_msg)

            self.config.summarizer.apply_init(self.llm, self.discussion)
            self.config.general_hints.apply(self.llm, self.discussion)
            self.task_hint.apply(self.llm, self.discussion, self.task_name)

            self.discussion.new_group()

        self.obs_block.apply(self.llm, self.discussion, obs, last_llm_output=self.last_response)

        self.config.summarizer.apply(self.llm, self.discussion)

        messages = self.discussion.flatten()
        response: LLMOutput = self.llm(
            APIPayload(
                messages=messages,
                tools=self.tools,  # You can update tools available tools now.
                tool_choice="any",
                cache_tool_definition=True,
                cache_complete_prompt=False,
                use_cache_breakpoints=True,
            )
        )
        action = response.action
        think = response.think
        last_summary = self.discussion.get_last_summary()
        if last_summary is not None:
            think = last_summary.content[0]["text"] + "\n" + think

        self.discussion.new_group()
        # self.discussion.append(response.tool_calls) # No need to append tool calls anymore.

        self.last_response = response
        self._responses.append(response)  # may be useful for debugging
        # self.messages.append(response.assistant_message)  # this is tool call

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


GPT_4_1 = OpenAIResponseModelArgs(
    model_name="gpt-4.1",
    max_total_tokens=200_000,
    max_input_tokens=200_000,
    max_new_tokens=2_000,
    temperature=0.1,
    vision_support=True,
)

GPT_4_1_CC_API = OpenAIChatModelArgs(
    model_name="gpt-4.1",
    max_total_tokens=200_000,
    max_input_tokens=200_000,
    max_new_tokens=2_000,
    temperature=0.1,
    vision_support=True,
)

GPT_5_mini = OpenAIChatModelArgs(
    model_name="gpt-5-mini-2025-08-07",
    max_total_tokens=400_000,
    max_input_tokens=400_000 - 4_000,
    max_new_tokens=4_000,
    temperature=1,  # Only temperature 1 works for gpt-5-mini
    vision_support=True,
)


GPT_5_nano = OpenAIChatModelArgs(
    model_name="gpt-5-nano-2025-08-07",
    max_total_tokens=400_000,
    max_input_tokens=400_000 - 4_000,
    max_new_tokens=4_000,
    temperature=1,  # Only temperature 1 works for gpt-5-nano
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

CLAUDE_SONNET_37 = ClaudeResponseModelArgs(
    model_name="claude-3-7-sonnet-20250219",
    max_total_tokens=200_000,
    max_input_tokens=200_000,
    max_new_tokens=2_000,
    temperature=0.1,
    vision_support=True,
)

CLAUDE_SONNET_4 = ClaudeResponseModelArgs(
    model_name="claude-sonnet-4-20250514",
    max_total_tokens=200_000,
    max_input_tokens=200_000,
    max_new_tokens=2_000,
    temperature=0.1,
    vision_support=True,
)

O3_RESPONSE_MODEL = OpenAIResponseModelArgs(
    model_name="o3-2025-04-16",
    max_total_tokens=200_000,
    max_input_tokens=200_000,
    max_new_tokens=2_000,
    temperature=None,  # O3 does not support temperature
    vision_support=True,
)
O3_CHATAPI_MODEL = OpenAIChatModelArgs(
    model_name="o3-2025-04-16",
    max_total_tokens=200_000,
    max_input_tokens=200_000,
    max_new_tokens=2_000,
    temperature=None,
    vision_support=True,
)

GPT_5 = OpenAIChatModelArgs(
    model_name="gpt-5",
    max_total_tokens=200_000,
    max_input_tokens=200_000,
    max_new_tokens=8_000,
    temperature=None,
    vision_support=True,
)


GPT_5_MINI = OpenAIChatModelArgs(
    model_name="gpt-5-mini-2025-08-07",
    max_total_tokens=200_000,
    max_input_tokens=200_000,
    max_new_tokens=2_000,
    temperature=1.0,
    vision_support=True,
)

GPT4_1_OPENROUTER_MODEL = OpenRouterModelArgs(
    model_name="openai/gpt-4.1",
    max_total_tokens=200_000,
    max_input_tokens=200_000,
    max_new_tokens=2_000,
    temperature=None,  # O3 does not support temperature
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
    keep_last_n_obs=None,
    multiaction=False,  # whether to use multi-action or not
    # action_subsets=("bid",),
    action_subsets=("coord",),
    # action_subsets=("coord", "bid"),
)

AGENT_CONFIG = ToolUseAgentArgs(
    model_args=CLAUDE_SONNET_37,
    config=DEFAULT_PROMPT_CONFIG,
)

OAI_AGENT = ToolUseAgentArgs(
    model_args=GPT_5_mini,
    config=DEFAULT_PROMPT_CONFIG,
)
GPT5_1_NANO_AGENT = ToolUseAgentArgs(
    model_args=GPT_5_nano,
    config=DEFAULT_PROMPT_CONFIG,
)
GPT5_1_MINI_AGENT = ToolUseAgentArgs(
    model_args=GPT_5_mini,
    config=DEFAULT_PROMPT_CONFIG,
)

OAI_CHATAPI_AGENT = ToolUseAgentArgs(
    model_args=O3_CHATAPI_MODEL,
    config=DEFAULT_PROMPT_CONFIG,
)

OAI_OPENROUTER_AGENT = ToolUseAgentArgs(
    model_args=GPT4_1_OPENROUTER_MODEL,
    config=DEFAULT_PROMPT_CONFIG,
)

OSWORLD_CLAUDE = ToolUseAgentArgs(
    model_args=CLAUDE_SONNET_37,
    config=PromptConfig(
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
        task_hint=TaskHint(use_task_hint=False),
        keep_last_n_obs=None,
        multiaction=False,  # whether to use multi-action or not
        action_subsets=("coord",),  # or "bid"
    ),
    action_set=OSWorldActionSet("computer_13"),  # or "pyautogui"
)

OSWORLD_OAI = ToolUseAgentArgs(
    model_args=GPT_4_1_MINI,
    config=PromptConfig(
        tag_screenshot=True,
        goal=Goal(goal_as_system_msg=True),
        obs=Obs(
            use_last_error=True,
            use_screenshot=True,
            use_axtree=False,
            use_dom=False,
            use_som=False,
            use_tabs=False,
        ),
        summarizer=Summarizer(do_summary=True),
        general_hints=GeneralHints(use_hints=False),
        task_hint=TaskHint(use_task_hint=False),
        keep_last_n_obs=1,  # keep only the last observation in the discussion
        multiaction=False,  # whether to use multi-action or not
        action_subsets=("coord",),
    ),
    action_set=OSWorldActionSet("computer_13"),
)
