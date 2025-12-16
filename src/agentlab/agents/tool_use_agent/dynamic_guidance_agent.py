import logging
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List

import bgym
import faiss
import numpy as np
import pandas as pd
import requests
from agentlab.agents.agent_args import AgentArgs
from agentlab.agents.tool_use_agent.cua_like_agent import (
    ADDITIONAL_ACTION_INSTRUCTIONS,
    GeneralHints,
    Goal,
    Obs,
    PromptConfig,
    StructuredDiscussion,
    Summarizer,
    TaskHint,
    simple_bgym_action_tool,
)
from agentlab.benchmarks.abstract_env import AbstractBenchmark as AgentLabBenchmark
from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.litellm_api import LiteLLMModelArgs
from agentlab.llm.llm_utils import image_to_png_base64_url
from agentlab.llm.response_api import (
    APIPayload,
    LLMOutput,
    MessageBuilder,
    OpenAIChatCompletionAPIMessageBuilder,
)
from agentlab.llm.tracking import cost_tracker_decorator
from bgym import Benchmark as BgymBenchmark
from browsergym.core.observation import extract_screenshot
from datasets import Dataset, disable_progress_bar, enable_progress_bar
from litellm import embedding

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

USER_AGENT_QUERY_SYS_MSG = """You are an agent in a multi-agent system for the ServiceNow platform. You are impersonate a real end user who is navigating ServiceNow and asking for help. Your task is to navigate the ServiceNow platform by asking guidance from the **Dynamic Guidance** agent.

For each navigation step, you receive:
- a screenshot of the *previous* UI state,
- a screenshot of the *current* UI state,
- and a short description of the *last action* taken.

Using this context:
- Briefly explain how the last action changed the interface or workflow (if at all).  
- Then ask—conversationally and in the first person—what you should do next. Keep the request generic and concise. Do not specify any implied goal in your ask, just ask what is the next step.

Write your response as a single concise paragraph inside `<query>...</query>` tags. Keep it natural, like you're chatting with someone helping you. Keep it under 40 words."""

USER_AGENT_ACTION_SYS_MSG = """You are an agent in a multi-agent system for the ServiceNow platform. You are impersonate a real end user who is navigating ServiceNow. Your role at this stage is to execute an action based on guidance you've received.

You are provided:
- a screenshot showing the current UI state, and
- an instruction from the **Dynamic Guidance** agent.

Using the screenshot to understand what is visible and actionable on the interface, interpret the instruction and carry out the corresponding action. If using tools is required, invoke them appropriately based on the instruction and what the screenshot allows.

Respond only with the action you perform—no explanation."""

USER_ACTION_HINT_QUERY_SYS_MSG = """You are an agent in a multi-agent system for the ServiceNow platform. Your job is to generate a search query based on the provided context in order to search a *hint* database to help you solve the user's goal.

You are provided:
- the user goal
- a screenshot showing the current UI state, and 
- [optional] the previous user query detailing the state of the UI at the time the query was made.

Using this context, generate a search query that will help you find the relevant hint to solve the user's goal.

Write your response as a single concise sentence or paragraph inside `<query>...</query>` tags. Keep it under 40 words."""

DYNAMIC_GUIDANCE_AGENT_PLANNING_SYS_MSG = """You are the **Dynamic Guidance** agent in a multi-agent ServiceNow system. 
Your role is to help an end-user achieve their goal by analyzing what is currently visible in the UI and deciding what they should do next.

Using the provided information, reason about the goal and create a plan describing the steps needed to achieve that goal.

Be factual, avoid assumptions beyond what is visible, and provide only relevant guidance.
Your output must use the following structure:
<plan>...</plan>"""

# TODO: bring prompt closer to prompt used in product: https://code.devsnc.com/dev/sn-help-assistant/blob/master/src/sn-help-assistant/behaviors/live-client/constants.js

DYNAMIC_GUIDANCE_AGENT_SYS_MSG = """You are the **Dynamic Guidance** agent in a multi-agent ServiceNow system. 
Your role is to help an end-user achieve their goal by analyzing what is currently visible in the UI and deciding what they should do next.

Using provided information, suggest the immediate next action the user should take right now. Keep this answer concise and to the point. Use less than 20 words.

Be factual, avoid assumptions beyond what is visible, and provide only relevant guidance.
Your output must use the following structure:
<guidance>...</guidance>"""


def action_from_generalized_bgym_action_tool(response: LLMOutput, tool_name: str = "get_action") -> str | None:
    """Extract the action string from the tool call in the LLM response."""
    # TODO: multiaction does not seem to work right now. We only extract a single action and I am unsure how it is processed by the env afterwards. make sure this works.
    actions = []
    if response.tool_calls is not None:
        for tc in response.tool_calls.tool_calls:
            if tc.name == tool_name:
                actions.append(tc.arguments.get("action"))

    action = "\n".join(actions)
    return action


def prepare_messagesbuilder_messages(messages: List[Dict[str, Any]], num_screenshots=5) -> List[MessageBuilder]:

    # remove screenshots older than num_screenshots
    messages = deepcopy(messages)
    cntr = 0
    # go over messages in reverse order to remove old screenshots
    # HACK: clean this up and improve logic
    for i in range(1, len(messages) + 1):
        message = messages[-i]
        if message["role"] == "user":
            # go over messages in normal order.
            for j in range(len(message["content"]) - 1):
                if (
                    message["content"][j]["type"] == "input_text"
                    and message["content"][j]["text"] == "[SCREENSHOT]"
                    and message["content"][j + 1]["type"] == "input_image"
                ):
                    cntr += 1
                    if cntr > num_screenshots:
                        # replace both j and j+1 with a single j with message [SCREENSHOT PLACEHOLDER]
                        message["content"][j] = {"type": "input_text", "text": "[SCREENSHOT PLACEHOLDER]"}
                        message["content"] = message["content"][: j + 1] + message["content"][j + 2 :]
                        # TODO: support removing more than 1 screenshot per user turn
                        break

    # convert to messagesbuilder messages
    new_messages = []
    for message in messages:
        new_message = OpenAIChatCompletionAPIMessageBuilder(role=message["role"])
        if isinstance(message["content"], str):
            new_message.add_text(message["content"])
        elif isinstance(message["content"], list):
            for content in message["content"]:
                if content["type"] == "input_text":
                    new_message.add_text(content["text"])
                elif content["type"] == "input_image":
                    new_message.add_image(content["image_url"])
        new_messages.append(new_message)
    return new_messages


@dataclass
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
        benchmark_action_set = deepcopy(benchmark.high_level_action_set_args).make_action_set().action_set
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

        # used to define whether to plan, do docs rag, episode-level hinting, ...
        self.first_iteration = True

        # custom
        self.screenshots = []
        # TODO: enable changing these flags via config
        self.use_docs_search = True
        self.use_planning = True
        # self.use_hinting = False
        self.use_hinting = True
        self.plan_str = None
        self.docs_str = None
        self.hints_str = None
        self.conversation_for_traces = []

        if self.use_hinting:
            self.prepare_hints_db_index()

    def prepare_hints_db_index(self):

        # TODO: change hints path
        hints_db_path = "/mnt/agentlab_results/2025-12-16_01-40-08_dg-us-anthropic-claude-sonnet-4-5-20250929-v1-0-on-workarena-dynamic-guidance/hints/us.anthropic.claude-sonnet-4-5-20250929-v1:0/hints_db.csv"
        # hints_db_path = "/mnt/agentlab_results/2025-11-18_20-21-04_cua-us-anthropic-claude-sonnet-4-5-20250929-v1-0-on-webarena/hints/us.anthropic.claude-sonnet-4-5-20250929-v1:0/hints_db.csv"
        hints_df = pd.read_csv(hints_db_path)
        self.hints_ds = Dataset.from_pandas(hints_df)

        def get_batch_embeddings(batch):
            embedding_response = embedding(model="gemini-embedding-001", input=batch, input_type="RETRIEVAL_DOCUMENT")
            hints_embeddings = [elem["embedding"] for elem in embedding_response.data]
            return hints_embeddings

        self.hints_ds = self.hints_ds.map(lambda x: {"embedding": get_batch_embeddings(x["hint"])}, batched=True, batch_size=32)
        self.hints_ds.add_faiss_index(column="embedding", metric_type=faiss.METRIC_INNER_PRODUCT)

    def obs_preprocessor(self, obs):
        obs = deepcopy(obs)
        if self.config.obs.skip_preprocessing:
            return obs
        page = obs.pop("page", None)
        if page is not None:
            obs["screenshot"] = extract_screenshot(page)
        return obs

    def set_task_name(self, task_name: str):
        self.task_name = task_name

    def get_docs(self, obs: Any) -> None:
        # HACK: quick and dirty servicenow docs search
        from bs4 import BeautifulSoup
        from googleapiclient.discovery import build
        from markdownify import markdownify as md

        SECTION_LABEL_SELECTORS = [
            "p.sectiontitle",
            "p.tasklabel",
            "p.sectiontitle.tasklabel",
            # Add any others you see in the DOM:
            "p.proceduretitle",
            "p.prereqtitle",
            "p.relatedtopicstitle",
            "p.notetitle",
        ]

        # Map exact label text -> heading level (default h3 if not listed)
        LABEL_LEVELS = {
            "Before you begin": 2,
            "Procedure": 2,
            "About this task": 2,
            "Role required": 3,
            "Results": 2,
            "Next steps": 2,
            "Related topics": 2,
        }

        def promote_section_labels_to_headings(soup: BeautifulSoup):
            for sel in SECTION_LABEL_SELECTORS:
                for p in soup.select(sel):
                    text = p.get_text(" ", strip=True)
                    if not text:
                        p.decompose()
                        continue
                    level = LABEL_LEVELS.get(text, 3)  # default to ### if unknown
                    h = soup.new_tag(f"h{min(max(level,1),6)}")
                    h.string = text
                    p.replace_with(h)

        def convert_html_to_markdown(html):

            soup = BeautifulSoup(html, "html.parser")
            # Drop obvious chrome if any slipped in
            for sel in ["nav", "footer", "aside", ".breadcrumbs", "[role='navigation']"]:
                for el in soup.select(sel):
                    el.decompose()
            promote_section_labels_to_headings(soup)

            # Optional: demote multiple H1s to H2s (keep structure tidy)
            h1s = soup.find_all("h1")
            for h in h1s[1:]:
                h.name = "h2"

            cleaned_html = str(soup)

            # Convert to Markdown
            md_text = md(cleaned_html, heading_style="ATX", code_language=None, escape_asterisks=False)  # #, ##, ### …  # don't guess languages

            # Tidy up whitespace
            md_text = re.sub(r"\n{3,}", "\n\n", md_text).strip() + "\n"
            return md_text

        def extract_main_text(url: str):

            # convert the url to the api endpoint format
            url = url.replace("https://www.servicenow.com/docs/bundle/", "https://servicenow-be-prod.servicenow.com/api/bundle/")

            response = requests.get(url, headers={"Accept": "application/json"})

            page_content = response.json()
            article_content = page_content["topic_html"]
            article_text = convert_html_to_markdown(article_content)
            return article_text

        NUM_RESULTS = 3  # limit number of docs returned
        DOMAIN = "servicenow.com/docs"  # e.g., "docs.servicenow.com" or None
        USE_SITESEARCH_PARAM = True  # False => use "site:example.com" in the query instead
        siterestrict = False

        google_api_key = os.getenv("GOOGLE_API_KEY")
        google_cse_id = os.getenv("GOOGLE_CSE_ID")
        if not google_api_key or not google_cse_id:
            raise ValueError("Missing GOOGLE_API_KEY or GOOGLE_CSE_ID in environment.")

        search_engine = build("customsearch", "v1", developerKey=google_api_key)
        cse = search_engine.cse()
        if siterestrict:
            cse = cse.siterestrict()

        # TODO: enable LLM to generate search term
        search_term = obs["goal_object"][0]["text"]

        params = {
            "q": search_term if not (DOMAIN and not USE_SITESEARCH_PARAM) else f"site:{DOMAIN} {search_term}",
            "cx": google_cse_id,
            "num": NUM_RESULTS,
            "fields": "items(title,link,snippet)",
        }

        if DOMAIN and USE_SITESEARCH_PARAM:
            params["siteSearch"] = DOMAIN
            params["siteSearchFilter"] = "i"  # include only this domain

        res = cse.list(**params).execute()
        items = res.get("items", []) or []
        all_texts = []
        for it in items:
            link = it.get("link")
            text = extract_main_text(link)
            if text:
                preview = text[:2500]
                if len(preview) < len(text):
                    preview += "\n\n... [truncated]"
                all_texts.append(preview)

        self.docs_str = "\n\n----------------\n\n".join(all_texts)

    # @cost_tracker_decorator
    def get_plan(self, obs: Any) -> None:
        current_image_base64 = image_to_png_base64_url(obs["screenshot"])
        goal_str = obs["goal_object"][0]["text"]

        planning_messages = [{"role": "system", "content": DYNAMIC_GUIDANCE_AGENT_PLANNING_SYS_MSG}]
        user_content = [
            {
                "type": "input_text",
                "text": "[SCREENSHOT]",
            },
            {"type": "input_image", "image_url": current_image_base64},
            {
                "type": "input_text",
                "text": "[GOAL]",
            },
            {
                "type": "input_text",
                "text": goal_str,
            },
        ]
        if self.docs_str:
            user_content.append({"type": "input_text", "text": "[DOCS]"})
            user_content.append({"type": "input_text", "text": self.docs_str})
        planning_messages.append({"role": "user", "content": user_content})

        planning_messages = prepare_messagesbuilder_messages(planning_messages)
        planning_response: LLMOutput = self.llm(
            APIPayload(
                messages=planning_messages,
                cache_tool_definition=True,
                cache_complete_prompt=False,
                use_cache_breakpoints=True,
                reasoning_effort="low",
            )
        )
        planning_response = planning_response.raw_response
        planning_response_text = planning_response.choices[0].message.content
        plan = planning_response_text.strip().split("<plan>")[1].split("</plan>")[0]
        self.plan_str = plan

    def get_hints(self, obs: Any, previous_user_message: str | None = None) -> None:
        # TODO: enable query rewrite with LLM for hint retrieval
        goal_str = obs["goal_object"][0]["text"]
        current_image_base64 = image_to_png_base64_url(obs["screenshot"])

        messages = [
            {"role": "system", "content": USER_ACTION_HINT_QUERY_SYS_MSG},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "[GOAL]",
                    },
                    {
                        "type": "input_text",
                        "text": goal_str,
                    },
                    {
                        "type": "input_text",
                        "text": "[SCREENSHOT]",
                    },
                    {
                        "type": "input_image",
                        "image_url": current_image_base64,
                    },
                ],
            },
        ]

        if previous_user_message is not None:
            messages[-1]["content"] += [
                {
                    "type": "input_text",
                    "text": "[PREVIOUS USER MESSAGE]",
                },
                {
                    "type": "input_text",
                    "text": previous_user_message,
                },
            ]

        hint_query_messages = prepare_messagesbuilder_messages(messages)    
        hint_query_response: LLMOutput = self.llm(
            APIPayload(
                messages=hint_query_messages,
                reasoning_effort="low",
            )
        )

        hint_query_response = hint_query_response.raw_response
        hint_query_str = hint_query_response.choices[0].message.content
        hint_query_str = hint_query_str.strip().split("<query>")[1].split("</query>")[0]

        # TODO: parametrize embedding model
        # NOTE: input_type could also be "QUESTION_ANSWERING"
        embedding_response = embedding(model="gemini-embedding-001", input=[hint_query_str], input_type="RETRIEVAL_QUERY")
        query_embedding = embedding_response.data[0]["embedding"]
        _, retrieved_hints = self.hints_ds.get_nearest_examples("embedding", np.array(query_embedding), k=3)
        retrieved_hints = retrieved_hints["hint"]
        self.hints_str = "\n".join(["* " + hint for hint in retrieved_hints])

    def get_step_hints(self, obs: Any) -> None:
        # TODO: enable step-level hinting
        pass

    # @cost_tracker_decorator
    def get_user_query(self, obs: Any) -> str:
        current_image_base64 = image_to_png_base64_url(obs["screenshot"])
        messages = [
            {"role": "system", "content": USER_AGENT_QUERY_SYS_MSG},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "[PREVIOUS SCREENSHOT]",
                    },
                    {
                        "type": "input_image",
                        "image_url": self.screenshots[-1],
                    },
                    {
                        "type": "input_text",
                        "text": "[CURRENT SCREENSHOT]",
                    },
                    {
                        "type": "input_image",
                        "image_url": current_image_base64,
                    },
                    {
                        "type": "input_text",
                        "text": "[LAST_ACTION]",
                    },
                    {
                        "type": "input_text",
                        "text": obs["last_action"],
                    },
                ],
            },
        ]

        messages = prepare_messagesbuilder_messages(messages)
        response = self.llm(
            APIPayload(
                messages=messages,
                reasoning_effort="low",
            )
        )
        response = response.raw_response
        response_text = response.choices[0].message.content
        response_text = response_text.strip().split("<query>")[1].split("</query>")[0]
        return response_text

    # @cost_tracker_decorator
    def get_user_action(self, obs: Any, dynamic_guidance_response_text: str):
        current_image_base64 = image_to_png_base64_url(obs["screenshot"])
        user_agent_action_sys_msg = deepcopy(USER_AGENT_ACTION_SYS_MSG)

        if self.config.multiaction:
            user_agent_action_sys_msg += "\nYou can take multiple actions in a single step, if needed."
        else:
            user_agent_action_sys_msg += "\nYou can only take one action at a time."

        user_agent_action_sys_msg += "\nAvailable browsergym actions that can be returned with get_action:\n" + self.action_set.describe()
        user_agent_action_sys_msg += ADDITIONAL_ACTION_INSTRUCTIONS

        user_action_messages = [
            {"role": "system", "content": user_agent_action_sys_msg},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "[SCREENSHOT]",
                    },
                    {
                        "type": "input_image",
                        "image_url": current_image_base64,
                    },
                    {
                        "type": "input_text",
                        "text": "[INSTRUCTION]",
                    },
                    {
                        "type": "input_text",
                        "text": dynamic_guidance_response_text,
                    },
                ],
            },
        ]
        user_action_messages = prepare_messagesbuilder_messages(user_action_messages)
        user_action_response: LLMOutput = self.llm(
            APIPayload(
                messages=user_action_messages,
                tools=self.tools,
                tool_choice="auto",  # auto must be enabled to get reasoning for tool calls.
                reasoning_effort="low",
            )
        )
        if self.config.use_generalized_bgym_action_tool:
            action = action_from_generalized_bgym_action_tool(user_action_response)
        else:
            action = user_action_response.action

        if action is None and self.config.use_noop_as_default_action:
            action = "noop()"  # default action is noop if none is provided

        self.last_response = user_action_response
        self._responses.append(user_action_response)  # may be useful for debugging

        return action

    @cost_tracker_decorator
    def get_action(self, obs: Any) -> float:
        self.llm.reset_stats()

        current_image_base64 = image_to_png_base64_url(obs["screenshot"])
        goal_str = obs["goal_object"][0]["text"]

        if self.use_docs_search and not self.docs_str:
            self.get_docs(obs)

        if self.use_planning and not self.plan_str:
            self.get_plan(obs)

        if self.first_iteration:

            if self.use_hinting:
                self.get_hints(obs)

            # Prepare context with available information
            self.conversation_for_traces.append({"role": "system", "content": DYNAMIC_GUIDANCE_AGENT_SYS_MSG})

            user_content = [
                {"type": "input_text", "text": "[GOAL]"},
                {"type": "input_text", "text": goal_str},
            ]
            if self.docs_str:
                user_content.append({"type": "input_text", "text": "[DOCS]"})
                user_content.append({"type": "input_text", "text": self.docs_str})
            if self.hints_str:
                user_content.append({"type": "input_text", "text": "[HINTS]"})
                user_content.append({"type": "input_text", "text": self.hints_str})
            if self.plan_str:
                user_content.append({"type": "input_text", "text": "[PLAN]"})
                user_content.append({"type": "input_text", "text": self.plan_str})
            user_content.append({"type": "input_text", "text": "[SCREENSHOT]"})
            user_content.append({"type": "input_image", "image_url": current_image_base64})
            self.conversation_for_traces.append({"role": "user", "content": user_content})
            self.first_iteration = False

        else:
            user_query = self.get_user_query(obs)
            self.conversation_for_traces.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "[SCREENSHOT]"},
                        {"type": "input_image", "image_url": current_image_base64},
                        {"type": "input_text", "text": "[QUERY]"},
                        {"type": "input_text", "text": user_query},
                    ],
                }
            )

            if self.use_hinting:
                self.get_hints(obs, user_query)
                self.conversation_for_traces.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "[HINTS]"},
                            {"type": "input_text", "text": self.hints_str},
                        ],
                    }
                )

        # using the current context, the agent llm instructs the user llm to perform a given action (in natural language)
        dynamic_guidance_messages = prepare_messagesbuilder_messages(deepcopy(self.conversation_for_traces))
        dynamic_guidance_response: LLMOutput = self.llm(
            APIPayload(
                messages=dynamic_guidance_messages,
                reasoning_effort="low",
            )
        )
        dynamic_guidance_response = dynamic_guidance_response.raw_response
        dynamic_guidance_response_text = dynamic_guidance_response.choices[0].message.content
        self.conversation_for_traces.append({"role": "assistant", "content": [{"type": "input_text", "text": dynamic_guidance_response_text}]})
        dynamic_guidance_response_text = dynamic_guidance_response_text.strip().split("<guidance>")[1].split("</guidance>")[0]

        # based on the current screenshot (for grounding) and the agent llm instruction ONLY, the user llm performs one of the provided actions
        action = self.get_user_action(obs, dynamic_guidance_response_text)

        self.screenshots.append(current_image_base64)

        # NOTE: we don't include the user action in the messages since the dynamic agent doesn't have access to the user action at inference time.
        # self.conversation_for_traces.append({"role": "user", "content": [{"type": "input_text", "text": f"[TOOL CALL]\n{action}"}]})

        chat_messages = prepare_messagesbuilder_messages(self.conversation_for_traces)
        agent_info = bgym.AgentInfo(
            think=dynamic_guidance_response_text,
            chat_messages=chat_messages,
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
    multiaction=True,  # multiaction enabled since dynamic guidance often tells the user to do 2-3 small actions in a row
    # TODO: reuse keep_last_n_obs to truncate screenshots (currently not doing it)
    keep_last_n_obs=5,
)


def get_dynamic_guidance_agent_config(
    model_name: str = "gemini-3.0-pro-preview",
    embedding_model_name: str = "gemini-embedding-001",
) -> DynamicGuidanceAgentArgs:

    return DynamicGuidanceAgentArgs(
        model_args=LiteLLMModelArgs(
            model_name=model_name,
            # embedding_model_name=embedding_model_name,
            max_new_tokens=20000,
            temperature=None,
        ),
        config=DYNAMIC_GUIDANCE_PROMPT_CONFIG,
    )


def get_dynamic_guidance_agent_config_with_hint(
    model_name: str = "gemini-3.0-pro-preview",
    embedding_model_name: str = "gemini-embedding-001",
    hints_db_path: str = None,
    hint_retrieval_mode: str = "direct",
) -> DynamicGuidanceAgentArgs:
    # NOTE: this is the same as the other one, we hardcode some stuff in the class for now...
    return DynamicGuidanceAgentArgs(
        model_args=LiteLLMModelArgs(
            model_name=model_name,
            # embedding_model_name=embedding_model_name,
            max_new_tokens=20000,
            temperature=None,
        ),
        config=DYNAMIC_GUIDANCE_PROMPT_CONFIG,
    )
