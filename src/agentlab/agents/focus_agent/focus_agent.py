import ast
import re
from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import Literal

import bgym
from browsergym.experiments import Agent, AgentInfo

import agentlab.agents.dynamic_prompting as dp
from agentlab.agents.generic_agent.generic_agent import GenericAgent, GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.llm.chat_api import ChatModelArgs
from agentlab.llm.llm_utils import (
    parse_html_tags_raise,
    retry,
)

from .llm_retriever_prompt import (
    LlmRetrieverDefenderPrompt,
    LlmRetrieverPrompt,
    LlmRetrieverPromptFlags,
    NeutralLlmRetrieverPrompt,
    RestrictiveLlmRetrieverPrompt,
)
from .llm_retriever_utils import LlmRetrieverUtils
from .utils import add_line_numbers_to_tree, get_nb_tokens


@dataclass
class FocusAgentArgs(GenericAgentArgs):
    flags: GenericPromptFlags = None
    chat_model_args: ChatModelArgs = None
    retriever_chat_model_args: ChatModelArgs = None
    retriever_prompt_flags: LlmRetrieverPromptFlags = None
    max_retry: int = 4
    keep_structure: bool = False
    strategy: Literal["bid", "role", "bid+role"] = "bid"
    benchmark: str = None
    retriever_type: Literal["line", "defender", "restrictive", "neutral"] = "line"

    agent_name: str = None

    def __post_init__(self):
        try:  # some attributes might be temporarily args.CrossProd for hyperparameter generation
            if self.agent_name == None:
                self.agent_name = f"FocusAgent-{self.chat_model_args.model_name}".replace("/", "_")
        except AttributeError:
            pass

    def set_benchmark(self, benchmark: bgym.Benchmark, demo_mode):
        self.benchmark = benchmark.name
        super().set_benchmark(benchmark, demo_mode)

    def set_reproducibility_mode(self):
        super().set_reproducibility_mode()
        self.retriever_chat_model_args.temperature = 0

    def make_agent(self) -> Agent:
        return FocusAgent(
            self.chat_model_args,
            self.flags,
            retriever_chat_model_args=self.retriever_chat_model_args,
            retriever_prompt_flags=self.retriever_prompt_flags,
            keep_structure=self.keep_structure,
            strategy=self.strategy,
            max_retry=self.max_retry,
            benchmark=self.benchmark,
            retriever_type=self.retriever_type,
        )


class FocusAgent(GenericAgent):
    def __init__(
        self,
        chat_model_args: ChatModelArgs,
        flags: GenericPromptFlags,
        retriever_chat_model_args: ChatModelArgs,
        retriever_prompt_flags: LlmRetrieverPromptFlags,
        keep_structure: bool = False,
        strategy: str = None,
        max_retry: int = 4,
        benchmark: str = None,
        retriever_type: str = None,
    ):
        chat_model_args.temperature = 0  # Set temperature to 0 for deterministic behavior
        super().__init__(chat_model_args, flags, max_retry)

        assert self.flags.obs.use_html is False, "FocusAgent does not support HTML input."

        self.retriever_prompt_flags = retriever_prompt_flags
        self.retriever_chat_model_args = retriever_chat_model_args
        self.keep_structure = keep_structure
        self.strategy = strategy
        self.benchmark = benchmark
        self.retriever_type = retriever_type
        self.retriever_chat_model = self.retriever_chat_model_args.make_model()

    @staticmethod
    def clean_list(text: str) -> str:
        """Clean the answer string by removing anything before or after the returned list."""
        matches = re.findall(r"\((\d+),\s*(\d+)\)", text)
        tuples = [(int(a), int(b)) for a, b in matches]
        return str(tuples)

    def prune_obs(self, obs: dict) -> dict:
        extra_info = {}

        match self.retriever_type:
            case "line":
                self.line_retriever_prompt = LlmRetrieverPrompt(
                    goal=obs["goal"],
                    tree=add_line_numbers_to_tree(obs["axtree_txt"]),
                    screenshot=obs["screenshot"],
                    history=obs["history"],
                    flags=self.retriever_prompt_flags,
                )
            case "defender":
                self.line_retriever_prompt = LlmRetrieverDefenderPrompt(
                    goal=obs["goal"],
                    tree=add_line_numbers_to_tree(obs["axtree_txt"]),
                    screenshot=obs["screenshot"],
                    history=obs["history"],
                    flags=self.retriever_prompt_flags,
                )
            case "restrictive":
                self.line_retriever_prompt = RestrictiveLlmRetrieverPrompt(
                    goal=obs["goal"],
                    tree=add_line_numbers_to_tree(obs["axtree_txt"]),
                    screenshot=obs["screenshot"],
                    history=obs["history"],
                    flags=self.retriever_prompt_flags,
                )
            case "neutral":
                self.line_retriever_prompt = NeutralLlmRetrieverPrompt(
                    goal=obs["goal"],
                    tree=add_line_numbers_to_tree(obs["axtree_txt"]),
                    screenshot=obs["screenshot"],
                    history=obs["history"],
                    flags=self.retriever_prompt_flags,
                )
            case _:
                raise ValueError(f"Unknown retriever type: {self.retriever_type}")

        answer_dict = retry(
            self.retriever_chat_model,
            self.line_retriever_prompt.prompt,
            n_retry=3,
            parser=partial(
                parse_html_tags_raise,
                keys=["think", "answer"],
                merge_multiple=True,
            ),
        )
        extra_info["retriever_answer"] = answer_dict

        answer = self.clean_list(answer_dict["answer"])

        line_ranges = ast.literal_eval(answer)
        if len(line_ranges) <= 0:
            return obs["axtree_txt"], extra_info

        line_numbers = []
        for line_range in line_ranges:
            if isinstance(line_range, tuple):
                start, end = line_range
                line_numbers.extend(range(start, end + 1))
        if self.keep_structure:
            return (
                LlmRetrieverUtils.remove_lines_keep_structure(
                    tree=obs["axtree_txt"],
                    line_numbers=line_numbers,
                    strategy=self.strategy,
                ),
                extra_info,
            )
        else:
            return (
                LlmRetrieverUtils.remove_lines(
                    tree=obs["axtree_txt"],
                    line_numbers=line_numbers,
                ),
                extra_info,
            )

    def get_action(self, obs):
        obs_history_copy = copy(self.obs_history)
        obs_history_copy.append(obs)
        history = dp.History(
            history_obs=obs_history_copy,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            flags=self.flags.obs,
        )
        obs["history"] = history.prompt
        new_obs, extra_info = self.prune_obs(obs)
        if get_nb_tokens(new_obs) < get_nb_tokens(obs["axtree_txt"]):
            obs["axtree_txt"] = new_obs

        action, info = super().get_action(obs)
        info.extra_info.update(extra_info)
        info.extra_info["retriever_prompt"] = self.line_retriever_prompt.prompt
        info.extra_info["pruned_tree"] = obs["axtree_txt"]
        info.html_page = self.format_html_page(
            agent_info=info,
            obs=obs,
        )
        return action, info

    def format_html_page(self, agent_info: AgentInfo, obs: dict) -> str:
        html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Agent Info</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
            }}
            h1 {{
                color: #333;
            }}
            h2 {{
                color: #555;
            }}
            pre {{
                background-color: #333; /* Dark grey background */
                color: #f4f4f4; /* Light grey text */
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                overflow-x: auto;
            }}
            code {{
                font-family: monospace;
            }}
            .image-container {{
                display: flex;
                justify-content: space-between;
                margin-bottom: 20px;
            }}
            .image-container img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            details {{
                margin-bottom: 10px;
            }}
            details pre {{
                max-height: 1200px; /* Set a max height for scrollability */
                overflow-y: auto; /* Enable vertical scrolling */
            }}
        </style>
    </head>
    <body>
       <div class="image-container">
            <figure>
                <img src="screenshot_pre_action_placeholder" alt="Pre-action Screenshot">
                <figcaption>Pre-action Screenshot</figcaption>
            </figure>
            <figure>
                <img src="screenshot_post_action_placeholder" alt="Post-action Screenshot">
                <figcaption>Post-action Screenshot</figcaption>
            </figure>
        </div>
        <h1>Agent Info</h1>
    """
        sections = {}
        line_retriever_agent_prompt = agent_info.get("retriever_prompt", "")
        line_retriever_agent_prompt_text = (
            line_retriever_agent_prompt[1].content if line_retriever_agent_prompt else ""
        )
        sections["LineRetriever Agent"] = {
            "Prompt": line_retriever_agent_prompt_text,
        }
        sections["LineRetriever Answer"] = {
            "Think": agent_info.extra_info.get("retriever_think", ""),
            "Answer": agent_info.extra_info.get("retriever_answer", ""),
        }
        sections["Pruned AxTree"] = {
            "Pruned Tree": agent_info.get("pruned_tree", ""),
        }
        for section_title, subsections in sections.items():
            html_template += f"""
            <h2>{section_title}</h2>
            """
            for subsection_title, content in subsections.items():
                if not content:
                    continue
                # wrap the prompt is a collapsible (default collapsed) and scrollable div
                if subsection_title in {"Prompt", "AxTree"}:
                    html_template += f"""
                    <h3>{subsection_title}</h3>
                    <details>
                        <summary>Expand Content</summary>
                        <pre><code>{content}</code></pre>
                    </details>
                    """
                else:
                    html_template += f"""
                    <h3>{subsection_title}</h3>
                    <pre><code>{content} </code></pre>
                    """
        html_template += """
        </body>
        </html>
        """
        return html_template
