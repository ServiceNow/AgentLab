"""
GenericAgent implementation for AgentLab

This module defines a `GenericAgent` class and its associated arguments for use in the AgentLab framework. \
The `GenericAgent` class is designed to interact with a chat-based model to determine actions based on \
observations. It includes methods for preprocessing observations, generating actions, and managing internal \
state such as plans, memories, and thoughts. The `GenericAgentArgs` class provides configuration options for \
the agent, including model arguments and flags for various behaviors.
"""

import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from warnings import warn

import pandas as pd
from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.chat_api import BaseModelArgs
from agentlab.llm.llm_utils import Discussion, ParseError, SystemMessage, retry
from agentlab.llm.tracking import cost_tracker_decorator
from agentlab.utils.hinting import HintsSource
from bgym import Benchmark
from browsergym.experiments.agent import Agent, AgentInfo

from .generic_agent_prompt import (
    GenericPromptFlags,
    MainPrompt,
    StepWiseContextIdentificationPrompt,
)


@dataclass
class GenericAgentArgs(AgentArgs):
    chat_model_args: BaseModelArgs = None
    flags: GenericPromptFlags = None
    max_retry: int = 4

    def __post_init__(self):
        try:  # some attributes might be temporarily args.CrossProd for hyperparameter generation
            # TODO: Rename the agent to HintUseAgent when appropriate
            self.agent_name = f"GenericAgent-hinter-{self.chat_model_args.model_name}".replace(
                "/", "_"
            )
        except AttributeError:
            pass

    def set_benchmark(self, benchmark: Benchmark, demo_mode):
        """Override Some flags based on the benchmark."""
        if benchmark.name.startswith("miniwob"):
            self.flags.obs.use_html = True

        self.flags.obs.use_tabs = benchmark.is_multi_tab
        self.flags.action.action_set = deepcopy(benchmark.high_level_action_set_args)

        # for backward compatibility with old traces
        if self.flags.action.multi_actions is not None:
            self.flags.action.action_set.multiaction = self.flags.action.multi_actions
        if self.flags.action.is_strict is not None:
            self.flags.action.action_set.strict = self.flags.action.is_strict

        # verify if we can remove this
        if demo_mode:
            self.flags.action.action_set.demo_mode = "all_blue"

    def set_reproducibility_mode(self):
        self.chat_model_args.temperature = 0

    def prepare(self):
        return self.chat_model_args.prepare_server()

    def close(self):
        return self.chat_model_args.close_server()

    def make_agent(self):
        return GenericAgent(
            chat_model_args=self.chat_model_args, flags=self.flags, max_retry=self.max_retry
        )


class GenericAgent(Agent):

    def __init__(
        self,
        chat_model_args: BaseModelArgs,
        flags: GenericPromptFlags,
        max_retry: int = 4,
    ):

        self.chat_llm = chat_model_args.make_model()
        self.chat_model_args = chat_model_args
        self.max_retry = max_retry

        self.flags = flags

        if self.flags.hint_db_path is not None and self.flags.use_task_hint:
            assert os.path.exists(
                self.flags.hint_db_path
            ), f"Hint database path {self.flags.hint_db_path} does not exist."
        self.action_set = self.flags.action.action_set.make_action_set()
        self._obs_preprocessor = dp.make_obs_preprocessor(flags.obs)

        self._init_hints_index()

        self._check_flag_constancy()
        self.reset(seed=None)

    def obs_preprocessor(self, obs: dict) -> dict:
        return self._obs_preprocessor(obs)

    def set_task_name(self, task_name: str):
        """Set the task name for task hints functionality."""
        self.task_name = task_name

    @cost_tracker_decorator
    def get_action(self, obs):

        self.obs_history.append(obs)

        system_prompt = SystemMessage(dp.SystemPrompt().prompt)

        # use those queries to retrieve from the database and pass to prompt if step-level
        self.queries = (
            self._get_queries()[0]
            if getattr(self.flags, "hint_level", "episode") == "step"
            else None
        )

        # get hints
        if self.flags.use_hints:
            task_hints = self._get_task_hints()
        else:
            task_hints = []

        main_prompt = MainPrompt(
            action_set=self.action_set,
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            previous_plan=self.plan,
            step=self.plan_step,
            flags=self.flags,
            llm=self.chat_llm,
            task_hints=task_hints,
        )

        # Set task name for task hints if available
        if self.flags.use_task_hint and hasattr(self, "task_name"):
            main_prompt.set_task_name(self.task_name)

        max_prompt_tokens, max_trunc_itr = self._get_maxes()

        human_prompt = dp.fit_tokens(
            shrinkable=main_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.chat_model_args.model_name,
            max_iterations=max_trunc_itr,
            additional_prompts=system_prompt,
        )
        try:
            # TODO, we would need to further shrink the prompt if the retry
            # cause it to be too long

            chat_messages = Discussion([system_prompt, human_prompt])
            ans_dict = retry(
                self.chat_llm,
                chat_messages,
                n_retry=self.max_retry,
                parser=main_prompt._parse_answer,
            )
            ans_dict["busted_retry"] = 0
            # inferring the number of retries, TODO: make this less hacky
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ParseError as e:
            ans_dict = dict(
                action=None,
                n_retry=self.max_retry + 1,
                busted_retry=1,
            )

        stats = self.chat_llm.get_stats()
        stats["n_retry"] = ans_dict["n_retry"]
        stats["busted_retry"] = ans_dict["busted_retry"]

        self.plan = ans_dict.get("plan", self.plan)
        self.plan_step = ans_dict.get("step", self.plan_step)
        self.actions.append(ans_dict["action"])
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        agent_info = AgentInfo(
            think=ans_dict.get("think", None),
            chat_messages=chat_messages,
            stats=stats,
            extra_info={"chat_model_args": asdict(self.chat_model_args)},
        )
        return ans_dict["action"], agent_info

    def _get_queries(self):
        """Retrieve queries for hinting."""
        system_prompt = SystemMessage(dp.SystemPrompt().prompt)
        query_prompt = StepWiseContextIdentificationPrompt(
            obs_history=self.obs_history,
            actions=self.actions,
            thoughts=self.thoughts,
            obs_flags=self.flags.obs,
            n_queries=self.flags.n_retrieval_queries,  # TODO
        )

        chat_messages = Discussion([system_prompt, query_prompt.prompt])
        # BUG: Parsing fails multiple times.
        ans_dict = retry(
            self.chat_llm,
            chat_messages,
            n_retry=self.max_retry,
            parser=query_prompt._parse_answer,
        )

        queries = ans_dict.get("queries", [])
        assert len(queries) <= self.flags.n_retrieval_queries

        # TODO: we should probably propagate these chat_messages to be able to see them in xray
        return queries, ans_dict.get("think", None)

    def reset(self, seed=None):
        self.seed = seed
        self.plan = "No plan yet"
        self.plan_step = -1
        self.memories = []
        self.thoughts = []
        self.actions = []
        self.obs_history = []

    def _check_flag_constancy(self):
        flags = self.flags
        if flags.obs.use_som:
            if not flags.obs.use_screenshot:
                warn(
                    """
Warning: use_som=True requires use_screenshot=True. Disabling use_som."""
                )
                flags.obs.use_som = False
        if flags.obs.use_screenshot:
            if not self.chat_model_args.vision_support:
                warn(
                    """
Warning: use_screenshot is set to True, but the chat model \
does not support vision. Disabling use_screenshot."""
                )
                flags.obs.use_screenshot = False
        return flags

    def _get_maxes(self):
        maxes = (
            self.flags.max_prompt_tokens,
            self.chat_model_args.max_total_tokens,
            self.chat_model_args.max_input_tokens,
        )
        maxes = [m for m in maxes if m is not None]
        max_prompt_tokens = min(maxes) if maxes else None
        max_trunc_itr = (
            self.flags.max_trunc_itr
            if self.flags.max_trunc_itr
            else 20  # dangerous to change the default value here?
        )
        return max_prompt_tokens, max_trunc_itr

    def _init_hints_index(self):
        """Initialize the block."""
        try:
            if self.flags.hint_type == "docs":
                if self.flags.hint_index_type == "direct":
                    print(
                        "WARNING: Hint index type 'direct' is not supported for docs. Using 'sparse' instead."
                    )
                    self.flags.hint_index_type = "sparse"
                if self.flags.hint_index_type == "sparse":
                    import bm25s

                    self.hint_index = bm25s.BM25.load(self.flags.hint_index_path, load_corpus=True)
                elif self.flags.hint_index_type == "dense":
                    from datasets import load_from_disk
                    from sentence_transformers import SentenceTransformer

                    self.hint_index = load_from_disk(self.flags.hint_index_path)
                    self.hint_index.load_faiss_index(
                        "embeddings", self.flags.hint_index_path.removesuffix("/") + ".faiss"
                    )
                    self.hint_retriever = SentenceTransformer(self.flags.hint_retriever_path)
                else:
                    raise ValueError(f"Unknown hint index type: {self.flags.hint_index_type}")
            else:
                # Use external path if provided, otherwise fall back to relative path
                if self.flags.hint_db_path and Path(self.flags.hint_db_path).exists():
                    hint_db_path = Path(self.flags.hint_db_path)
                else:
                    hint_db_path = Path(__file__).parent / self.flags.hint_db_rel_path

                if hint_db_path.exists():
                    self.hint_db = pd.read_csv(hint_db_path, header=0, index_col=None, dtype=str)
                    # Verify the expected columns exist
                    if (
                        "task_name" not in self.hint_db.columns
                        or "hint" not in self.hint_db.columns
                    ):
                        print(
                            f"Warning: Hint database missing expected columns. Found: {list(self.hint_db.columns)}"
                        )
                        self.hint_db = pd.DataFrame(columns=["task_name", "hint"])
                else:
                    print(f"Warning: Hint database not found at {hint_db_path}")
                    self.hint_db = pd.DataFrame(columns=["task_name", "hint"])
                self.hints_source = HintsSource(
                    hint_db_path=hint_db_path.as_posix(),
                    hint_retrieval_mode=self.flags.hint_retrieval_mode,
                    skip_hints_for_current_task=self.flags.skip_hints_for_current_task,
                )
        except Exception as e:
            # Fallback to empty database on any error
            print(f"Warning: Could not load hint database: {e}")
            self.hint_db = pd.DataFrame(columns=["task_name", "hint"])

    def _get_task_hints(self) -> list[str]:
        """Get hints for a specific task."""
        if not self.flags.use_task_hint:
            return []

        if self.flags.hint_type == "docs":
            if self.flags.hint_index_type == "direct":
                print(
                    "WARNING: Hint index type 'direct' is not supported for docs. Using 'sparse' instead."
                )
                self.flags.hint_index_type = "sparse"
            if not hasattr(self, "hint_index"):
                print("WARNING: Hint index not initialized. Initializing now.")
                self._init_hints_index()
            if self.flags.hint_query_type == "goal":
                query = self.obs_history[-1]["goal_object"][0]["text"]
            elif self.flags.hint_query_type == "llm":
                queries, _ = self._get_queries()
                # HACK: only 1 query supported
                query = queries[0]
            else:
                raise ValueError(f"Unknown hint query type: {self.flags.hint_query_type}")

            print(f"Query: {query}")
            if self.flags.hint_index_type == "sparse":
                import bm25s

                query_tokens = bm25s.tokenize(query)
                docs, _ = self.hint_index.retrieve(query_tokens, k=self.flags.hint_num_results)
                docs = [elem["text"] for elem in docs[0]]
                # HACK: truncate to 20k characters (should cover >99% of the cases)
                for doc in docs:
                    if len(doc) > 20000:
                        doc = doc[:20000]
                        doc += " ...[truncated]"
            elif self.flags.hint_index_type == "dense":
                query_embedding = self.hint_retriever.encode(query)
                _, docs = self.hint_index.get_nearest_examples(
                    "embeddings", query_embedding, k=self.flags.hint_num_results
                )
                docs = docs["text"]

            return docs

        # Check if hint_db has the expected structure
        if (
            self.hint_db.empty
            or "task_name" not in self.hint_db.columns
            or "hint" not in self.hint_db.columns
        ):
            return []

        try:
            # When step-level, pass queries as goal string to fit the llm_prompt
            goal_or_queries = self.obs_history[-1]["goal_object"][0]["text"]
            if self.flags.hint_level == "step" and self.queries:
                goal_or_queries = "\n".join(self.queries)

            task_hints = self.hints_source.choose_hints(
                self.chat_llm,
                self.task_name,
                goal_or_queries,
            )

            hints = []
            for hint in task_hints:
                hint = hint.strip()
                if hint:
                    hints.append(f"- {hint}")

            return hints
        except Exception as e:
            print(f"Warning: Error getting hints for task {self.task_name}: {e}")

        return []
