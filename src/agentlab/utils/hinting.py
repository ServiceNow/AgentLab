import fnmatch
import json
import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import requests
from agentlab.llm.chat_api import ChatModel
import re
import json
from agentlab.llm.response_api import APIPayload

logger = logging.getLogger(__name__)


class HintsSource:

    def __init__(
        self,
        hint_db_path: str,
        hint_retrieval_mode: Literal["direct", "llm", "emb"] = "direct",
        skip_hints_for_current_task: bool = False,
        skip_hints_for_current_goal: bool = False,
        top_n: int = 4,
        embedder_model: str = "Qwen/Qwen3-Embedding-0.6B",
        embedder_server: str = "http://localhost:5000",
        llm_prompt: str = """We're choosing hints to help solve the following task:\n{goal}.\n
You need to choose the most relevant hints topic from the following list:\n\nHint topics:\n{topics}\n
Choose hint topic for the task and return only its number. Use the following output format: 
<choice>index</choice> for e.g. <choice>0</choice> for the topic with index 0. If you don't know the answer, return <choice>-1</choice>""",
    ) -> None:
        self.hint_db_path = hint_db_path
        self.hint_retrieval_mode = hint_retrieval_mode
        self.skip_hints_for_current_task = skip_hints_for_current_task
        self.skip_hints_for_current_goal = skip_hints_for_current_goal
        self.top_n = top_n
        self.embedder_model = embedder_model
        self.embedder_server = embedder_server
        self.llm_prompt = llm_prompt

        if Path(hint_db_path).is_absolute():
            self.hint_db_path = Path(hint_db_path).as_posix()
        else:
            self.hint_db_path = (Path(__file__).parent / self.hint_db_path).as_posix()
        self.hint_db = pd.read_csv(
            self.hint_db_path,
            header=0,
            index_col=None,
            converters={
                "trace_paths_json": lambda x: json.loads(x) if pd.notna(x) else [],
                "source_trace_goals": lambda x: json.loads(x) if pd.notna(x) else [],
            },
        )
        logger.info(f"Loaded {len(self.hint_db)} hints from database {self.hint_db_path}")
        if self.hint_retrieval_mode == "emb":
            self.load_hint_vectors()

    def load_hint_vectors(self):
        self.uniq_hints = self.hint_db.drop_duplicates(subset=["hint"], keep="first")
        logger.info(
            f"Encoding {len(self.uniq_hints)} unique hints with semantic keys using {self.embedder_model} model."
        )
        hints = self.uniq_hints["hint"].tolist()
        semantic_keys = self.uniq_hints["semantic_keys"].tolist()
        lines = [f"{k}: {h}" for h, k in zip(hints, semantic_keys)]
        emb_path = f"{self.hint_db_path}.embs.npy"
        assert os.path.exists(emb_path), f"Embedding file not found: {emb_path}"
        logger.info(f"Loading hint embeddings from: {emb_path}")
        emb_dict = np.load(emb_path, allow_pickle=True).item()
        self.hint_embeddings = np.array([emb_dict[k] for k in lines])
        logger.info(f"Loaded hint embeddings shape: {self.hint_embeddings.shape}")

    def choose_hints(self, llm, task_name: str, goal: str) -> list[str]:
        """Choose hints based on the task name."""
        logger.info(
            f"Choosing hints for task: {task_name}, goal: {goal} from db: {self.hint_db_path} using mode: {self.hint_retrieval_mode}"
        )
        if self.hint_retrieval_mode == "llm":
            return self.choose_hints_llm(llm, goal, task_name)
        elif self.hint_retrieval_mode == "direct":
            return self.choose_hints_direct(task_name)
        elif self.hint_retrieval_mode == "emb":
            return self.choose_hints_emb(goal, task_name)
        else:
            raise ValueError(f"Unknown hint retrieval mode: {self.hint_retrieval_mode}")

    def choose_hints_llm(self, llm, goal: str, task_name: str) -> list[str]:
        """Choose hints using LLM to filter the hints."""
        topic_to_hints = defaultdict(list)
        skip_hints = []
        if self.skip_hints_for_current_task:
            skip_hints += self.get_current_task_hints(task_name)
        if self.skip_hints_for_current_goal:
            skip_hints += self.get_current_goal_hints(goal)
        for _, row in self.hint_db.iterrows():
            hint = row["hint"]
            if hint in skip_hints:
                continue
            topic_to_hints[row["semantic_keys"]].append(hint)
        logger.info(f"Collected {len(topic_to_hints)} hint topics")
        hint_topics = list(topic_to_hints.keys())
        topics = "\n".join([f"{i}. {h}" for i, h in enumerate(hint_topics)])
        prompt = self.llm_prompt.format(goal=goal, topics=topics)

        if isinstance(llm, ChatModel):
            response: str = llm(messages=[dict(role="user", content=prompt)])["content"]
        else:
            response: str = llm(APIPayload(messages=[llm.msg.user().add_text(prompt)])).think
        try:
            matches = re.findall(r"<choice>(-?\d+)</choice>", response)
            if not matches:
                logger.error(f"No choice tags found in LLM response: {response}")
                return []
            if len(matches) > 1:
                logger.warning(
                    f"LLM selected multiple topics for retrieval using only the first one."
                )
            topic_number = int(matches[0])
            if topic_number < 0 or topic_number >= len(hint_topics):
                logger.error(f"Wrong LLM hint id response: {response}, no hints")
                return []
            hint_topic = hint_topics[topic_number]
            hints = list(set(topic_to_hints[hint_topic]))
            logger.info(f"LLM hint topic {topic_number}:'{hint_topic}', chosen hints: {hints}")
        except Exception as e:
            logger.exception(f"Failed to parse LLM hint id response: {response}:\n{e}")
            hints = []
        return hints

    def choose_hints_emb(self, goal: str, task_name: str) -> list[str]:
        """Choose hints using embeddings to filter the hints."""
        try:
            goal_embeddings = self._encode([goal], prompt="task description")
            hint_embeddings = self.hint_embeddings.copy()
            all_hints = self.uniq_hints["hint"].tolist()
            skip_hints = []
            if self.skip_hints_for_current_task:
                skip_hints += self.get_current_task_hints(task_name)
            if self.skip_hints_for_current_goal:
                skip_hints += self.get_current_goal_hints(goal)
            hint_embeddings = []
            id_to_hint = {}
            for hint, emb in zip(all_hints, self.hint_embeddings):
                if hint in skip_hints:
                    continue
                hint_embeddings.append(emb.tolist())
                id_to_hint[len(hint_embeddings) - 1] = hint
            logger.info(f"Prepared hint embeddings for {len(hint_embeddings)} hints")
            similarities = self._similarity(goal_embeddings.tolist(), hint_embeddings)
            top_indices = similarities.argsort()[0][-self.top_n :].tolist()
            logger.info(f"Top hint indices based on embedding similarity: {top_indices}")
            hints = [id_to_hint[idx] for idx in top_indices]
            logger.info(f"Embedding-based hints chosen: {hints}")
        except Exception as e:
            logger.exception(f"Failed to choose hints using embeddings: {e}")
            hints = []
        return hints

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
        raise ValueError("Failed to encode hints")

    def _similarity(
        self,
        texts1: list,
        texts2: list,
        timeout: int = 2,
        max_retries: int = 5,
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
        raise ValueError("Failed to compute similarity")

    def choose_hints_direct(self, task_name: str) -> list[str]:
        hints = self.get_current_task_hints(task_name)
        logger.info(f"Direct hints chosen: {hints}")
        return hints

    def get_current_task_hints(self, task_name):
        hints_df = self.hint_db[
            self.hint_db["task_name"].apply(lambda x: fnmatch.fnmatch(x, task_name))
        ]
        return hints_df["hint"].tolist()

    def get_current_goal_hints(self, goal_str: str):
        mask = self.hint_db["source_trace_goals"].apply(lambda goals: goal_str in goals)
        return self.hint_db.loc[mask, "hint"].tolist()
