import logging
import re
from copy import copy
from dataclasses import dataclass
from functools import partial

import bgym
from browsergym.experiments import Agent

import agentlab.agents.dynamic_prompting as dp
from agentlab.agents.generic_agent.generic_agent import GenericAgent, GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.llm.chat_api import ChatModelArgs

from .embedding_retriever import OpenAIRetriever, OpenAIRetrieverArgs
from .utils import get_chunks_from_tokenizer


@dataclass
class EmbeddingRetrieverAgentArgs(GenericAgentArgs):
    flags: GenericPromptFlags = None
    chat_model_args: ChatModelArgs = None
    retriever_args: OpenAIRetrieverArgs = None
    max_retry: int = 4
    agent_name: str = None

    def __post_init__(self):
        try:  # some attributes might be temporarily args.CrossProd for hyperparameter generation
            if (
                self.agent_name == None
            ):  # some attributes might be temporarily args.CrossProd for hyperparameter generation
                self.agent_name = (
                    f"EmbeddingRetrieverAgent-{self.chat_model_args.model_name}".replace("/", "_")
                )
        except AttributeError:
            pass

    def make_agent(self) -> Agent:
        return EmbeddingRetrieverAgent(
            self.chat_model_args,
            self.flags,
            self.retriever_args,
            self.max_retry,
        )


class EmbeddingRetrieverAgent(GenericAgent):
    def __init__(
        self,
        chat_model_args: ChatModelArgs,
        flags: GenericPromptFlags,
        retriever_args: OpenAIRetrieverArgs,
        max_retry: int = 4,
    ):
        super().__init__(chat_model_args, flags, max_retry)
        self.retriever = retriever_args.make_model()

    def get_new_obs(self, obs: dict) -> dict:
        query = obs["goal"] + "\n" + obs["history"]
        axtree_txt = obs["axtree_txt"] if self.flags.obs.use_ax_tree else obs["pruned_dom"]
        axtree_chunks = []
        if self.retriever.args.use_recursive_text_splitter:
            try:
                from langchain.text_splitter import (
                    RecursiveCharacterTextSplitter,
                )
            except ImportError:
                raise ImportError(
                    "langchain is not installed. Please install it using `pip agentlab[retrievers]`."
                )

            text_splitter = RecursiveCharacterTextSplitter()
            axtree_chunks = text_splitter.split_text(axtree_txt)
        else:
            axtree_chunks = get_chunks_from_tokenizer(
                axtree_txt, self.retriever.args.chunk_size, self.retriever.args.overlap
            )

        scores, indices = self.retriever.retrieve(query, axtree_chunks)

        new_tree = ""
        for i, index in enumerate(indices.tolist()):
            new_tree += f"\n\nChunk {i}:\n{axtree_chunks[index]}"

        return new_tree

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
        obs["axtree_txt"] = self.get_new_obs(obs)
        action, info = super().get_action(obs)
        info.extra_info["pruned_tree"] = obs["axtree_txt"]
        info.extra_info["retriever_query"] = obs["goal"] + "\n" + obs["history"]
        return action, info
