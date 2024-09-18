import ast
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, List, Optional

import requests
from langchain.schema import AIMessage, BaseMessage
from openai import AzureOpenAI, OpenAI

from agentlab.llm.langchain_utils import _convert_messages_to_dict


class LLMTracker:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0

    def __call__(self, input_tokens: int, output_tokens: int, cost: float):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost += cost

    @property
    def stats(self):
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
        }


@contextmanager
def set_tracker(tracker: LLMTracker):
    global current_tracker
    previous_tracker = globals().get("current_tracker", None)
    current_tracker = tracker
    yield
    current_tracker = previous_tracker


def get_action_decorator(get_action):
    def wrapper(self, obs):
        tracker = LLMTracker()
        with set_tracker(tracker):
            action, agent_info = get_action(self, obs)
        agent_info.get("stats").update(tracker.stats)
        return action, agent_info

    return wrapper


def get_pricing(api: str = "openrouter", api_key: str = None):
    if api == "openrouter":
        assert api_key, "OpenRouter API key is required"
        # query api to get model metadata
        url = "https://openrouter.ai/api/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            raise ValueError("Failed to get model metadata")

        model_metadata = response.json()
        return {
            model["id"]: {k: float(v) for k, v in model["pricing"].items()}
            for model in model_metadata["data"]
        }
    elif api == "openai":
        url = "https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/community/langchain_community/callbacks/openai_info.py"
        response = requests.get(url)

        if response.status_code == 200:
            content = response.text
            tree = ast.parse(content)
            cost_dict = None
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    if node.targets[0].id == "MODEL_COST_PER_1K_TOKENS":
                        cost_dict = ast.literal_eval(node.value)
                        break
            if cost_dict:
                cost_dict = {k: v / 1000 for k, v in cost_dict.items()}
                res = {}
                for k in cost_dict:
                    if k.endswith("-completion"):
                        continue
                    prompt_key = k
                    completion_key = k + "-completion"
                    if completion_key in cost_dict:
                        res[k] = {
                            "prompt": cost_dict[prompt_key],
                            "completion": cost_dict[completion_key],
                        }
                return res
            else:
                raise ValueError("Cost dictionary not found.")
        else:
            raise ValueError(f"Failed to retrieve the file. Status code: {response.status_code}")


class ChatModel(ABC):

    @abstractmethod
    def __init__(self, model_name, api_key=None, temperature=0.5, max_tokens=100):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = OpenAI()

        self.input_cost = 0.0
        self.output_cost = 0.0

    def __call__(self, messages: List[BaseMessage]) -> str:
        messages_formated = _convert_messages_to_dict(messages)
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages_formated,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = input_tokens * self.input_cost + output_tokens * self.output_cost

        global current_tracker
        if "current_tracker" in globals() and isinstance(current_tracker, LLMTracker):
            current_tracker(input_tokens, output_tokens, cost)

        return AIMessage(content=completion.choices[0].message.content)

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        return self(messages)


class OpenRouterChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        pricings = get_pricing(api="openrouter", api_key=api_key)

        self.input_cost = pricings[model_name]["prompt"]
        self.output_cost = pricings[model_name]["completion"]

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )


class OpenAIChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = api_key or os.getenv("OPENAI_API_KEY")

        pricings = get_pricing(api="openai")

        self.input_cost = float(pricings[model_name]["prompt"])
        self.output_cost = float(pricings[model_name]["completion"])

        self.client = OpenAI(
            api_key=api_key,
        )


class AzureChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        endpoint=None,
        temperature=0.5,
        max_tokens=100,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = api_key or os.getenv("OPENAI_API_KEY")

        pricings = get_pricing(api="openai")

        self.input_cost = float(pricings[model_name]["prompt"])
        self.output_cost = float(pricings[model_name]["completion"])

        self.client = AzureOpenAI(
            api_key=api_key, azure_endpoint=endpoint, api_version="2024-02-01"
        )
