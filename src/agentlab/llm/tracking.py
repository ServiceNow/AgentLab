import os
from contextlib import contextmanager
from typing import Any, List, Optional

import requests
from langchain.schema import AIMessage, BaseMessage
from openai import OpenAI

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


class OpenRouterChatModel:
    def __init__(
        self,
        model_name,
        openrouter_api_key=None,
        openrouter_api_base="https://openrouter.ai/api/v1",
        temperature=0.5,
        max_tokens=100,
    ):
        self.model_name = model_name
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_api_base = openrouter_api_base
        self.temperature = temperature
        self.max_tokens = max_tokens

        openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")

        # query api to get model metadata
        url = "https://openrouter.ai/api/v1/models"
        headers = {"Authorization": f"Bearer {openrouter_api_key}"}
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            raise ValueError("Failed to get model metadata")

        model_metadata = response.json()
        pricings = {model["id"]: model["pricing"] for model in model_metadata["data"]}

        self.input_cost = float(pricings[model_name]["prompt"])
        self.output_cost = float(pricings[model_name]["completion"])

        self.client = OpenAI(
            base_url=openrouter_api_base,
            api_key=openrouter_api_key,
        )

    def __call__(self, messages: List[BaseMessage]) -> str:
        messages_formated = _convert_messages_to_dict(messages)
        completion = self.client.chat.completions.create(
            model=self.model_name, messages=messages_formated
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
