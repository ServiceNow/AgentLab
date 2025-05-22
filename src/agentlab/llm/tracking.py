import os
import re
import threading
from contextlib import contextmanager
from functools import cache

import requests
from langchain_community.callbacks import bedrock_anthropic_callback, openai_info
from typing import Optional
import logging

TRACKER = threading.local()


class LLMTracker:
    def __init__(self, suffix=""):
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0
        self.input_tokens_key = "input_tokens_" + suffix if suffix else "input_tokens"
        self.output_tokens_key = "output_tokens_" + suffix if suffix else "output_tokens"
        self.cost_key = "cost_" + suffix if suffix else "cost"

    def __call__(self, input_tokens: int, output_tokens: int, cost: float):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost += cost

    @property
    def stats(self):
        return {
            self.input_tokens_key: self.input_tokens,
            self.output_tokens_key: self.output_tokens,
            self.cost_key: self.cost,
        }

    def add_tracker(self, tracker: "LLMTracker"):
        self(tracker.input_tokens, tracker.output_tokens, tracker.cost)

    def __repr__(self):
        return f"LLMTracker(input_tokens={self.input_tokens}, output_tokens={self.output_tokens}, cost={self.cost})"


@contextmanager
def set_tracker(suffix=""):
    global TRACKER
    if not hasattr(TRACKER, "instance"):
        TRACKER.instance = None
    previous_tracker = TRACKER.instance  # type: LLMTracker
    TRACKER.instance = LLMTracker(suffix)
    try:
        yield TRACKER.instance
    finally:
        # If there was a previous tracker, add the current one to it
        if isinstance(previous_tracker, LLMTracker):
            previous_tracker.add_tracker(TRACKER.instance)
        # Restore the previous tracker
        TRACKER.instance = previous_tracker


def cost_tracker_decorator(get_action, suffix=""):
    def wrapper(self, obs):
        with set_tracker(suffix) as tracker:
            action, agent_info = get_action(self, obs)
        agent_info.get("stats").update(tracker.stats)
        return action, agent_info

    return wrapper


@cache
def get_pricing_openrouter():
    """Returns a dictionary of model pricing for OpenRouter models."""
    api_key = os.getenv("OPENROUTER_API_KEY")
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


def get_pricing_openai():
    """Returns a dictionary of model pricing for OpenAI models."""
    cost_dict = openai_info.MODEL_COST_PER_1K_TOKENS
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


def _remove_version_suffix(model_name):
    no_version = re.sub(r"-v\d+(?:[.:]\d+)?$", "", model_name)
    return re.sub(r"anthropic.", "", no_version)


def get_pricing_anthropic():
    """Returns a dictionary of model pricing for Anthropic models."""
    input_cost_dict = bedrock_anthropic_callback.MODEL_COST_PER_1K_INPUT_TOKENS
    output_cost_dict = bedrock_anthropic_callback.MODEL_COST_PER_1K_OUTPUT_TOKENS

    res = {}
    for k, v in input_cost_dict.items():
        k = _remove_version_suffix(k)
        res[k] = {"prompt": v / 1000}

    for k, v in output_cost_dict.items():
        k = _remove_version_suffix(k)
        if k not in res:
            res[k] = {}
        res[k]["completion"] = v / 1000
    return res


class TrackAPIPricingMixin:
    """Mixin class to handle pricing information for different models.
    This populates the tracker.stats used by the cost_tracker_decorator

    Usage: provide the pricing_api to use in the constructor.
    """

    def __init__(self, *args, **kwargs):
        pricing_api = kwargs.pop("pricing_api", None)
        self._pricing_api = pricing_api
        super().__init__(*args, **kwargs)
        self.set_pricing_attributes()

    def __call__(self, *args, **kwargs):
        """Call the API and update the pricing tracker."""
        response = self._call_api(*args, **kwargs)
        self.update_pricing_tracker(response)
        return self._parse_response(response)

    def fetch_pricing_information_from_provider(self) -> Optional[dict]:
        """
        Fetch the pricing information dictionary for the given provider.
        Returns a dict mapping model names to pricing info, or None if not found.
        """
        pricing_fn_map = {
            "openai": get_pricing_openai,
            "anthropic": get_pricing_anthropic,
            "openrouter": get_pricing_openrouter,
        }
        pricing_fn = pricing_fn_map.get(self._pricing_api, None)
        if pricing_fn is None:
            logging.warning(
                f"Unsupported provider: {self._pricing_api}. Supported providers are: {list(pricing_fn_map.keys())}"
            )
            return None
        return pricing_fn()

    def set_pricing_attributes(self) -> None:
        """Set the pricing attributes for the model based on the provider."""
        model_to_price_dict = self.fetch_pricing_information_from_provider()
        model_costs = model_to_price_dict.get(self.model_name) if model_to_price_dict else None
        if model_costs:
            self.input_cost = float(model_costs["prompt"])
            self.output_cost = float(model_costs["completion"])
        else:
            logging.warning(f"Model {self.model_name} not found in the pricing information.")
            self.input_cost = 0.0
            self.output_cost = 0.0

    def update_pricing_tracker(self, raw_response) -> None:
        """Update the pricing tracker with the input and output tokens and cost."""

        input_tokens, output_tokens = self.get_tokens_counts_from_response(raw_response)
        cost = input_tokens * self.input_cost + output_tokens * self.output_cost

        if hasattr(TRACKER, "instance") and isinstance(
            TRACKER.instance, LLMTracker
        ):
            TRACKER.instance(input_tokens, output_tokens, cost)

    def get_tokens_counts_from_response(self, response) -> tuple:
        """Get the input and output tokens counts from the response, provider-agnostic."""
        # Try OpenAI/Anthropic style
        usage = getattr(response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "input_tokens", None) or getattr(
                usage, "prompt_tokens", None
            )
            output_tokens = getattr(usage, "output_tokens", None) or getattr(
                usage, "completion_tokens", None
            )
            if input_tokens is not None and output_tokens is not None:
                return input_tokens, output_tokens

        # Try dict style
        if isinstance(response, dict) and "usage" in response:
            usage = response["usage"]
            input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
            output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
            if input_tokens is not None and output_tokens is not None:
                return input_tokens, output_tokens

        logging.warning(
            "Unable to extract input and output tokens from the response. Defaulting to 0."
        )
        return 0, 0
