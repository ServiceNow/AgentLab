import importlib
import logging
import os
import re
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cache, partial
from typing import Optional

import llm_prices
import requests

langchain_community = importlib.util.find_spec("langchain_community")
if langchain_community is not None:
    from langchain_community.callbacks import bedrock_anthropic_callback, openai_info
else:
    bedrock_anthropic_callback = None
    openai_info = None
from litellm import completion_cost, get_model_info

TRACKER = threading.local()

# Fallback cache factors, only used for models that llm-prices has no row for. Models in the
# llm-prices table are priced with their own listed cache read and cache write prices.
ANTHROPIC_CACHE_PRICING_FACTOR = {
    "cache_read_tokens": 0.1,  # Cost for 5 min ephemeral cache. See Pricing Here: https://docs.anthropic.com/en/docs/about-claude/pricing#model-pricing
    "cache_write_tokens": 1.25,
}

OPENAI_CACHE_PRICING_FACTOR = {
    "cache_read_tokens": 0.5,  # This is a an upper bound. See Pricing Here: https://platform.openai.com/docs/pricing
    "cache_write_tokens": 1,
}


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
    try:
        cost_dict = openai_info.MODEL_COST_PER_1K_TOKENS
    except Exception as e:
        logging.warning(
            f"Failed to get OpenAI pricing: {e}. "
            "Please install langchain-community or use LiteLLM API for pricing information."
        )
        return {}
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
    try:
        input_cost_dict = bedrock_anthropic_callback.MODEL_COST_PER_1K_INPUT_TOKENS
        output_cost_dict = bedrock_anthropic_callback.MODEL_COST_PER_1K_OUTPUT_TOKENS
    except Exception as e:
        logging.warning(
            f"Failed to get Anthropic pricing: {e}. "
            "Please install langchain-community or use LiteLLM API for pricing information."
        )
        return {}

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


def _token_count(obj, name: str) -> int:
    """Returns the integer token count `obj.name`, or 0 when it is missing or not an int."""
    value = getattr(obj, name, 0)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def get_pricing_litellm(model_name):
    """Returns a dictionary of model pricing for a LiteLLM model."""
    try:
        info = get_model_info(model_name)
    except Exception as e:
        logging.error(f"Error fetching model info for {model_name}: {e} from litellm")
        info = {}
    return {
        model_name: {
            "prompt": info.get("input_cost_per_token", 0.0),
            "completion": info.get("output_cost_per_token", 0.0),
        }
    }


class TrackAPIPricingMixin:
    """Mixin class to handle pricing information for different models.
    This populates the tracker.stats used by the cost_tracker_decorator

    Usage: provide the pricing_api to use in the constructor.
    """

    def reset_stats(self):
        self.stats = Stats()

    def init_pricing_tracker(self, pricing_api=None):
        """Initialize the pricing tracker with the given API."""
        self._pricing_api = pricing_api
        self.set_pricing_attributes()
        self.reset_stats()

    def __call__(self, *args, **kwargs):
        """Call the API and update the pricing tracker."""
        # 'self' here calls ._call_api() method of the subclass
        response = self._call_api(*args, **kwargs)
        usage = dict(getattr(response, "usage", {}))
        if "prompt_tokens_details" in usage and usage["prompt_tokens_details"]:
            usage["cached_tokens"] = usage["prompt_tokens_details"].cached_tokens
        if "input_tokens_details" in usage and usage["input_tokens_details"]:
            usage["cached_tokens"] = usage["input_tokens_details"].cached_tokens
        usage = {f"usage_{k}": v for k, v in usage.items() if isinstance(v, (int, float))}
        usage |= {"n_api_calls": 1}
        usage |= {"effective_cost": self.get_effective_cost(response)}
        self.stats.increment_stats_dict(usage)
        self.update_pricing_tracker(response)
        return self._parse_response(response)

    def fetch_pricing_information_from_provider(self) -> Optional[dict]:
        """
        Fetch the pricing information dictionary for the given provider.

        Returns:
            Optional[dict]: A dict mapping model names to pricing info, or None if not found.
        """
        pricing_fn_map = {
            "openai": get_pricing_openai,
            "anthropic": get_pricing_anthropic,
            "openrouter": get_pricing_openrouter,
            "litellm": partial(get_pricing_litellm, self.model_name),
        }
        pricing_fn = pricing_fn_map.get(self._pricing_api, None)
        if pricing_fn is None:
            logging.warning(
                f"Unsupported provider: {self._pricing_api}. Supported providers are: {list(pricing_fn_map.keys())}"
            )
            return None
        return pricing_fn()

    def set_pricing_attributes(self) -> None:
        """Set the pricing attributes for the model based on the provider.

        OpenAI and Anthropic models are priced from the llm-prices table first, which also
        gives the model's own cache read and cache write prices for the effective cost. Models
        it has no row for (and every other provider) use the provider pricing below.
        """
        self._llm_price = None
        if self._pricing_api in ("openai", "anthropic"):
            price = llm_prices.lookup_provider(self._pricing_api, self.model_name)
            if price is not None and price.input is not None and price.output is not None:
                self._llm_price = price
                self.input_cost = price.input_per_token
                self.output_cost = price.output_per_token
                return

        model_to_price_dict = self.fetch_pricing_information_from_provider()
        model_costs = model_to_price_dict.get(self.model_name) if model_to_price_dict else None
        if model_costs:
            self.input_cost = float(model_costs["prompt"])
            self.output_cost = float(model_costs["completion"])
        else:
            # use litellm to get model info if not found in the pricing dict
            try:
                model_info = get_model_info(self.model_name)
                self.input_cost = float(model_info.get("input_cost_per_token", 0.0))
                self.output_cost = float(model_info.get("output_cost_per_token", 0.0))
            except Exception as e:
                logging.warning(f"Failed to fetch pricing for {self.model_name}: {e}")
                self.input_cost = 0.0
                self.output_cost = 0.0

    def update_pricing_tracker(self, raw_response) -> None:
        """Update the pricing tracker with the input and output tokens and cost."""

        input_tokens, output_tokens = self.get_tokens_counts_from_response(raw_response)
        cost = input_tokens * self.input_cost + output_tokens * self.output_cost

        if hasattr(TRACKER, "instance") and isinstance(TRACKER.instance, LLMTracker):
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

    def get_effective_cost(self, response):
        """Get the effective cost from the response based on the provider."""
        if self._pricing_api == "anthropic":
            return self.get_effective_cost_from_antrophic_api(response)
        elif self._pricing_api == "openai":
            return self.get_effective_cost_from_openai_api(response)
        elif self._pricing_api == "litellm":
            return completion_cost(response)
        else:
            logging.warning(
                f"Unsupported provider: {self._pricing_api}. No effective cost calculated."
            )
            return 0.0

    def _get_effective_cost_from_llm_prices(
        self, new_input_tokens, output_tokens, cache_read_tokens, cache_write_5m, cache_write_1h
    ) -> Optional[float]:
        """Price one response with the model's llm-prices row.

        Cache reads and cache writes are billed at the model's listed prices, and the
        long-context prices apply above the model's threshold.

        Args:
            new_input_tokens: Uncached input tokens.
            output_tokens: Output tokens, reasoning tokens included.
            cache_read_tokens: Input tokens served from the prompt cache.
            cache_write_5m: Input tokens written to the cache with a 5-minute lifetime.
            cache_write_1h: Input tokens written to the cache with a 1-hour lifetime.

        Returns:
            Optional[float]: The cost in USD, or None when the model has no llm-prices row or
                a price the usage needs is not listed.
        """
        price = getattr(self, "_llm_price", None)
        counts = (
            new_input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_write_5m,
            cache_write_1h,
        )
        if price is None or not all(isinstance(c, int) and not isinstance(c, bool) for c in counts):
            return None
        usage = llm_prices.Usage(
            input=max(new_input_tokens, 0),
            output=max(output_tokens, 0),
            cache_read=max(cache_read_tokens, 0),
            cache_write_5m=max(cache_write_5m, 0),
            cache_write_1h=max(cache_write_1h, 0),
        )
        cost = price.cost_usage(usage)
        return None if cost is None else cost.total

    def get_effective_cost_from_antrophic_api(self, response) -> float:
        """
        Get the effective cost from the Anthropic API response.

        Anthropic usage 'input_tokens' are new input tokens (tokens that are not cached).
        Anthropic has different pricing for cache write and cache read tokens.
        See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#tracking-cache-performance

        Args:
            response: The response object from the Anthropic API.

        Returns:
            float: The effective cost calculated from the response.
        """
        usage = getattr(response, "usage", {})
        new_input_tokens = _token_count(usage, "input_tokens")  # new input tokens
        output_tokens = _token_count(usage, "output_tokens")
        cache_read_tokens = _token_count(usage, "cache_read_input_tokens")
        cache_write_tokens = _token_count(usage, "cache_creation_input_tokens")
        # Newer SDKs split cache writes by lifetime; the total alone means 5-minute writes.
        cache_creation = getattr(usage, "cache_creation", None)
        cache_write_1h = min(
            _token_count(cache_creation, "ephemeral_1h_input_tokens"), cache_write_tokens
        )
        cache_write_5m = cache_write_tokens - cache_write_1h

        effective_cost = self._get_effective_cost_from_llm_prices(
            new_input_tokens, output_tokens, cache_read_tokens, cache_write_5m, cache_write_1h
        )
        if effective_cost is not None:
            return effective_cost

        cache_read_cost = self.input_cost * ANTHROPIC_CACHE_PRICING_FACTOR["cache_read_tokens"]
        cache_write_cost = self.input_cost * ANTHROPIC_CACHE_PRICING_FACTOR["cache_write_tokens"]

        # Calculate the effective cost
        effective_cost = (
            new_input_tokens * self.input_cost
            + output_tokens * self.output_cost
            + cache_read_tokens * cache_read_cost
            + cache_write_tokens * cache_write_cost
        )
        if effective_cost < 0:
            logging.warning(
                "Anthropic: Negative effective cost detected.(Impossible! Likely a bug)"
            )
        return effective_cost

    def get_effective_cost_from_openai_api(self, response) -> float:
        """
        Get the effective cost from the OpenAI API response.

        OpenAI usage 'prompt_tokens' are the total input tokens (cache read tokens + new input tokens).
        See https://openai.com/index/api-prompt-caching/
        OpenAI has only one price for cache tokens, i.e., cache read price (generally 50% cheaper).
        OpenAI has no extra charge for cache write tokens.
        See Pricing Here: https://platform.openai.com/docs/pricing

        Args:
            response: The response object from the OpenAI API.

        Returns:
            float: The effective cost calculated from the response.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            logging.warning("No usage information found in the response. Defaulting cost to 0.0.")
            return 0.0
        api_type = "chatcompletion" if hasattr(usage, "prompt_tokens_details") else "response"
        if api_type == "chatcompletion":
            total_input_tokens = usage.prompt_tokens  # (cache read tokens + new input tokens)
            output_tokens = usage.completion_tokens
            cached_input_tokens = (
                usage.prompt_tokens_details.cached_tokens if usage.prompt_tokens_details else 0
            )
            new_input_tokens = total_input_tokens - cached_input_tokens
        elif api_type == "response":
            total_input_tokens = usage.input_tokens  # (cache read tokens + new input tokens)
            output_tokens = usage.output_tokens
            cached_input_tokens = (
                usage.input_tokens_details.cached_tokens if usage.input_tokens_details else 0
            )
            new_input_tokens = total_input_tokens - cached_input_tokens
        else:
            logging.warning(f"Unsupported API type: {api_type}. Defaulting cost to 0.0.")
            return 0.0
        effective_cost = self._get_effective_cost_from_llm_prices(
            new_input_tokens, output_tokens, cached_input_tokens, 0, 0
        )
        if effective_cost is not None:
            return effective_cost
        cache_read_cost = self.input_cost * OPENAI_CACHE_PRICING_FACTOR["cache_read_tokens"]
        effective_cost = (
            self.input_cost * new_input_tokens
            + cached_input_tokens * cache_read_cost
            + self.output_cost * output_tokens
        )
        if effective_cost < 0:
            logging.warning(
                f"OpenAI: Negative effective cost detected.(Impossible! Likely a bug). "
                f"New input tokens: {total_input_tokens}"
            )
        return effective_cost


@dataclass
class Stats:
    stats_dict: dict = field(default_factory=lambda: defaultdict(float))

    def increment_stats_dict(self, stats_dict: dict):
        """increment the stats_dict with the given values."""
        for k, v in stats_dict.items():
            self.stats_dict[k] += v
