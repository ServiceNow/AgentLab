import os
import time
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from anthropic.types import Usage as AnthropicUsage
from openai.types.completion_usage import CompletionUsage, PromptTokensDetails
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)

import agentlab.llm.tracking as tracking
from agentlab.llm.chat_api import (
    AzureChatModel,
    OpenAIChatModel,
    OpenRouterChatModel,
    make_system_message,
    make_user_message,
)
from agentlab.llm.response_api import (
    ClaudeResponseModelArgs,
    OpenAIChatModelArgs,
    OpenAIResponseModelArgs,
)


def test_get_action_decorator():
    action, agent_info = tracking.cost_tracker_decorator(lambda x, y: call_llm())(None, None)
    assert action == "action"
    assert agent_info["stats"] == {
        "input_tokens": 1,
        "output_tokens": 1,
        "cost": 1.0,
    }


OPENROUTER_API_KEY_AVAILABLE = os.environ.get("OPENROUTER_API_KEY") is not None

OPENROUTER_MODELS = (
    "anthropic/claude-3.5-sonnet",
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemini-pro-1.5",
)


@pytest.mark.skipif(not OPENROUTER_API_KEY_AVAILABLE, reason="OpenRouter API key is not available")
def test_get_pricing_openrouter():
    pricing = tracking.get_pricing_openrouter()
    assert isinstance(pricing, dict)
    assert all(isinstance(v, dict) for v in pricing.values())
    for model in OPENROUTER_MODELS:
        assert model in pricing
        assert isinstance(pricing[model], dict)
        assert all(isinstance(v, float) for v in pricing[model].values())


def test_get_pricing_openai():
    pricing = tracking.get_pricing_openai()
    assert isinstance(pricing, dict)
    assert all("prompt" in pricing[model] and "completion" in pricing[model] for model in pricing)
    assert all(isinstance(pricing[model]["prompt"], float) for model in pricing)
    assert all(isinstance(pricing[model]["completion"], float) for model in pricing)


# List prices in USD per 1M tokens, copied by hand from the providers' pricing pages
# (https://platform.openai.com/docs/pricing, https://docs.anthropic.com/en/docs/about-claude/pricing)
# so that the effective costs below are checked against the published numbers.
GPT_5 = {"input": 1.25, "cached_input": 0.125, "output": 10.0}
CLAUDE_OPUS_5_5 = {"input": 4.0, "cache_read": 0.20, "cache_write_5m": 5.0, "output": 20.0}
CLAUDE_SONNET_4_6 = {
    "input": 3.0,
    "cache_read": 0.30,
    "cache_write_5m": 3.75,
    "cache_write_1h": 6.0,
    "output": 15.0,
}


def _make_openai_model(args_cls, model_name):
    with patch("agentlab.llm.response_api.OpenAI"):
        return args_cls(model_name=model_name).make_model()


def _make_claude_model(model_name):
    with patch("agentlab.llm.response_api.Anthropic"):
        return ClaudeResponseModelArgs(model_name=model_name).make_model()


def test_openai_chat_completion_effective_cost_gpt5_cached_tokens():
    model = _make_openai_model(OpenAIChatModelArgs, "gpt-5")
    assert model.input_cost == pytest.approx(GPT_5["input"] / 1e6)
    assert model.output_cost == pytest.approx(GPT_5["output"] / 1e6)

    # 100k prompt tokens, 80k of them served from the prompt cache
    usage = CompletionUsage(
        prompt_tokens=100_000,
        completion_tokens=2_000,
        total_tokens=102_000,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=80_000),
    )
    response = SimpleNamespace(usage=usage)
    expected = (
        20_000 * GPT_5["input"] + 80_000 * GPT_5["cached_input"] + 2_000 * GPT_5["output"]
    ) / 1e6
    assert model.get_effective_cost(response) == pytest.approx(expected)  # $0.055
    # the fixed 0.5 cache factor priced the same call at $0.095
    assert expected == pytest.approx(0.055)


def test_openai_responses_api_effective_cost_gpt5_cached_tokens():
    model = _make_openai_model(OpenAIResponseModelArgs, "gpt-5")
    usage = ResponseUsage(
        input_tokens=100_000,
        input_tokens_details=InputTokensDetails(cached_tokens=80_000),
        output_tokens=2_000,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=1_500),
        total_tokens=102_000,
    )
    response = SimpleNamespace(usage=usage)
    expected = (
        20_000 * GPT_5["input"] + 80_000 * GPT_5["cached_input"] + 2_000 * GPT_5["output"]
    ) / 1e6
    assert model.get_effective_cost(response) == pytest.approx(expected)


def test_anthropic_effective_cost_claude_opus_5_5_cache():
    model = _make_claude_model("claude-opus-5-5")
    assert model.input_cost == pytest.approx(CLAUDE_OPUS_5_5["input"] / 1e6)
    assert model.output_cost == pytest.approx(CLAUDE_OPUS_5_5["output"] / 1e6)

    usage = AnthropicUsage(
        input_tokens=1_000,
        cache_read_input_tokens=50_000,
        cache_creation_input_tokens=4_000,
        output_tokens=1_500,
    )
    response = SimpleNamespace(usage=usage)
    expected = (
        1_000 * CLAUDE_OPUS_5_5["input"]
        + 50_000 * CLAUDE_OPUS_5_5["cache_read"]
        + 4_000 * CLAUDE_OPUS_5_5["cache_write_5m"]
        + 1_500 * CLAUDE_OPUS_5_5["output"]
    ) / 1e6
    assert model.get_effective_cost(response) == pytest.approx(expected)  # $0.064


def test_anthropic_effective_cost_one_hour_cache_writes():
    model = _make_claude_model("claude-sonnet-4-6")
    # newer anthropic SDKs split cache writes by lifetime
    usage = SimpleNamespace(
        input_tokens=500,
        cache_read_input_tokens=10_000,
        cache_creation_input_tokens=3_000,
        cache_creation=SimpleNamespace(
            ephemeral_5m_input_tokens=1_000, ephemeral_1h_input_tokens=2_000
        ),
        output_tokens=800,
    )
    expected = (
        500 * CLAUDE_SONNET_4_6["input"]
        + 10_000 * CLAUDE_SONNET_4_6["cache_read"]
        + 1_000 * CLAUDE_SONNET_4_6["cache_write_5m"]
        + 2_000 * CLAUDE_SONNET_4_6["cache_write_1h"]
        + 800 * CLAUDE_SONNET_4_6["output"]
    ) / 1e6
    assert model.get_effective_cost(SimpleNamespace(usage=usage)) == pytest.approx(expected)


def test_effective_cost_falls_back_for_models_without_llm_prices_row():
    with patch("agentlab.llm.response_api.OpenAI"):
        model = OpenAIChatModelArgs(model_name="my-org/my-finetuned-model").make_model()
    assert model._llm_price is None
    model.input_cost, model.output_cost = 1e-6, 2e-6
    usage = CompletionUsage(
        prompt_tokens=1_000,
        completion_tokens=100,
        total_tokens=1_100,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=400),
    )
    factor = tracking.OPENAI_CACHE_PRICING_FACTOR["cache_read_tokens"]
    expected = 600 * 1e-6 + 400 * 1e-6 * factor + 100 * 2e-6
    assert model.get_effective_cost(SimpleNamespace(usage=usage)) == pytest.approx(expected)


def call_llm():
    if hasattr(tracking.TRACKER, "instance") and isinstance(
        tracking.TRACKER.instance, tracking.LLMTracker
    ):
        tracking.TRACKER.instance(1, 1, 1)
    return "action", {"stats": {}}


def test_tracker():
    with tracking.set_tracker() as tracker:
        _, _ = call_llm()

    assert tracker.stats["cost"] == 1


def test_imbricate_trackers():
    with tracking.set_tracker() as tracker4:
        with tracking.set_tracker() as tracker1:
            _, _ = call_llm()
        with tracking.set_tracker() as tracker3:
            _, _ = call_llm()
            _, _ = call_llm()
            with tracking.set_tracker() as tracker1bis:
                _, _ = call_llm()

    assert tracker1.stats["cost"] == 1
    assert tracker1bis.stats["cost"] == 1
    assert tracker3.stats["cost"] == 3
    assert tracker4.stats["cost"] == 4


def test_threaded_trackers():
    """thread_2 occurs in the middle of thread_1, results should be separate."""
    import threading

    def thread_1(results=None):
        with tracking.set_tracker() as tracker:
            time.sleep(1)
            _, _ = call_llm()
            time.sleep(1)
        results[0] = tracker.stats

    def thread_2(results=None):
        time.sleep(1)
        with tracking.set_tracker() as tracker:
            _, _ = call_llm()
        results[1] = tracker.stats

    results = [None] * 2
    threads = [
        threading.Thread(target=partial(thread_1, results=results)),
        threading.Thread(target=partial(thread_2, results=results)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(result["cost"] == 1 for result in results)


OPENAI_API_KEY_AVAILABLE = os.environ.get("OPENAI_API_KEY") is not None


@pytest.mark.pricy
@pytest.mark.skipif(not OPENAI_API_KEY_AVAILABLE, reason="OpenAI API key is not available")
def test_openai_chat_model():
    chat_model = OpenAIChatModel("gpt-4o-mini")
    assert chat_model.input_cost > 0
    assert chat_model.output_cost > 0

    messages = [
        make_system_message("You are an helpful virtual assistant"),
        make_user_message("Give the third prime number"),
    ]
    with tracking.set_tracker() as tracker:
        answer = chat_model(messages)
    assert "5" in answer.get("content")
    assert tracker.stats["cost"] > 0


AZURE_OPENAI_API_KEY_AVAILABLE = (
    os.environ.get("AZURE_OPENAI_API_KEY") is not None
    and os.environ.get("AZURE_OPENAI_ENDPOINT") is not None
)


@pytest.mark.pricy
@pytest.mark.skipif(
    not AZURE_OPENAI_API_KEY_AVAILABLE, reason="Azure OpenAI API key is not available"
)
def test_azure_chat_model():
    chat_model = AzureChatModel(model_name="gpt-4.1-nano", deployment_name="gpt-4.1-nano")
    assert chat_model.input_cost > 0
    assert chat_model.output_cost > 0

    messages = [
        make_system_message("You are an helpful virtual assistant"),
        make_user_message("Give the third prime number"),
    ]
    with tracking.set_tracker() as tracker:
        answer = chat_model(messages)
    assert "5" in answer.get("content")
    assert tracker.stats["cost"] > 0


@pytest.mark.pricy
@pytest.mark.skipif(not OPENROUTER_API_KEY_AVAILABLE, reason="OpenRouter API key is not available")
def test_openrouter_chat_model():
    chat_model = OpenRouterChatModel("openai/gpt-4o-mini")
    assert chat_model.input_cost > 0
    assert chat_model.output_cost > 0

    messages = [
        make_system_message("You are an helpful virtual assistant"),
        make_user_message("Give the third prime number"),
    ]
    with tracking.set_tracker() as tracker:
        answer = chat_model(messages)
    assert "5" in answer.get("content")
    assert tracker.stats["cost"] > 0
