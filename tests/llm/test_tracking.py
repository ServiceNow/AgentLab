import os
import time
from functools import partial

import pytest

import agentlab.llm.tracking as tracking


def test_get_action_decorator():
    action, agent_info = tracking.get_action_decorator(lambda x, y: call_llm())(None, None)
    assert action == "action"
    assert agent_info["stats"] == {
        "input_tokens": 1,
        "output_tokens": 1,
        "cost": 1.0,
    }


OPENROUTER_API_KEY_AVAILABLE = os.environ.get("OPENROUTER_API_KEY") is not None

OPENROUTER_MODELS = (
    "openai/o1-mini-2024-09-12",
    "openai/o1-preview-2024-09-12",
    "openai/gpt-4o-2024-08-06",
    "openai/gpt-4o-2024-05-13",
    "anthropic/claude-3.5-sonnet:beta",
    "anthropic/claude-3.5-sonnet",
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemini-pro-1.5",
    "qwen/qwen-2-vl-72b-instruct",
)


@pytest.mark.skipif(not OPENROUTER_API_KEY_AVAILABLE, reason="OpenRouter API key is not available")
def test_get_pricing_openrouter():
    pricing = tracking.get_pricing(api="openrouter", api_key=os.environ["OPENROUTER_API_KEY"])
    assert isinstance(pricing, dict)
    assert all(isinstance(v, dict) for v in pricing.values())
    for model in OPENROUTER_MODELS:
        assert model in pricing
        assert isinstance(pricing[model], dict)
        assert all(isinstance(v, float) for v in pricing[model].values())


def test_get_pricing_openai():
    pricing = tracking.get_pricing(api="openai")
    assert isinstance(pricing, dict)
    assert all("prompt" in pricing[model] and "completion" in pricing[model] for model in pricing)
    assert all(isinstance(pricing[model]["prompt"], float) for model in pricing)
    assert all(isinstance(pricing[model]["completion"], float) for model in pricing)


def call_llm():
    if isinstance(tracking.TRACKER.instance, tracking.LLMTracker):
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
