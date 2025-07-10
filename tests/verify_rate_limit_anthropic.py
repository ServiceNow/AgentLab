import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def make_request(messages):
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022", max_tokens=10, messages=messages
    )
    return response.usage


def make_message(text):
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
    }


def add_cache_control(message: dict, cache_type="ephemeral"):
    message["content"][0]["cache_control"] = {"type": cache_type}


def remove_cache_control(message: dict):
    if "cache_control" in message["content"][0]:
        del message["content"][0]["cache_control"]


def test_rate_limit_single(thread_id):
    # Create ~100k token message that will be cached
    big_text = "This is a large block of text for caching. " * 10000  # ~100k tokens
    medium_text = "This is a large block of text for caching. " * 2000  # ~10k tokens

    print(f"Thread {thread_id}: Starting rate limit test with cached content...")

    # Rebuild conversation each time (simulating web agent)
    messages = []

    # Add all previous conversation turns
    for i in range(5):
        if i == 0:
            messages.append(make_message(big_text))
            t0 = time.time()
        else:
            messages.append(make_message(medium_text))
        add_cache_control(messages[-1])
        try:
            usage = make_request(messages)
            dt = time.time() - t0
            print(f"{dt:.2f}: Thread {thread_id}: {usage}")
        except Exception as e:
            print(f"Thread {thread_id}: Error - {e}")
            break
        remove_cache_control(messages[-1])


def test_rate_limit_parallel(num_threads=3):
    print(f"Starting parallel rate limit test with {num_threads} threads...")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(test_rate_limit_single, i) for i in range(num_threads)]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Thread completed with error: {e}")


def test_rate_limit():
    # Original single-threaded version
    test_rate_limit_single(0)


if __name__ == "__main__":
    # Use parallel version to quickly exhaust rate limits
    test_rate_limit_parallel(num_threads=3)

    # Or use original single-threaded version
    # test_rate_limit()
