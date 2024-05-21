import random
from huggingface_hub import InferenceClient
import asyncio
import time
from functools import partial

# Your bearer token for authorization
token = "42"

# Model endpoint (hardcoded)
model_endpoint = "https://6b4e3cbe-5fb2-4ae5-a87f-4a2c2d8bc119.job.console.elementai.com"

# Creating the client instance
client = InferenceClient(model=model_endpoint, token=token)

# variable:
# input_seq_length = 2_000
input_seq_length = 15_872
max_new_tokens = 20

# prompt = random string of "hello " repeated input_seq_length/2 times
prompt = "hello " * (input_seq_length // 2)

# make sure the prompt are different in case there's caching speedups
first_words = [
    "hello",
    "by",
    "the",
]


# Generate a prompt with a random first word
def generate_prompt():
    first_word = random.choice(first_words)
    prompt = (first_word + " ") * (input_seq_length // 2)
    print(f"prompt: {first_word}")
    return prompt


async def make_request():
    loop = asyncio.get_event_loop()
    prompt = generate_prompt()  # Generate a new prompt for each request
    # Making the inference request asynchronously using run_in_executor
    partial_func = partial(client.text_generation, prompt=prompt, max_new_tokens=max_new_tokens)
    result = await loop.run_in_executor(None, partial_func)

    print(result)  # Print the result as soon as it is available
    return result


async def main():
    start_time = time.time()
    end_time = start_time + 60  # 1 minute later
    count = 0
    tasks = []

    while time.time() < end_time:
        # Schedule a new request
        tasks.append(make_request())
        count += 1

        # If you want to limit the number of parallel jobs, you can do so here
        if len(tasks) >= 10:  # Adjust this number based on your desired level of parallelism
            await asyncio.gather(*tasks)
            tasks = []  # Reset the list of tasks

    # Wait for any remaining tasks to complete
    if tasks:
        await asyncio.gather(*tasks)

    print(f"Total queries in 1 minute: {count}")
    # total number of tokens in a minute:
    tokens_per_minute = count * (input_seq_length + max_new_tokens)
    print(f"Total tokens generated in 1 minute: {tokens_per_minute}")
    # total number of tokens that we could generate in 1h
    tokens_per_hour = tokens_per_minute * 60
    print(f"Total tokens generated in 1 hour: {tokens_per_hour}")


if __name__ == "__main__":
    asyncio.run(main())
