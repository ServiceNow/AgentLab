from openai import NOT_GIVEN

from agentlab.llm.chat_api import (
    AnthropicModelArgs,
    AzureModelArgs,
    BedrockModelArgs,
    OpenAIModelArgs,
    OpenRouterModelArgs,
    SelfHostedModelArgs,
)

default_oss_llms_args = {
    "n_retry_server": 4,
    "temperature": 0.01,
}

CLOSED_SOURCE_APIS = [
    "openai",
    "reka",
    "test",
]

CHAT_MODEL_ARGS_DICT = {
    "openai/gpt-5-2025-08-07": OpenAIModelArgs(
        model_name="gpt-5-2025-08-07",
        max_total_tokens=400_000,
        max_input_tokens=256_000,
        max_new_tokens=128_000,
        temperature=1,  # gpt-5 supports temperature of 1 only
        vision_support=True,
    ),
    "openai/gpt-5-nano-2025-08-07": OpenAIModelArgs(
        model_name="gpt-5-nano-2025-08-07",
        max_total_tokens=400_000,
        max_input_tokens=256_000,
        max_new_tokens=128_000,
        temperature=1,  # gpt-5 supports temperature of 1 only
        vision_support=True,
    ),
    "openai/gpt-5-mini-2025-08-07": OpenAIModelArgs(
        model_name="gpt-5-mini-2025-08-07",
        max_total_tokens=400_000,
        max_input_tokens=256_000,
        max_new_tokens=128_000,
        temperature=1,  # gpt-5 supports temperature of 1 only
        vision_support=True,
    ),
    "openai/gpt-4.1-nano-2025-04-14": OpenAIModelArgs(
        model_name="gpt-4.1-nano-2025-04-14",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=16_384,
        vision_support=True,
    ),
    "openai/gpt-4.1-mini-2025-04-14": OpenAIModelArgs(
        model_name="gpt-4.1-mini-2025-04-14",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=16_384,
        vision_support=True,
    ),
    "openai/gpt-4.1-2025-04-14": OpenAIModelArgs(
        model_name="gpt-4.1-2025-04-14",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=16_384,
        vision_support=True,
    ),
    "openai/o3-mini-2025-01-31": OpenAIModelArgs(
        model_name="o3-mini-2025-01-31",
        max_total_tokens=200_000,
        max_input_tokens=200_000,
        max_new_tokens=100_000,
        vision_support=False,
    ),
    "openai/o3-2025-04-16": OpenAIModelArgs(
        model_name="o3-2025-04-16",
        max_total_tokens=200_000,
        max_input_tokens=200_000,
        max_new_tokens=None,
        temperature=1,
        vision_support=True,
    ),
    "openai/gpt-4o-mini-2024-07-18": OpenAIModelArgs(
        model_name="gpt-4o-mini-2024-07-18",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=16_384,
        vision_support=True,
    ),
    "openai/gpt-4-1106-preview": OpenAIModelArgs(
        model_name="gpt-4-1106-preview",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=4_096,
    ),
    "openai/gpt-4-vision-preview": OpenAIModelArgs(
        model_name="gpt-4-vision-preview",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=16_384,  # I think this model has very small default value if we don't set max_new_tokens
        vision_support=True,
    ),
    "openai/gpt-4o-2024-05-13": OpenAIModelArgs(
        model_name="gpt-4o-2024-05-13",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=4_096,  # I think this model has very small default value if we don't set max_new_tokens
        vision_support=True,
    ),
    "openai/gpt-3.5-turbo-0125": OpenAIModelArgs(
        model_name="gpt-3.5-turbo-0125",
        max_total_tokens=16_384,
        max_input_tokens=16_384,
        max_new_tokens=4096,
    ),
    "openai/gpt-3.5-turbo-1106": OpenAIModelArgs(
        model_name="gpt-3.5-turbo-1106",
        max_total_tokens=16_384,
        max_input_tokens=16_384,
        max_new_tokens=4096,
    ),
    "openai/o1-mini": OpenAIModelArgs(
        model_name="openai/o1-mini",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=64_000,
        temperature=1e-1,
    ),
    "azure/gpt-35-turbo/gpt-35-turbo": AzureModelArgs(
        model_name="gpt-35-turbo",
        max_total_tokens=8_192,
        max_input_tokens=7500,
        max_new_tokens=500,
    ),
    "azure/gpt-4o-2024-05-13": AzureModelArgs(
        model_name="gpt-4o",
        max_total_tokens=128_000,
        max_input_tokens=100_000,
        max_new_tokens=16_384,
        vision_support=True,
    ),
    "azure/gpt-4o-2024-08-06": AzureModelArgs(
        model_name="gpt-4o",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=16_384,
        vision_support=True,
    ),
    "azure/gpt-4o-mini-2024-07-18": AzureModelArgs(
        model_name="gpt-4o-mini",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=16_384,
        vision_support=True,
    ),
    "azure/gpt-4.1-2025-04-14": AzureModelArgs(
        model_name="gpt-4.1",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=16_384,
        vision_support=True,
    ),
    "azure/gpt-4.1-mini-2025-04-14": AzureModelArgs(
        model_name="gpt-4.1-mini",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=16_384,
        vision_support=True,
    ),
    "azure/gpt-4.1-nano-2025-04-14": AzureModelArgs(
        model_name="gpt-4.1-nano",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=16_384,
        vision_support=True,
    ),
    "azure/gpt-5-2025-08-07": AzureModelArgs(
        model_name="gpt-5",
        max_total_tokens=400_000,
        max_input_tokens=256_000,
        max_new_tokens=128_000,
        temperature=1,  # temperature param not supported by gpt-5
        vision_support=True,
    ),
    "azure/gpt-5-mini-2025-08-07": AzureModelArgs(
        model_name="gpt-5-mini",
        max_total_tokens=400_000,
        max_input_tokens=256_000,
        max_new_tokens=128_000,
        temperature=1,  # temperature param not supported by gpt-5
        vision_support=True,
    ),
    "azure/gpt-5-nano-2025-08-07": AzureModelArgs(
        model_name="gpt-5-nano",
        max_total_tokens=400_000,
        max_input_tokens=256_000,
        max_new_tokens=128_000,
        temperature=1,  # temperature param not supported by gpt-5
        vision_support=True,
    ),
    # ---------------- Anthropic ----------------#
    "anthropic/claude-3-7-sonnet-20250219": AnthropicModelArgs(
        model_name="claude-3-7-sonnet-20250219",
        max_new_tokens=16_384,
        temperature=1e-1,
    ),
    "anthropic/claude-sonnet-4-20250514": AnthropicModelArgs(
        model_name="claude-sonnet-4-20250514",
        max_new_tokens=16_384,
        temperature=1e-1,
    ),
    # ------------ Anthropic / Bedrock ------------#
    "bedrock/claude-3-7-sonnet": BedrockModelArgs(
        model_name="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        max_new_tokens=16_384,
        temperature=1e-1,
    ),
    "bedrock/claude-4-0-sonnet": BedrockModelArgs(
        model_name="us.anthropic.claude-sonnet-4-20250514-v1:0",
        max_new_tokens=16_384,
        temperature=1e-1,
    ),
    "bedrock/claude-4-5-sonnet": BedrockModelArgs(
        model_name="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        max_new_tokens=16_384,
        temperature=1e-1,
    ),
    # ---------------- OSS LLMs ----------------#
    "meta-llama/Meta-Llama-3-70B-Instruct": SelfHostedModelArgs(
        model_name="meta-llama/Meta-Llama-3-70B-Instruct",
        max_total_tokens=8_192,
        max_input_tokens=8_192 - 512,
        max_new_tokens=512,
        backend="huggingface",
        **default_oss_llms_args,
    ),
    "meta-llama/Meta-Llama-3-8B-Instruct": SelfHostedModelArgs(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        max_total_tokens=16_384,
        max_input_tokens=16_384 - 512,
        max_new_tokens=512,
        backend="huggingface",
        **default_oss_llms_args,
    ),
    "mistralai/Mixtral-8x22B-Instruct-v0.1": SelfHostedModelArgs(
        model_name="mistralai/Mixtral-8x22B-Instruct-v0.1",
        max_total_tokens=32_000,
        max_input_tokens=30_000,
        max_new_tokens=2_000,
        backend="huggingface",
        **default_oss_llms_args,
    ),
    # ---------------- OPENROUTER ----------------#
    "openrouter/deepseek/deepseek-r1": OpenRouterModelArgs(
        model_name="deepseek/deepseek-r1",
        max_total_tokens=128_000,
        max_input_tokens=100_000,
        max_new_tokens=128_000,
        temperature=1e-1,
    ),
    "openrouter/meta-llama/llama-3.1-405b-instruct": OpenRouterModelArgs(
        model_name="meta-llama/llama-3.1-405b-instruct",
        max_total_tokens=128_000,
        max_input_tokens=100_000,
        max_new_tokens=28_000,
        temperature=1e-1,
    ),
    "openrouter/meta-llama/llama-3.1-70b-instruct": OpenRouterModelArgs(
        model_name="meta-llama/llama-3.1-70b-instruct",
        max_total_tokens=128_000,
        max_input_tokens=100_000,
        max_new_tokens=28_000,
        temperature=1e-1,
    ),
    "openrouter/meta-llama/llama-3-70b-instruct": OpenRouterModelArgs(
        model_name="meta-llama/llama-3-70b-instruct",
        max_total_tokens=128_000,
        max_input_tokens=100_000,
        max_new_tokens=28_000,
        temperature=1e-1,
    ),
    "openrouter/meta-llama/llama-4-maverick": OpenRouterModelArgs(
        model_name="meta-llama/llama-4-maverick",
        max_total_tokens=128_000,
        max_input_tokens=100_000,
        max_new_tokens=28_000,
        temperature=1e-1,
        vision_support=True,
    ),
    "openrouter/meta-llama/llama-3.1-8b-instruct:free": OpenRouterModelArgs(
        model_name="meta-llama/llama-3.1-8b-instruct:free",
        max_total_tokens=128_000,
        max_input_tokens=100_000,
        max_new_tokens=28_000,
        temperature=1e-1,
    ),
    "openrouter/meta-llama/llama-3.1-8b-instruct": OpenRouterModelArgs(
        model_name="meta-llama/llama-3.1-8b-instruct",
        max_total_tokens=128_000,
        max_input_tokens=100_000,
        max_new_tokens=28_000,
        temperature=1e-1,
    ),
    "openrouter/anthropic/claude-3.5-sonnet:beta": OpenRouterModelArgs(
        model_name="anthropic/claude-3.5-sonnet:beta",
        max_total_tokens=200_000,
        max_input_tokens=200_000,
        max_new_tokens=8_192,
        temperature=1e-1,
        vision_support=True,
    ),
    "openrouter/qwen/qwen-2-72b-instruct": OpenRouterModelArgs(
        model_name="qwen/qwen-2-72b-instruct",
        max_total_tokens=32_000,
        max_input_tokens=30_000,
        max_new_tokens=2_000,
        temperature=1e-1,
    ),
    "openrouter/anthropic/claude-3.7-sonnet": OpenRouterModelArgs(
        model_name="anthropic/claude-3.7-sonnet",
        max_total_tokens=200_000,
        max_input_tokens=200_000,
        max_new_tokens=8_192,
        temperature=1e-1,
        vision_support=True,
    ),
    "openrouter/openai/o1-mini-2024-09-12": OpenRouterModelArgs(
        model_name="openai/o1-mini-2024-09-12",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=64_000,
        temperature=1e-1,
    ),
    "openrouter/openai/gpt-oss-120b": OpenRouterModelArgs(
        model_name="openai/gpt-oss-120b",
        max_total_tokens=131_072,
        max_input_tokens=131_072 - 2000,
        max_new_tokens=2000,
        temperature=1e-1,
    ),
    "openrouter/openai/gpt-oss-20b": OpenRouterModelArgs(
        model_name="openai/gpt-oss-20b",
        max_total_tokens=131_072,
        max_input_tokens=131_072 - 2000,
        max_new_tokens=2000,
        temperature=1e-1,
    ),
    "openrouter/openai/gpt-5-nano": OpenRouterModelArgs(
        model_name="openai/gpt-5-nano",
        max_total_tokens=400_000,
        max_input_tokens=400_000 - 4_000,
        max_new_tokens=4_000,
        temperature=1e-1,
    ),
    "openrouter/openai/gpt-5-mini": OpenRouterModelArgs(
        model_name="openai/gpt-5-mini",
        max_total_tokens=400_000,
        max_input_tokens=400_000 - 4_000,
        max_new_tokens=4_000,
        temperature=1e-1,
    ),
    "openrouter/openai/gpt-5-chat": OpenRouterModelArgs(
        model_name="openai/gpt-5-chat",
        max_total_tokens=400_000,
        max_input_tokens=400_000 - 4_000,
        max_new_tokens=4_000,
        temperature=1e-1,
    ),
    "openrouter/openai/o3-mini": OpenRouterModelArgs(
        model_name="openai/o3-mini",
        max_total_tokens=200_000,
        max_input_tokens=100_000 - 4_000,
        max_new_tokens=4_000,
        temperature=1e-1,
    ),
}
