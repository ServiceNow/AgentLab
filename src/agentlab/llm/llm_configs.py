import os

from agentlab.llm.chat_api import APIModelArgs, ToolkitModelArgs, SelfHostedModelArgs
from agentlab.llm import toolkit_configs

default_oss_llms_args = {
    "infer_tokens_length": True,
    "max_new_tokens": 512,
    "temperature": 0.01,
}

CLOSED_SOURCE_APIS = [
    "openai",
    "reka",
    "test",
]

CHAT_MODEL_ARGS_DICT = {
    "openai/gpt-4-1106-preview": APIModelArgs(
        model_name="openai/gpt-4-1106-preview",
        max_total_tokens=128_000,
        max_input_tokens=40_000,  # make sure we don't bust budget
        max_new_tokens=4000,
    ),
    "openai/gpt-4-vision-preview": APIModelArgs(
        model_name="openai/gpt-4-vision-preview",
        max_total_tokens=128_000,
        max_input_tokens=40_000,  # make sure we don't bust budget
        max_new_tokens=4000,  # I think this model has very small default value if we don't set max_new_tokens
        vision_support=True,
    ),
    "openai/gpt-4o-2024-05-13": APIModelArgs(
        model_name="openai/gpt-4o-2024-05-13",
        max_total_tokens=128_000,
        max_input_tokens=40_000,  # make sure we don't bust budget
        max_new_tokens=4000,  # I think this model has very small default value if we don't set max_new_tokens
        vision_support=True,
    ),
    "openai/gpt-3.5-turbo-0125": APIModelArgs(
        model_name="openai/gpt-3.5-turbo-0125",
        max_total_tokens=16_384,
        max_input_tokens=15_000,
        max_new_tokens=1_000,
    ),
    "openai/gpt-3.5-turbo-1106": APIModelArgs(
        model_name="openai/gpt-3.5-turbo-1106",
        max_total_tokens=16_384,
        max_input_tokens=15_000,
        max_new_tokens=1_000,
    ),
    "azure/gpt-35-turbo/gpt-35-turbo": APIModelArgs(
        model_name="azure/gpt-35-turbo/gpt-35-turbo",
        max_total_tokens=16_384,
        # max_input_tokens=15_000,
        max_new_tokens=1_000,
    ),
    # ---------------- OSS LLMs ----------------#
    "meta-llama/Meta-Llama-3-70B-Instruct": ToolkitModelArgs(
        model_name="meta-llama/Meta-Llama-3-70B-Instruct",
        training_total_tokens=8_192,
        max_total_tokens=8_192,
        model_size=70,
        is_model_operational=True,
        **default_oss_llms_args,
    ),
    "meta-llama/Meta-Llama-3-8B-Instruct": ToolkitModelArgs(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        training_total_tokens=8_192,
        max_total_tokens=16_384,
        model_size=8,
        is_model_operational=True,
        **default_oss_llms_args,
    ),
    "mistralai/Mixtral-8x22B-Instruct-v0.1": ToolkitModelArgs(
        model_name="mistralai/Mixtral-8x22B-Instruct-v0.1",
        training_total_tokens=64_000,
        max_total_tokens=32_000,
        model_size=176,  # 4x44b
        is_model_operational=False,
        **default_oss_llms_args,
        info="it can fit into 8 GPUs, but there's still a TGI bug. Looks like a layer is not supported by TGI?",
    ),
}
