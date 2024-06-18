from agentlab.llm.chat_api import ChatModelArgs, OpenAIChatModelArgs
from agentlab.llm import toolkit_configs

default_oss_llms_args = {
    "infer_tokens_length": True,
    "max_trunk_itr": int(1e6),
    "max_new_tokens": 512,
}

CLOSED_SOURCE_APIS = [
    "openai",
    "reka",
    "test",
]

CHAT_MODEL_ARGS_DICT = {
    "openai/gpt-4-1106-preview": OpenAIChatModelArgs(
        model_name="openai/gpt-4-1106-preview",
        max_total_tokens=128_000,
        max_input_tokens=40_000,  # make sure we don't bust budget
        max_new_tokens=4000,
    ),
    "openai/gpt-4-vision-preview": OpenAIChatModelArgs(
        model_name="openai/gpt-4-vision-preview",
        max_total_tokens=128_000,
        max_input_tokens=40_000,  # make sure we don't bust budget
        max_new_tokens=4000,  # I think this model has very small default value if we don't set max_new_tokens
        vision_support=True,
    ),
    "openai/gpt-4o-2024-05-13": OpenAIChatModelArgs(
        model_name="openai/gpt-4o-2024-05-13",
        max_total_tokens=128_000,
        max_input_tokens=40_000,  # make sure we don't bust budget
        max_new_tokens=4000,  # I think this model has very small default value if we don't set max_new_tokens
        vision_support=True,
    ),
    "openai/gpt-3.5-turbo-0125": OpenAIChatModelArgs(
        model_name="openai/gpt-3.5-turbo-0125",
        max_total_tokens=16_384,
        max_input_tokens=15_000,
        max_new_tokens=1_000,
    ),
    "openai/gpt-3.5-turbo-1106": OpenAIChatModelArgs(
        model_name="openai/gpt-3.5-turbo-1106",
        max_total_tokens=16_384,
        max_input_tokens=15_000,
        max_new_tokens=1_000,
    ),
}
