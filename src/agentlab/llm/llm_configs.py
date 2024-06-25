from agentlab.llm.chat_api import OpenAIChatModelArgs, ToolkitModelArgs
from agentlab.llm import toolkit_configs

default_oss_llms_args = {
    "infer_tokens_length": True,
    "max_trunk_itr": int(1e6),
    "max_new_tokens": 512,
    "temperature": 0.01,
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
    # "reka/reka-core": ChatModelArgs(
    #     model_name="reka/reka-core",
    #     max_total_tokens=128_000,
    #     vision_support=True,
    #     **default_oss_llms_args,
    # ),
    # "reka/reka-edge": ChatModelArgs(
    #     model_name="reka/reka-edge",
    #     max_total_tokens=128_000,
    #     vision_support=True,
    #     **default_oss_llms_args,
    # ),
    # "reka/reka-flash": ChatModelArgs(
    #     model_name="reka/reka-flash",
    #     max_total_tokens=128_000,
    #     vision_support=True,
    #     **default_oss_llms_args,
    # ),
    # ---------------- OSS LLMs ----------------#
    ## SOTA
    "finetuning/Meta-Llama-3-8B-Instruct": ToolkitModelArgs(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        model_path=f"{toolkit_configs.FINETUNING_CKPT_PATH}/meta-llama/Meta-Llama-3-8B-Instruct/finetuning_output/",
        training_total_tokens=8_192,
        model_size=8,
        is_model_operational=True,
        **default_oss_llms_args,
    ),
    "finetuning/debug": ToolkitModelArgs(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        model_path=f"{toolkit_configs.FINETUNING_CKPT_PATH}/meta-llama/Meta-Llama-3-8B-Instruct/finetuning_output/ATOMIC_TASKS_240604/ckpt_itr_0",
        training_total_tokens=8_192,
        model_size=8,
        is_model_operational=True,
        **default_oss_llms_args,
    ),
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
    "deepseek-ai/DeepSeek-V2-Chat": ToolkitModelArgs(
        model_name="deepseek-ai/DeepSeek-V2-Chat",
        model_path=f"{toolkit_configs.BASE_MODELS_PATH}/deepseek-ai/DeepSeek-V2-Chat/",
        training_total_tokens=128_000,
        max_total_tokens=128_000,
        model_size=236,
        is_model_operational=True,
        **default_oss_llms_args,
        info="looks like sharding is not supported, but the model doesn't fit on a single GPU...",
    ),
    "microsoft/Phi-3-mini-128k-instruct": ToolkitModelArgs(
        model_name="microsoft/Phi-3-mini-128k-instruct",
        training_total_tokens=128_000,
        model_size=3.8,
        is_model_operational=False,
        shard_support=False,
        extra_tgi_args={"TRUST_REMOTE_CODE": "true"},
        **default_oss_llms_args,
    ),
    "microsoft/Phi-3-small-8k-instruct": ToolkitModelArgs(
        model_name="microsoft/Phi-3-small-8k-instruct",
        training_total_tokens=8_000,
        model_size=7,
        is_model_operational=False,
        shard_support=False,
        extra_tgi_args={"TRUST_REMOTE_CODE": "true"},
        **default_oss_llms_args,
    ),
    "microsoft/Phi-3-medium-4k-instruct": ToolkitModelArgs(
        model_name="microsoft/Phi-3-medium-4k-instruct",
        training_total_tokens=4_000,
        model_size=14,
        is_model_operational=False,
        shard_support=False,
        extra_tgi_args={"TRUST_REMOTE_CODE": "true"},
        **default_oss_llms_args,
    ),
    "microsoft/WizardLM-2-8x22B": ToolkitModelArgs(
        model_name="microsoft/WizardLM-2-8x22B",
        model_path=f"{toolkit_configs.BASE_MODELS_PATH}/microsoft/WizardLM-2-8x22B/",
        training_total_tokens=64_000,
        max_total_tokens=32_000,
        model_size=176,  # 4x44b
        is_model_operational=True,
        **default_oss_llms_args,
    ),
    "Salesforce/xLAM-v0.1-r": ToolkitModelArgs(
        model_name="Salesforce/xLAM-v0.1-r",
        training_total_tokens=32_768,
        model_size="8x7",
        is_model_operational=True,
        **default_oss_llms_args,
    ),
    "CohereForAI/c4ai-command-r-plus": ToolkitModelArgs(
        model_name="CohereForAI/c4ai-command-r-plus",
        training_total_tokens=128_000,
        model_size=104,
        is_model_operational=False,
        **default_oss_llms_args,
        info="seems like its not supported by TGI",
    ),
    "CohereForAI/c4ai-command-r-v01": ToolkitModelArgs(
        model_name="CohereForAI/c4ai-command-r-v01",
        training_total_tokens=128_000,
        model_size=35,
        is_model_operational=False,
        **default_oss_llms_args,
        info="seems like its not supported by TGI",
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
    "databricks/dbrx-instruct": ToolkitModelArgs(
        model_name="databricks/dbrx-instruct",
        training_total_tokens=32_768,
        model_size="4x36",
        is_model_operational=False,
        **default_oss_llms_args,
    ),
    "deepseek-ai/deepseek-coder-6.7b-instruct": ToolkitModelArgs(
        model_name="deepseek-ai/deepseek-coder-6.7b-instruct",
        training_total_tokens=16_384,
        model_size=7,
        is_model_operational=True,
        tgi_image=toolkit_configs.TGI_IMAGE_LLMD,
        **default_oss_llms_args,
    ),
    "deepseek-ai/deepseek-coder-6.7b-base": ToolkitModelArgs(
        model_name="deepseek-ai/deepseek-coder-6.7b-base",
        training_total_tokens=16_384,
        model_size=7,
        is_model_operational=True,
        tgi_image=toolkit_configs.TGI_IMAGE_LLMD,
        **default_oss_llms_args,
    ),
    "deepseek-ai/deepseek-coder-33b-instruct": ToolkitModelArgs(
        model_name="deepseek-ai/deepseek-coder-33b-instruct",
        training_total_tokens=16_384,
        model_size=33,
        is_model_operational=True,
        tgi_image=toolkit_configs.TGI_IMAGE_LLMD,
        **default_oss_llms_args,
    ),
    "deepseek-ai/deepseek-coder-33b-base": ToolkitModelArgs(
        model_name="deepseek-ai/deepseek-coder-33b-base",
        training_total_tokens=16_384,
        model_size=33,
        is_model_operational=True,
        tgi_image=toolkit_configs.TGI_IMAGE_LLMD,
        **default_oss_llms_args,
    ),
    ## MistralAI
    "mistralai/Mixtral-8x7B-Instruct-v0.1": ToolkitModelArgs(
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        training_total_tokens=32_768,
        model_size="8x7",  # 4x14b
        is_model_operational=False,
        **default_oss_llms_args,
    ),
    "mistralai/Mistral-7B-Instruct-v0.2": ToolkitModelArgs(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        training_total_tokens=32_768,
        model_size=7,
        is_model_operational=False,
        **default_oss_llms_args,
    ),
    ## CodeLLAMA
    "codellama/CodeLlama-7b-Python-hf": ToolkitModelArgs(
        model_name="codellama/CodeLlama-7b-Python-hf",
        training_total_tokens=16_384,
        model_size=7,
        is_model_operational=True,
        **default_oss_llms_args,
    ),
    "codellama/CodeLlama-13b-Python-hf": ToolkitModelArgs(
        model_name="codellama/CodeLlama-13b-Python-hf",
        training_total_tokens=16_384,
        model_size=13,
        is_model_operational=True,
        **default_oss_llms_args,
    ),
    "codellama/CodeLlama-34b-Python-hf": ToolkitModelArgs(
        model_name="codellama/CodeLlama-34b-Python-hf",
        training_total_tokens=16_384,
        model_size=34,
        is_model_operational=True,
        **default_oss_llms_args,
    ),
    "codellama/CodeLlama-70b-Python-hf": ToolkitModelArgs(
        model_name="codellama/CodeLlama-70b-Python-hf",
        training_total_tokens=4_096,
        model_size=70,
        is_model_operational=False,
        **default_oss_llms_args,
    ),
    "codellama/CodeLlama-7b-instruct-hf": ToolkitModelArgs(
        model_name="codellama/CodeLlama-7b-instruct-hf",
        training_total_tokens=16_384,
        model_size=7,
        is_model_operational=True,
        **default_oss_llms_args,
    ),
    "codellama/CodeLlama-13b-instruct-hf": ToolkitModelArgs(
        model_name="codellama/CodeLlama-13b-instruct-hf",
        training_total_tokens=16_384,
        model_size=13,
        is_model_operational=True,
        **default_oss_llms_args,
    ),
    "codellama/CodeLlama-34b-instruct-hf": ToolkitModelArgs(
        model_name="codellama/CodeLlama-34b-instruct-hf",
        training_total_tokens=16_384,
        model_size=34,
        is_model_operational=True,
        **default_oss_llms_args,
    ),
    "codellama/CodeLlama-70b-instruct-hf": ToolkitModelArgs(
        model_name="codellama/CodeLlama-70b-instruct-hf",
        training_total_tokens=4_096,
        model_size=70,
        is_model_operational=False,
        **default_oss_llms_args,
    ),
    ## Bigcode
    "bigcode/starcoder2": ToolkitModelArgs(
        model_name="bigcode/starcoder2",
        training_total_tokens=16_384,
        model_size=15,
        is_model_operational=False,
        sliding_window=True,
        **default_oss_llms_args,
    ),
    "bigcode/starcoder2-7b": ToolkitModelArgs(
        model_name="bigcode/starcoder2-7b",
        training_total_tokens=16_384,
        model_size=3,
        is_model_operational=False,
        sliding_window=True,
        **default_oss_llms_args,
    ),
    "HuggingFaceH4/starchat2-15b-v0.1": ToolkitModelArgs(
        model_name="HuggingFaceH4/starchat2-15b-v0.1",
        training_total_tokens=16_384,
        model_size=3,
        is_model_operational=False,
        sliding_window=True,
        **default_oss_llms_args,
    ),
    "bigcode/starcoder": ToolkitModelArgs(
        model_name="bigcode/starcoder",
        training_total_tokens=8_192,
        model_size=15,
        is_model_operational=False,
        **default_oss_llms_args,
    ),
    "bigcode/starcoderbase": ToolkitModelArgs(
        model_name="bigcode/starcoderbase",
        training_total_tokens=8_192,
        model_size=15,
        is_model_operational=False,
        **default_oss_llms_args,
    ),
    "bigcode/starcoderbase-1b": ToolkitModelArgs(
        model_name="bigcode/starcoderbase-1b",
        training_total_tokens=8_192,
        model_size=1,
        is_model_operational=False,
        **default_oss_llms_args,
    ),
    "bigcode/starcoderbase-3b": ToolkitModelArgs(
        model_name="bigcode/starcoderbase-3b",
        training_total_tokens=8_192,
        model_size=3,
        is_model_operational=False,
        **default_oss_llms_args,
    ),
    "bigcode/starcoderbase-7b": ToolkitModelArgs(
        model_name="bigcode/starcoderbase-7b",
        training_total_tokens=8_192,
        model_size=7,
        is_model_operational=False,
        **default_oss_llms_args,
    ),
    "bigcode/starcoderplus": ToolkitModelArgs(
        model_name="bigcode/starcoderplus",
        model_path="/mnt/llmd/base_models/starcoderplus",
        training_total_tokens=8_192,
        model_size=15,
        is_model_operational=True,
        **default_oss_llms_args,
    ),
    "HuggingFaceH4/starchat-beta": ToolkitModelArgs(
        model_name="HuggingFaceH4/starchat-beta",
        model_path="/mnt/llmd/base_models/starchat-beta",
        training_total_tokens=8_192,
        model_size=15,
        is_model_operational=True,
        **default_oss_llms_args,
    ),
    ## Others
    "THUDM/agentlm-70b": ToolkitModelArgs(
        model_name="THUDM/agentlm-70b",
        training_total_tokens=4_096,
        model_size=70,
        is_model_operational=False,
        **default_oss_llms_args,
    ),
    ## Test
    "microsoft/Phi-3-mini-4k-instruct": ToolkitModelArgs(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        training_total_tokens=4_096,
        model_size=3.8,
        is_model_operational=True,
        extra_tgi_args={"TRUST_REMOTE_CODE": "true"},
        **default_oss_llms_args,
    ),
    "microsoft/phi-1": ToolkitModelArgs(
        model_name="microsoft/phi-1",
        training_total_tokens=2_048,
        model_size=1,
        is_model_operational=True,
        **default_oss_llms_args,
        info="somehow, doesn't work in the latest TGI image",
    ),
}

# TODO: the base infra hparams could be infered from total params
# NOTE: optimizing for a 8-16k context window
INFRA_HPARAMS_DICT_BASE = {
    1: {"gpu": 1, "gpu_mem": 16, "cpu": 6, "mem": 64},  # NOTE: for tests
    4: {"gpu": 1, "cpu": 6, "mem": 64},  # NOTE: for tests
    8: {"gpu": 1, "cpu": 6, "mem": 64},
    22: {"gpu": 2, "cpu": 8, "mem": 128},
    41: {"gpu": 2, "cpu": 8, "mem": 128},
    56: {"gpu": 2, "cpu": 8, "mem": 256},
    72: {"gpu": 4, "cpu": 12, "mem": 256},
    200: {
        "gpu": 4,
        "cpu": 12,
        "mem": 512,
    },
    300: {
        "gpu": 5,
        "cpu": 12,
        "mem": 512,
    },
}

# TODO: definetely needs improvement
CONTEXT_WINDOW_EXTRA_GPU = {
    8_096: 0,
    16_384: 0,
    32_768: 1,
    64_000: 2,
    128_000: 3,
}
