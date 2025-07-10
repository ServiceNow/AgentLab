from browsergym.experiments.benchmark import HighLevelActionSetArgs
from .vl_agent.ui_agent import UIAgentArgs
from .vl_model.llama_model import LlamaModelArgs
from .vl_model.openrouter_api_model import OpenRouterAPIModelArgs
from .vl_model.qwen_model import QwenModelArgs
from .vl_prompt.ui_prompt import UIPromptArgs


VL_MODEL_ARGS_DICT = {
    "gpt_4o": OpenRouterAPIModelArgs(
        base_url="https://openrouter.ai/api/v1",
        model_id="openai/gpt-4o-2024-11-20",
        max_tokens=8192,
        reproducibility_config={"temperature": 0.1},
    ),
    "claude_37_sonnet": OpenRouterAPIModelArgs(
        base_url="https://openrouter.ai/api/v1",
        model_id="anthropic/claude-3.7-sonnet",
        max_tokens=8192,
        reproducibility_config={"temperature": 0.1},
    ),
    "gemini_20_flash": OpenRouterAPIModelArgs(
        base_url="https://openrouter.ai/api/v1",
        model_id="google/gemini-2.0-flash-001",
        max_tokens=8192,
        reproducibility_config={"temperature": 0.1},
    ),
    "llama_32_11b": LlamaModelArgs(
        model_path="meta-llama/Llama-3.2-11B-Vision-Instruct",
        torch_dtype="bfloat16",
        accelerator_config={"mixed_precision": "bf16", "cpu": False},
        reproducibility_config={"temperature": 0.1},
        max_length=32768,
        max_new_tokens=8192,
        checkpoint=None,
        device=None,
    ),
    "llama_32_90b": LlamaModelArgs(
        model_path="meta-llama/Llama-3.2-90B-Vision-Instruct",
        torch_dtype="bfloat16",
        accelerator_config={"mixed_precision": "bf16", "cpu": False},
        reproducibility_config={"temperature": 0.1},
        max_length=32768,
        max_new_tokens=8192,
        checkpoint=None,
        device=None,
    ),
    "qwen_25_vl_3b": QwenModelArgs(
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
        accelerator_config={"mixed_precision": "bf16", "cpu": False},
        reproducibility_config={"temperature": 0.1},
        max_length=32768,
        max_new_tokens=8192,
        checkpoint=None,
        device=None,
    ),
    "qwen_25_vl_7b": QwenModelArgs(
        model_path="Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
        accelerator_config={"mixed_precision": "bf16", "cpu": False},
        reproducibility_config={"temperature": 0.1},
        max_length=32768,
        max_new_tokens=8192,
        checkpoint=None,
        device=None,
    ),
    "qwen_25_vl_32b": QwenModelArgs(
        model_path="Qwen/Qwen2.5-VL-32B-Instruct",
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
        accelerator_config={"mixed_precision": "bf16", "cpu": False},
        reproducibility_config={"temperature": 0.1},
        max_length=32768,
        max_new_tokens=8192,
        checkpoint=None,
        device=None,
    ),
    "qwen_25_vl_72b": QwenModelArgs(
        model_path="Qwen/Qwen2.5-VL-72B-Instruct",
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
        accelerator_config={"mixed_precision": "bf16", "cpu": False},
        reproducibility_config={"temperature": 0.1},
        max_length=32768,
        max_new_tokens=8192,
        checkpoint=None,
        device=None,
    ),
}

VL_PROMPT_ARGS_DICT = {
    "ui_prompt": UIPromptArgs(
        use_screenshot_history=False,
        use_tabs=False,
        use_error=True,
        use_abstract_example=True,
        use_concrete_example=True,
        use_location_reasoning=True,
    )
}

VL_AGENT_ARGS_DICT = {
    "ui_agent": UIAgentArgs(
        main_vl_model_args=VL_MODEL_ARGS_DICT["gpt_4o"],
        auxiliary_vl_model_args=VL_MODEL_ARGS_DICT["qwen_25_vl_7b"],
        ui_prompt_args=VL_PROMPT_ARGS_DICT["ui_prompt"],
        action_set_args=HighLevelActionSetArgs(["coord"]),
        max_num_retries=3,
    )
}
