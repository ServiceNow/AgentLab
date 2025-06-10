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
    "llama_32_11b": LlamaModelArgs(
        model_path="meta-llama/Llama-3.2-11B-Vision-Instruct",
        torch_dtype="bfloat16",
        accelerator_config={"mixed_precision": "bf16", "cpu": False},
        reproducibility_config={"temperature": 0.1},
        max_length=32768,
        max_new_tokens=8192,
        checkpoint_file=None,
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
        checkpoint_file=None,
        device=None,
    ),
}

VL_PROMPT_ARGS_DICT = {
    "ui_prompt": UIPromptArgs(
        use_tabs=True,
        use_history=True,
        use_error=True,
        use_abstract_example=True,
        use_concrete_example=True,
    )
}

VL_AGENT_ARGS_DICT = {
    "ui_agent": UIAgentArgs(
        main_vl_model_args=VL_MODEL_ARGS_DICT["gpt_4o"],
        auxiliary_vl_model_args=VL_MODEL_ARGS_DICT["llama_32_11b"],
        ui_prompt_args=VL_PROMPT_ARGS_DICT["ui_prompt"],
        action_set_args=HighLevelActionSetArgs(subsets=["coord"]),
        max_num_retries=4,
    )
}
