from browsergym.experiments.benchmark import HighLevelActionSetArgs
from .vl_agent.ui_agent import UIAgentArgs
from .vl_model.llama_model import LlamaModelArgs
from .vl_model.openrouter_api_model import OpenRouterAPIModelArgs
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
        checkpoint_dir=None,
        max_length=32768,
        max_new_tokens=8192,
        reproducibility_config={"temperature": 0.1},
    ),
}

VL_PROMPT_ARGS_DICT = {
    "ui_prompt": UIPromptArgs(
        action_set_args=HighLevelActionSetArgs(subsets=["coord"]),
        use_screenshot=True,
        use_screenshot_som=False,
        use_tabs=True,
        use_history=True,
        use_error=True,
        use_abstract_example=True,
        use_concrete_example=False,
    )
}

VL_AGENT_ARGS_DICT = {
    "ui_agent": UIAgentArgs(
        main_vl_model_args=VL_MODEL_ARGS_DICT["gpt_4o"],
        auxiliary_vl_model_args=VL_MODEL_ARGS_DICT["llama_32_11b"],
        ui_prompt_args=VL_PROMPT_ARGS_DICT["ui_prompt"],
        max_retry=4,
    )
}
