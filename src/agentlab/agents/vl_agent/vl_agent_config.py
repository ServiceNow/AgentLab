from .vl_agent import UIAgentArgs
from .vl_model_config import VL_MODEL_ARGS_DICT
from .vl_prompt_config import VL_PROMPT_FLAGS_DICT


VL_AGENT_ARGS_DICT = {
    "ui_agent-llama_32_11b": UIAgentArgs(
        general_model_args=VL_MODEL_ARGS_DICT["llama_32_11b"],
        grounding_model_args=VL_MODEL_ARGS_DICT["llama_32_11b"],
        prompt_flags=VL_PROMPT_FLAGS_DICT["default"],
        max_retry=4,
    )
}
