from .vl_agent import VLAgentArgs
from .vl_model_config import VL_MODEL_ARGS_DICT
from .vl_prompt_config import VL_PROMPT_FLAGS


VL_AGENT_ARGS_DICT = {
    "vl_agent_llama_32_11b": VLAgentArgs(
        vl_model_args=VL_MODEL_ARGS_DICT["llama_32_11b"], vl_prompt_flags=VL_PROMPT_FLAGS
    )
}
