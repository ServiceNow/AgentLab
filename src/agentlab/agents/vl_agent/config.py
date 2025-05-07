from browsergym.experiments.benchmark import HighLevelActionSetArgs
from .vl_agent import UIAgentArgs
from .vl_model import LlamaModelArgs
from .vl_prompt import VLPromptFlags
import agentlab.agents.dynamic_prompting as dp


VL_MODEL_ARGS_DICT = {
    "llama_32_11b": LlamaModelArgs(
        model_name="llama_32_11b",
        model_path="meta-llama/Llama-3.2-11B-Vision-Instruct",
        torch_dtype="bfloat16",
        checkpoint_dir=None,
        max_length=32768,
        max_new_tokens=8192,
        reproducibility_config={"temperature": 0.1},
    )
}


VL_PROMPT_FLAGS_DICT = {
    "default": VLPromptFlags(
        obs_flags=dp.ObsFlags(
            use_tabs=True,
            use_error_logs=True,
            use_past_error_logs=False,
            use_screenshot=True,
            use_som=False,
            openai_vision_detail="auto",
        ),
        action_flags=dp.ActionFlags(
            action_set=HighLevelActionSetArgs(subsets=["coord"]),
            long_description=True,
            individual_examples=False,
        ),
        use_thinking=True,
        use_concrete_example=False,
        use_abstract_example=True,
        enable_chat=False,
        extra_instructions=None,
    )
}

VL_AGENT_ARGS_DICT = {
    "ui_agent-llama_32_11b": UIAgentArgs(
        general_vl_model_args=VL_MODEL_ARGS_DICT["llama_32_11b"],
        grounding_vl_model_args=VL_MODEL_ARGS_DICT["llama_32_11b"],
        vl_prompt_flags=VL_PROMPT_FLAGS_DICT["default"],
        max_retry=4,
    )
}
