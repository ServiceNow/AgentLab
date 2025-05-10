from browsergym.experiments.benchmark import HighLevelActionSetArgs
from .vl_agent import UIAgentArgs
from .vl_model import LlamaModelArgs
from .vl_prompt import UIPromptArgs
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

VL_PROMPT_ARGS_DICT = {
    "ui_prompt-default": UIPromptArgs(
        prompt_name="ui_prompt-default",
        obs_flags=dp.ObsFlags(
            use_tabs=True,
            use_error_logs=True,
            use_past_error_logs=False,
            use_screenshot=True,
            use_som=False,
        ),
        action_flags=dp.ActionFlags(
            action_set=HighLevelActionSetArgs(subsets=["coord"]),
            long_description=True,
            individual_examples=False,
        ),
        extra_instructions=None,
        enable_chat=False,
        use_thinking=True,
        use_abstract_example=True,
        use_concrete_example=False,
    )
}

VL_AGENT_ARGS_DICT = {
    "ui_agent-llama_32_11b-llama_32_11b": UIAgentArgs(
        agent_name="ui_agent-llama_32_11b-llama_32_11b",
        main_vl_model_args=VL_MODEL_ARGS_DICT["llama_32_11b"],
        auxiliary_vl_model_args=VL_MODEL_ARGS_DICT["llama_32_11b"],
        ui_prompt_args=VL_PROMPT_ARGS_DICT["ui_prompt-default"],
        max_retry=4,
    )
}
