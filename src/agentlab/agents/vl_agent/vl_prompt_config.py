import agentlab.agents.dynamic_prompting as dp
from browsergym.experiments.benchmark import HighLevelActionSetArgs
from .vl_prompt import VLPromptFlags


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
