from .vl_prompt import VLPromptFlags
import agentlab.agents.dynamic_prompting as dp
import bgym

VL_OBS_FLAGS = dp.ObsFlags(
    use_tabs=True,
    use_error_logs=True,
    use_past_error_logs=False,
    use_screenshot=True,
    use_som=False,
    openai_vision_detail="auto",
)

VL_ACTION_FLAGS = dp.ActionFlags(
    action_set=bgym.HighLevelActionSetArgs(subsets=["coord"]),
    long_description=True,
    individual_examples=False,
)


VL_PROMPT_FLAGS = VLPromptFlags(
    obs=VL_OBS_FLAGS,
    action=VL_ACTION_FLAGS,
    use_thinking=True,
    use_concrete_example=False,
    use_abstract_example=True,
    enable_chat=False,
    extra_instructions=None,
)
