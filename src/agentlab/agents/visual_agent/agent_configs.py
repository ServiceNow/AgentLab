from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

from .visual_agent import VisualAgentArgs
from .visual_agent_prompts import PromptFlags
import agentlab.agents.dynamic_prompting as dp
import bgym

# the other flags are ignored for this agent.
DEFAULT_OBS_FLAGS = dp.ObsFlags(
    use_tabs=True,  # will be overridden by the benchmark when set_benchmark is called after initalizing the agent
    use_error_logs=True,
    use_past_error_logs=False,
    use_screenshot=True,
    use_som=False,
    openai_vision_detail="auto",
)

DEFAULT_ACTION_FLAGS = dp.ActionFlags(
    action_set=bgym.HighLevelActionSetArgs(subsets=["coord"]),
    long_description=True,
    individual_examples=False,
)


DEFAULT_PROMPT_FLAGS = PromptFlags(
    obs=DEFAULT_OBS_FLAGS,
    action=DEFAULT_ACTION_FLAGS,
    use_thinking=True,
    use_concrete_example=False,
    use_abstract_example=True,
    enable_chat=False,
    extra_instructions=None,
)

VISUAL_AGENT_4o = VisualAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-2024-05-13"],
    flags=DEFAULT_PROMPT_FLAGS,
)


VISUAL_AGENT_CLAUDE_3_5 = VisualAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openrouter/anthropic/claude-3.5-sonnet:beta"],
    flags=DEFAULT_PROMPT_FLAGS,
)

VISUAL_AGENT_QWEN_2_5_VL_32B = VisualAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openrouter/qwen/qwen2.5-vl-32b-instruct"],
    flags=DEFAULT_PROMPT_FLAGS,
)

def get_som_agent(llm_config: str):
    """Creates basic 1-step vision SOM agent"""
    assert llm_config in CHAT_MODEL_ARGS_DICT, f"Unsupported LLM config: {llm_config}"
    obs_flags = dp.ObsFlags(
        use_tabs=True, 
        use_error_logs=True,
        use_past_error_logs=False,
        use_screenshot=True,
        use_som=True,
        openai_vision_detail="auto",
    )
    action_flags = dp.ActionFlags(
        action_set=bgym.HighLevelActionSetArgs(subsets=["bid"]),
        long_description=True,
        individual_examples=False,
    )
    som_prompt_flags = PromptFlags(
        obs=obs_flags,
        action=action_flags,
        use_thinking=True,
        use_concrete_example=False,
        use_abstract_example=True,
        enable_chat=False,
        extra_instructions=None,
    )

    agent_args = VisualAgentArgs(
        chat_model_args=CHAT_MODEL_ARGS_DICT[llm_config],
        flags=som_prompt_flags,
    )
    model_name = agent_args.chat_model_args.model_name
    agent_args.agent_name = f"VisualAgent-som-{model_name}".replace("/", "_")

    return agent_args


VISUAL_SOM_AGENT_LLAMA4_17B_INSTRUCT = get_som_agent("openrouter/meta-llama/llama-4-maverick")
