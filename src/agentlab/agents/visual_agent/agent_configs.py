import bgym
from bgym import HighLevelActionSetArgs

import agentlab.agents.dynamic_prompting as dp
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

from .visual_agent import VisualAgentArgs
from .visual_agent_prompts import PromptFlags

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
    action_set=HighLevelActionSetArgs(subsets=["coord"]),
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
