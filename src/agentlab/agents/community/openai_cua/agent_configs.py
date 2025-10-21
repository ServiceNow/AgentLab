from bgym import HighLevelActionSetArgs

from .agent import OpenAIComputerUseAgentArgs

OPENAI_CUA_AGENT_ARGS = OpenAIComputerUseAgentArgs(
    model="computer-use-preview",
    tool_type="computer_use_preview",
    display_width=1024,
    display_height=768,
    environment="browser",
    reasoning_summary="concise",
    truncation="auto",
    action_set=HighLevelActionSetArgs(
        subsets=("chat", "coord"),
        demo_mode=None,
    ),
    enable_safety_checks=False,
    implicit_agreement=True,
)
