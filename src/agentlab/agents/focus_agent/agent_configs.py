from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_4o
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

from .focus_agent import FocusAgentArgs
from .llm_retriever_prompt import LlmRetrieverPromptFlags

FLAGS_GPT_4o = FLAGS_GPT_4o.copy()
FLAGS_GPT_4o.obs.use_think_history = True


FOCUS_AGENT_4_1_MINI = FocusAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-mini-2025-04-14"],
    flags=FLAGS_GPT_4o,
    retriever_chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-mini-2025-04-14"],
    retriever_prompt_flags=LlmRetrieverPromptFlags(
        use_abstract_example=False,
        use_concrete_example=False,
        use_screenshot=False,
        use_history=False,
    ),
    max_retry=4,
)

FOCUS_AGENT_CLAUDE_3_7_RETRIEVER_4_1_MINI = FocusAgentArgs(
    agent_name="FocusAgent-claude-3-7-Retriever-4.1-mini",
    chat_model_args=CHAT_MODEL_ARGS_DICT["anthropic/claude-3-7-sonnet-20250219"],
    flags=FLAGS_GPT_4o,
    retriever_chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-mini-2025-04-14"],
    retriever_prompt_flags=LlmRetrieverPromptFlags(
        use_abstract_example=False,
        use_concrete_example=False,
        use_screenshot=False,
        use_history=False,
    ),
    max_retry=4,
    keep_structure=False,
    retriever_type="line",
)

FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI = FocusAgentArgs(
    agent_name="FocusAgent-4.1-Retriever-4.1-mini",
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-2025-04-14"],
    flags=FLAGS_GPT_4o,
    retriever_chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-mini-2025-04-14"],
    retriever_prompt_flags=LlmRetrieverPromptFlags(
        use_abstract_example=False,
        use_concrete_example=False,
        use_screenshot=False,
        use_history=False,
    ),
    max_retry=4,
    keep_structure=False,
    retriever_type="line",
)

FOCUS_AGENT_4_1_RETRIEVER_5_MINI = FocusAgentArgs(
    agent_name="FocusAgent-4.1-Retriever-5-mini",
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-2025-04-14"],
    flags=FLAGS_GPT_4o,
    retriever_chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-5-mini-2025-08-07"],
    retriever_prompt_flags=LlmRetrieverPromptFlags(
        use_abstract_example=False,
        use_concrete_example=False,
        use_screenshot=False,
        use_history=False,
    ),
    max_retry=4,
    keep_structure=False,
    retriever_type="line",
)

FOCUS_AGENT_CLAUDE_3_7_RETRIEVER_5_MINI = FocusAgentArgs(
    agent_name="FocusAgent-claude-3-7-Retriever-5-mini",
    chat_model_args=CHAT_MODEL_ARGS_DICT["anthropic/claude-3-7-sonnet-20250219"],
    flags=FLAGS_GPT_4o,
    retriever_chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-5-mini-2025-08-07"],
    retriever_prompt_flags=LlmRetrieverPromptFlags(
        use_abstract_example=False,
        use_concrete_example=False,
        use_screenshot=False,
        use_history=False,
    ),
    max_retry=4,
    keep_structure=False,
    retriever_type="line",
)

FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI_WITH_STRUCTURE = FocusAgentArgs(
    agent_name="FocusAgent-4.1-Retriever-4.1-mini-with-structure",
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-2025-04-14"],
    flags=FLAGS_GPT_4o,
    retriever_chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-mini-2025-04-14"],
    retriever_prompt_flags=LlmRetrieverPromptFlags(
        use_abstract_example=False,
        use_concrete_example=False,
        use_screenshot=False,
        use_history=False,
    ),
    max_retry=4,
    keep_structure=True,
    strategy="bid",
    retriever_type="line",
)

##### Security Defense Agents #####

DEFENDER_FOCUS_AGENT_4_1_RETRIEVER_4_1_MINI = FocusAgentArgs(
    agent_name="DefenderFocusAgent-4.1-Retriever-4.1-mini",
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-2025-04-14"],
    flags=FLAGS_GPT_4o,
    retriever_chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-mini-2025-04-14"],
    retriever_prompt_flags=LlmRetrieverPromptFlags(
        use_abstract_example=False,
        use_concrete_example=False,
        use_screenshot=False,
        use_history=False,
    ),
    max_retry=4,
    keep_structure=False,
    retriever_type="defender",
)

DEFENDER_FOCUS_AGENT_CLAUDE_3_7_RETRIEVER_4_1_MINI = FocusAgentArgs(
    agent_name="DefenderRetrieverAgent-claude-3-7-Retriever-4.1-mini",
    chat_model_args=CHAT_MODEL_ARGS_DICT["anthropic/claude-3-7-sonnet-20250219"],
    flags=FLAGS_GPT_4o,
    retriever_chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-mini-2025-04-14"],
    retriever_prompt_flags=LlmRetrieverPromptFlags(
        use_abstract_example=False,
        use_concrete_example=False,
        use_screenshot=False,
        use_history=False,
    ),
    max_retry=4,
    keep_structure=False,
    retriever_type="defender",
)
