from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

from .agent import WebDreamerArgs, WebDreamerFlags

FLAGS_4o_MINI = WebDreamerFlags(use_refiner=True, num_controller_samples=5, num_value_samples=20)

AGENT_4o_MINI = WebDreamerArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["azure/gpt-4o-mini-2024-07-18"],
    flags=FLAGS_4o_MINI,
)
