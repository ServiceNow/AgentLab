"""Test script for Apriel Slam-15B model on WorkArena L1 benchmark."""

import os
from pathlib import Path

from agentlab.agents.generic_agent import GenericAgentArgs
from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_4o
from agentlab.experiments.study import make_study
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT
from agentlab.llm.vllm_server import VLLMModelArgs
# Configure MiniWoB URL
DATA_ROOT_PATH = "/mnt/adea/data"
MINIWOB_URL = f"file://{DATA_ROOT_PATH}/finetuning/.miniwob-plusplus/miniwob/html/miniwob/"
os.environ.setdefault("MINIWOB_URL", MINIWOB_URL)

# Configure Apriel flags (disable thinking for this model)
apriel_flags = FLAGS_GPT_4o
apriel_flags.use_thinking = False
apriel_flags.obs.use_think_history = False


MODELS={
    "ServiceNow-AI/Apriel-1.5-15b-Thinker": VLLMModelArgs(
    model_name="ServiceNow-AI/Apriel-1.5-15b-Thinker",
    model_size=15,
    max_total_tokens=40_000,
    backend="vllm",
    n_gpus=2,
    tensor_parallel_size=2,
    )
}

# Create agent with custom Apriel endpoint
AGENT_APRIEL_15B = GenericAgentArgs(
    chat_model_args=MODELS["ServiceNow-AI/Apriel-1.5-15b-Thinker"],
    flags=apriel_flags,
)
AGENT_SLAM_15B = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["apriel/slam-15b"],
    flags=apriel_flags,
)
AGENT_SLAM_15B.chat_model_args.base_url = "https://apriel1p6-15b-runai-nowllm.inference.ta121237.dgxcloud.ai/v1"
AGENT_SLAM_15B.chat_model_args.api_key = "-cPh_74qWJBTooW6MTf6ew:R1zzSE2wSYHwy8CdvPx3s3xdCuQRbhSiY7ydscQEOZw"
AGENT_SLAM_15B.chat_model_args.model_name = "Apriel-1p6-15B-Thinker"

# Run study
study = make_study(
    benchmark="workarena_l1",
    agent_args=[AGENT_SLAM_15B],
    comment="Apriel Slam-15B evaluation",
)


study.run(n_jobs=10, exp_root=Path(""), parallel_backend="ray")