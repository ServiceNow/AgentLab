from agentlab.experiments.study import make_study

from agentlab.agents.generic_agent import AGENT_CUSTOM
from agentlab.agents.generic_agent import GenericAgentArgs
from pathlib import Path
from agentlab.llm.chat_api import (
    AnthropicModelArgs,
    AzureModelArgs,
    OpenAIModelArgs,
    OpenRouterModelArgs,
    SelfHostedModelArgs,
)

from finetuning.toolkit_utils.chat_api import ToolkitModelArgs, VLLMModelArgs

import os
os.environ["SNOW_INSTANCE_URL"]="https://empmassimo17.service-now.com/"
os.environ["SNOW_INSTANCE_UNAME"]="admin"
os.environ["SNOW_INSTANCE_PWD"]="AE_82fH4ZPntQuJ"

CHECKPOINT_ACCOUNT_NAME_SUFFIX = (
    # "ui_assist"
    "adea"
)
DATA_ROOT_PATH = f"/mnt/{CHECKPOINT_ACCOUNT_NAME_SUFFIX}/data"
FINETUNING_PATH = Path(DATA_ROOT_PATH) / "finetuning"
MINIWOB_URL = str(FINETUNING_PATH) + "/.miniwob-plusplus/miniwob/html/miniwob/"
MINIWOB_URL = "file://" + MINIWOB_URL
if os.getenv("MINIWOB_URL", None) is None:
    os.environ["MINIWOB_URL"] = MINIWOB_URL

import bgym
from bgym import HighLevelActionSetArgs

from agentlab.agents import dynamic_prompting as dp
from agentlab.experiments import args
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

from agentlab.agents.generic_agent import GenericAgentArgs


from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags



from agentlab.agents.generic_agent import AGENT_4o_MINI 
from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_4o

# Customize Apriel endpoint if needed

apriel_flags=FLAGS_GPT_4o

apriel_flags.use_thinking=False
apriel_flags.obs.use_think_history=False

AGENT_SLAM_15B_CUSTOM = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["apriel/slam-15b"],
    flags=apriel_flags,
)

# Override base_url, api_key, and model_name for custom endpoint
AGENT_SLAM_15B_CUSTOM.chat_model_args.base_url = "https://infer-rc10-runai-nowllm.inference.ta121237.dgxcloud.ai/v1"
AGENT_SLAM_15B_CUSTOM.chat_model_args.api_key = "-cPh_74qWJBTooW6MTf6ew:R1zzSE2wSYHwy8CdvPx3s3xdCuQRbhSiY7ydscQEOZw"
AGENT_SLAM_15B_CUSTOM.chat_model_args.model_name = "openai/Slam-15B"  # Change model name if needed
# You can also override other params like temperature
# AGENT_SLAM_15B_CUSTOM.chat_model_args.temperature = 0.8
# AGENT_SLAM_15B_CUSTOM.chat_model_args.max_new_tokens = 10_000


study = make_study(
    benchmark="workarena_l1",  # or "webarena", "workarena_l1" ...
    agent_args=[AGENT_SLAM_15B_CUSTOM],  # Use the customized agent
    comment="My first study",
)


path = ""

study.run(n_jobs=10, exp_root=Path(path), parallel_backend="ray")