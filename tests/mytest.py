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






FLAGS_CUSTOM = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=True,
        use_history=True,
        use_past_error_logs=False,
        use_action_history=True,
        use_think_history=False,
        use_diff=False,
        html_type="pruned_html",
        use_screenshot=False,
        use_som=False,
        extract_visible_tag=True,
        extract_clickable_tag=False,
        extract_coords="False",
        filter_visible_elements_only=False,
    ),
    action=dp.ActionFlags(
        action_set=HighLevelActionSetArgs(
            subsets=["bid"],
            multiaction=False,
        ),
        long_description=False,
        individual_examples=True,
    ),
    use_plan=False,
    use_criticise=False,
    use_thinking=True,
    use_memory=False,
    use_concrete_example=True,
    use_abstract_example=True,
    use_hints=True,
    enable_chat=False,
    max_prompt_tokens=40_000,
    be_cautious=True,
    extra_instructions=None,
)

dic={"Qwen/Qwen3-8B": VLLMModelArgs(  # This is the example, already VLLMModelArgs
        model_name="Qwen/Qwen3-8B",
        training_total_tokens=128_000,
        max_total_tokens=16000,
        backend="vllm",
        model_size=8,
        n_gpus=2,
        tensor_parallel_size=2,
        # model_url="https://nonpaying-andra-coordinal.ngrok-free.dev/v1",
    ),
    "openai/gpt-oss-20b": VLLMModelArgs(
        model_name="openai/gpt-oss-20b",
        max_total_tokens=32_000,
        model_size=21,
        backend="vllm",
        n_gpus=2,
        tensor_parallel_size=2,
    )
    ,
    "ServiceNow-AI/Apriel-1.5-15b-Thinker": VLLMModelArgs(
    model_name="ServiceNow-AI/Apriel-1.5-15b-Thinker",
    model_size=15,
    max_total_tokens=40_000,
    backend="vllm",
    n_gpus=2,
    tensor_parallel_size=2,
    ),}

FLAGS_CUSTOM.use_thinking = False
FLAGS_CUSTOM.obs.use_think_history = False

AGENT_CUSTOM = GenericAgentArgs(
    chat_model_args=dic["ServiceNow-AI/Apriel-1.5-15b-Thinker"],
     flags=FLAGS_CUSTOM,)

AGENT_CUSTOM.chat_model_args.temperature = 0.6

study = make_study(
    benchmark="workarena_l2_agent_curriculum_eval",  # or "webarena", "workarena_l1" ...
    agent_args=[AGENT_CUSTOM],
    comment="My first study",
)

study.run(n_jobs=20, exp_root=Path("/home/toolkit/AGENTLAB_TEST/Workarena_l2/"), parallel_backend="ray")