"""
Note: This script is a convenience script to launch experiments instead of using
the command line.

Don't push your changes to this file to git unless you are making structural changes.
"""

from agentlab.analyze.inspect_results import get_most_recent_folder
from agentlab.experiments.launch_exp import main
from agentlab.experiments.exp_utils import RESULTS_DIR

# set basic config of loggig to debug
import logging

logging.getLogger().setLevel(logging.INFO)

exp_args_list = None

## select your experiment group here from exp_configs.py
# exp_group_name = "generic_agent_test"  ## this will make a very quick test
exp_group_name = "tgi_toolkit_test"  ## this will make a very quick test
# exp_group_name = "generic_agent_eval_llm"
# exp_group_name = "random_search"
# exp_group_name = "ablation_study_GPT_3_5"
# exp_group_name = "finetuning_eval"


## you can also specify the experiment group name directly here to relaunch it
# exp_group_name = "2024-01-22_23-46-25_random_search_prompt_OSS_LLMs"

# WorkArena Ablation Study for ICML
# exp_group_name = "2024-02-01_03-20-14_ablation_study_browsergym_workarena"

# MiniWob Ablation Study for ICML
# exp_group_name = "2024-02-01_03-24-01_ablation_study_browsergym_miniwob"


# exp_group_name = get_most_recent_folder(RESULTS_DIR).name

# relaunch_mode = "incomplete_only"
# relaunch_mode = "all_errors"
relaunch_mode = None


main(
    exp_root=RESULTS_DIR,
    exp_group_name=exp_group_name,
    exp_args_list=exp_args_list,
    n_jobs=5,  # 1 for debugging, -1 for all cores except 2
    relaunch_mode=relaunch_mode,  # choices = [None, 'incomplete_only', 'all_errors', 'server_errors']. if not None, make sure you're pointing to an existing experiment directory
    auto_accept=True,  # skip the prompt to accept the experiment
)
