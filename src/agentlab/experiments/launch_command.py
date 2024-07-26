"""
Note: This script is a convenience script to launch experiments instead of using
the command line.

Don't push your changes to this file to git unless you are making structural changes.
"""

# set basic config of loggig to debug
import logging

from agentlab.analyze.inspect_results import get_most_recent_folder
from agentlab.experiments.exp_utils import RESULTS_DIR
from agentlab.experiments.launch_exp import (
    import_object,
    make_study,
    relaunch_study,
    run_experiments,
    study_agent_on_benchmark,
)

logging.getLogger().setLevel(logging.INFO)

exp_args_list = None

## select your experiment group here from generic_agent/exp_configs.py, as a Python path
exp_config = "agentlab.agents.generic_agent.exp_config.final_run"
# exp_config = "agentlab.agents.generic_agent.exp_config.generic_agent_test"  ## this will make a very quick test
# exp_config = "agentlab.agents.generic_agent.exp_config.generic_agent_eval_llm"
# exp_config = "agentlab.agents.generic_agent.exp_config.random_search"

## select your agent config here from generic_agent/agent_config.py, as a Python path
agent_config = "agentlab.agents.generic_agent.agent_config.AGENT_3_5"
# agent_config = "agentlab.agents.generic_agent.agent_config.AGENT_4o"
# agent_config = "agentlab.agents.generic_agent.agent_config.AGENT_4o_VISION"
# agent_config = "agentlab.agents.generic_agent.agent_config.AGENT_70B"
# agent_config = None # if exp_config uses a default agent

## select the benchmark to run on
benchmark = "miniwob"
# benchmark = "workarena.l1"
# benchmark = "workarena.l2"
# benchmark = "workarena.l3"
# benchmark = "webarena"

# You can also decide to relaunch experiments with the following options
# relaunch_mode = "incomplete_only"
# relaunch_mode = "all_errors"
relaunch_mode = None

## If you want to relaunch experiments, specify the exp_root here, for example:
# exp_root = "<path-to-your-results>/2024-01-22_23-46-25_final_run"
# usually, you can just use the default:
exp_root = RESULTS_DIR  # + "/2024-01-22_23-46-25_final_run" if relaunching

## Extra arguments to pass to the experiment group
extra_kwargs = {}  # anything you want to pass to the experiment group defined in exp_config

## Number of parallel jobs
n_jobs = 1
# n_jobs = -1  # to use all available cores


if relaunch_mode is not None:
    assert exp_root is not None, "You must specify an exp_root to relaunch experiments."
    exp_args_list, exp_dir = relaunch_study(exp_root, relaunch_mode)
else:
    # we launch an experiment using the exp_config
    assert exp_config is not None, "You must specify an exp_config."
    study_func = import_object(exp_config)
    if agent_config is not None:
        agent = import_object(agent_config)
        exp_args_list, exp_dir = study_agent_on_benchmark(
            exp_root, study_func, agent, benchmark, extra_kwargs
        )
    else:
        exp_args_list, exp_dir = make_study(exp_root, study_func, extra_kwargs)

# run the experiments
run_experiments(n_jobs, exp_args_list, exp_dir)
