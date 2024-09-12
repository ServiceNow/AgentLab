"""
Note: This script is a convenience script to launch experiments instead of using
the command line.

Don't push your changes to this file to git unless you are making structural changes.
"""

from copy import deepcopy
import logging

from agentlab.agents.generic_agent import AGENT_4o, AGENT_4o_MINI
from agentlab.analyze.inspect_results import get_most_recent_folder
from agentlab.experiments import study_generators
from agentlab.experiments.exp_utils import RESULTS_DIR
from agentlab.experiments.launch_exp import make_study_dir, run_experiments, relaunch_study
from agentlab.agents.generic_agent.generic_agent import GenericAgent

logging.getLogger().setLevel(logging.INFO)


def set_temp(agent: GenericAgent, temperature=0):
    agent = deepcopy(agent)
    agent.chat_model_args.temperature = temperature
    return agent


if __name__ == "__main__":

    agent = set_temp(AGENT_4o_MINI)

    ## select the benchmark to run on
    # benchmark = "miniwob"
    benchmark = "miniwob_tiny_test"
    # benchmark = "workarena.l1"
    # benchmark = "workarena.l2"
    # benchmark = "workarena.l3"
    # benchmark = "webarena"

    study_name, exp_args_list = study_generators.run_agents_on_benchmark(agent, benchmark)
    study_dir = make_study_dir(RESULTS_DIR, study_name)

    # ## alternatively, relaunch an existing study
    # study_dir = get_most_recent_folder(RESULTS_DIR, contains=None)
    # exp_args_list, study_dir = relaunch_study(study_dir, relaunch_mode="incomplete_or_error")

    ## Number of parallel jobs
    n_jobs = 3  # Make sure to use 1 job when debugging in VSCode
    # n_jobs = -1  # to use all available cores

    # run the experiments
    run_experiments(n_jobs, exp_args_list, study_dir)
