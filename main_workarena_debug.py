"""
Note: This script is a convenience script to launch experiments instead of using
the command line.

Copy this script and modify at will, but don't push your changes to the
repository.
"""

import os
import logging
from copy import deepcopy

import bgym
from agentlab.agents.tool_use_agent.tool_use_agent import (
    DEFAULT_PROMPT_CONFIG,
    GPT_4_1,
    GPT_4_1_MINI,
    ToolUseAgentArgs,
)
from agentlab.experiments.study import Study

logging.getLogger().setLevel(logging.INFO)

os.environ["AGENTLAB_EXP_ROOT"] = "/home/toolkit/agentlab_results"
os.environ["SNOW_INSTANCE_URL"] = "https://researchworkarenademo.service-now.com"
os.environ["SNOW_INSTANCE_UNAME"] = "admin"
os.environ["SNOW_INSTANCE_PWD"] = "Snow@123"


config = deepcopy(DEFAULT_PROMPT_CONFIG)
# config.keep_last_n_obs = 1
config.obs.use_som = True

agent_configs = [
    ToolUseAgentArgs(
        # model_args=GPT_4_1,
        model_args=GPT_4_1_MINI,
        config=config,
    ),
    # ToolUseAgentArgs(
    #     model_args=GPT_4_1,
    #     config=config,
    # ),
]

for agent_config in agent_configs:
    agent_config.config.action_subsets = ("workarena",)  # use the workarena action set
    agent_config.config.multiaction = False  # use single action per step

# ## select the benchmark to run on
# benchmark = "miniwob_tiny_test"
benchmark = "workarena_l1"

benchmark = bgym.DEFAULT_BENCHMARKS[benchmark]()  # type: bgym.Benchmark

benchmark = benchmark.subset_from_glob("task_name", "*order*")

# take a task from here: https://docs.google.com/spreadsheets/d/15cMaSNvQPKMltxG9XYp2MinHt0702fNaBoI0EfRkfxU/edit?gid=0#gid=0
# task_name = "order-apple-mac-book-pro15"
# task_name = "order-apple-watch"
# task_name = "order-developer-laptop"
# task_name = "order-development-laptop-p-c"
# task_name = "order-ipad-mini"
# task_name = "order-ipad-pro"
# task_name = "order-loaner-laptop"
# task_name = "order-sales-laptop"
# task_name = "order-standard-laptop"
# benchmark = benchmark.subset_from_list(
#     [f"workarena.servicenow.{task_name}"], benchmark_name_suffix=task_name
# )

# for env_args in benchmark.env_args_list:
#     print(env_args.task_name)
#     env_args.max_steps = 15

relaunch = False

## Number of parallel jobs
n_jobs = 10  # Make sure to use 1 job when debugging in VSCode
parallel_backend = "ray"
# parallel_backend = "sequential"  # activate sequential backend for debugging in VSCode

if __name__ == "__main__":  # necessary for dask backend

    if relaunch:
        #  relaunch an existing study
        study = Study.load_most_recent(contains=None)
        study.find_incomplete(include_errors=True)

    else:
        study = Study(agent_configs, benchmark, logging_level_stdout=logging.WARNING)

    study.run(
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,  # "ray", "joblib" or "sequential"
        strict_reproducibility=False,
        n_relaunch=3,
    )
