"""
Note: This script is a convenience script to launch experiments instead of using
the command line.

Copy this script and modify at will, but don't push your changes to the
repository.
"""

import logging

import bgym
from bgym import DEFAULT_BENCHMARKS

from agentlab.agents.tool_use_agent.tool_use_agent import (
    DEFAULT_PROMPT_CONFIG,
    GPT_4_1_MINI,
    OPENAI_MODEL_CONFIG,
    ToolUseAgentArgs,
)
from agentlab.experiments.study import Study

logging.getLogger().setLevel(logging.INFO)

agent_config = ToolUseAgentArgs(
    model_args=OPENAI_MODEL_CONFIG,
    config=GPT_4_1_MINI,
)


agent_config.config.action_subsets = ("workarena",)  # use the workarena action set

agent_args = [agent_config]


# ## select the benchmark to run on
# benchmark = "miniwob_tiny_test"
benchmark = "workarena_l1"


benchmark = DEFAULT_BENCHMARKS[benchmark](n_seeds=1)  # type: bgym.Benchmark
benchmark = benchmark.subset_from_glob("task_name", "*create*")

# for env_args in benchmark.env_args_list:
#     print(env_args.task_name)
#     env_args.max_steps = 15

relaunch = False

## Number of parallel jobs
n_jobs = 10  # Make sure to use 1 job when debugging in VSCode
parallel_backend = "ray"
parallel_backend = "sequential"

if __name__ == "__main__":  # necessary for dask backend

    if relaunch:
        #  relaunch an existing study
        study = Study.load_most_recent(contains=None)
        study.find_incomplete(include_errors=True)

    else:
        study = Study(agent_args, benchmark, logging_level_stdout=logging.WARNING)

    study.run(
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,  # "ray", "joblib" or "sequential"
        strict_reproducibility=False,
        n_relaunch=3,
    )
