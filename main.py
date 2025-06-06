"""
Note: This script is a convenience script to launch experiments instead of using
the command line.

Copy this script and modify at will, but don't push your changes to the
repository.
"""

import logging

from agentlab.agents.generic_agent import (
    AGENT_LLAMA3_70B,
    AGENT_LLAMA31_70B,
    RANDOM_SEARCH_AGENT,
    AGENT_4o,
    AGENT_4o_MINI,
    AGENT_o3_MINI,
    AGENT_o1_MINI,
    AGENT_37_SONNET,
    AGENT_CLAUDE_SONNET_35,
    AGENT_AZURE_4o_MINI,
    AGENT_AZURE_4o,
    AGENT_AZURE_4o_VISION,
    AGENT_AZURE_4o_MINI_VISION,
    AGENT_AZURE_41,
    AGENT_AZURE_41_MINI,
    AGENT_AZURE_41_VISION,
    AGENT_AZURE_41_MINI_VISION,
)
from agentlab.experiments.study import Study
from bgym import DEFAULT_BENCHMARKS

logging.getLogger().setLevel(logging.INFO)

# choose your agent or provide a new agent
# agent_args = [AGENT_4o_MINI]
# agent_args = [AGENT_4o]
agent_args = [AGENT_AZURE_41]


# ## select the benchmark to run on
# benchmark = "miniwob_tiny_test"
# benchmark = "miniwob"
benchmark = "workarena_l1"
# benchmark = "workarena_l2"
# benchmark = "workarena_l3"
# benchmark = "webarena"
benchmark = DEFAULT_BENCHMARKS[benchmark]()

## debugging
benchmark = benchmark.subset_from_list(
    [
        "workarena.servicenow.order-ipad-mini",
        "workarena.servicenow.order-apple-mac-book-pro15",
        "workarena.servicenow.order-apple-watch",
        "workarena.servicenow.order-developer-laptop",
        "workarena.servicenow.order-development-laptop-p-c",
        "workarena.servicenow.order-ipad-mini",
        "workarena.servicenow.order-ipad-pro",
        "workarena.servicenow.order-loaner-laptop",
        "workarena.servicenow.order-sales-laptop",
        "workarena.servicenow.order-standard-laptop",
    ]
)


# benchmark.env_args_list = benchmark.env_args_list[:2]
# for env_args in benchmark.env_args_list:
#     env_args.headless = False  # for seeing the task

# Set reproducibility_mode = True for reproducibility
# this will "ask" agents to be deterministic. Also, it will prevent you from launching if you have
# local changes. For your custom agents you need to implement set_reproducibility_mode
reproducibility_mode = False

# Set relaunch = True to relaunch an existing study, this will continue incomplete
# experiments and relaunch errored experiments
relaunch = False

## Number of parallel jobs
# n_jobs = 1
n_jobs = 12  # Make sure to use 1 job when debugging in VSCode
# n_jobs = -1  # to use all available cores

## backend type
parallel_backend = "ray"
# parallel_backend = "sequential"


if __name__ == "__main__":  # necessary for dask backend

    if reproducibility_mode:
        [a.set_reproducibility_mode() for a in agent_args]

    if relaunch:
        #  relaunch an existing study
        study = Study.load_most_recent(contains=None)
        study.find_incomplete(include_errors=True)

    else:
        study = Study(
            agent_args,
            benchmark,
            # logging_level=logging.DEBUG,
            # logging_level_stdout=logging.DEBUG,
            logging_level=logging.WARNING,
            logging_level_stdout=logging.WARNING,
        )

    study.run(
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        strict_reproducibility=reproducibility_mode,
        n_relaunch=3,
    )

    if reproducibility_mode:
        study.append_to_journal(strict_reproducibility=True)
