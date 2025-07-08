"""
Note: This script is a convenience script to launch experiments instead of using
the command line.

Copy this script and modify at will, but don't push your changes to the
repository.
"""

import logging
import argparse
import os
from typing import cast

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
)
from browsergym.experiments.benchmark.utils import make_env_args_list_from_repeat_tasks
from browsergym.experiments.benchmark.metadata.utils import task_list_from_metadata, task_metadata
import numpy as np

from agentlab.experiments.study import Study
from agentlab.agents.tool_use_agent.tool_use_agent import AGENT_CONFIG


# Parse command line arguments
parser = argparse.ArgumentParser(description='Run AgentLab experiments')
parser.add_argument('--exp-root', type=str, 
                   default="/Users/had.nekoeiqachkanloo/hadi/AgentLab/agentlab_results_no_hint/",
                   help='Root directory for experiment results')
parser.add_argument('--use-task-hint', action='store_true', default=False,
                   help='Whether to use task hints')
parser.add_argument('--hint-db-path', type=str,
                   default="/Users/had.nekoeiqachkanloo/hadi/AgentLab/agentlab_results_no_hint/hint_db_updated.csv",
                   help='Path to hint database CSV file')
args = parser.parse_args()

# Set environment variables from command line arguments
os.environ["AGENTLAB_EXP_ROOT"] = os.path.expandvars(args.exp_root)

logging.getLogger().setLevel(logging.INFO)

# choose your agent or provide a new agent
# agent_args = [AGENT_4o_MINI]
# agent_args = [AGENT_4o]

# Ensure the hint database directory exists
hint_db_dir = os.path.dirname(args.hint_db_path)
if not os.path.exists(hint_db_dir):
    os.makedirs(hint_db_dir, exist_ok=True)
    print(f"Created hint database directory: {hint_db_dir}")

# Create an empty hint database if it doesn't exist and hints are enabled
if not os.path.exists(args.hint_db_path):
    import pandas as pd
    empty_db = pd.DataFrame({
        "time_stamp": [],
        "task_name": [],
        "task_seed": [],
        "base_llm": [],
        "agent_name": [],
        "domain_name": [],
        "user_name": [],
        "source": [],
        "semantic_keys": [],
        "hint": []
    })
    empty_db.to_csv(args.hint_db_path, index=False)
    print(f"Created empty hint database at: {args.hint_db_path}")

AGENT_CONFIG.config.task_hint.hint_db_rel_path = args.hint_db_path
agent_args = cast(list, [AGENT_CONFIG])

from bgym import DEFAULT_BENCHMARKS

benchmark = DEFAULT_BENCHMARKS["miniwob"]()
benchmark.env_args_list = make_env_args_list_from_repeat_tasks(
    task_list=task_list_from_metadata(metadata=task_metadata("miniwob")),
    max_steps=10,
    n_repeats=10,
    seeds_rng=np.random.RandomState(42),
)

# Filter to get 5 seeds for each task instead of just 5 total experiments
selected_tasks = [
    "miniwob.use-colorwheel-2",
    "miniwob.count-shape", 
    "miniwob.form-sequence-2",
    "miniwob.click-scroll-list",
    "miniwob.daily-calendar",
    "miniwob.drag-items-grid",
    "miniwob.grid-coordinate",
    "miniwob.hot-cold",
    "miniwob.right-angle",
    "miniwob.social-media-all"
]

# Get all experiments for the selected tasks
filtered_env_args = []
for env_args in benchmark.env_args_list:
    task_name = env_args.task_name
    if task_name in selected_tasks:
        filtered_env_args.append(env_args)

# Sort by task name and seed to group them
filtered_env_args.sort(key=lambda x: (x.task_name, x.task_seed))

# Take first 5 seeds for each task
final_env_args = []
current_task = None
current_task_count = 0
max_seeds_per_task = 10

for env_args in filtered_env_args:
    if env_args.task_name != current_task:
        current_task = env_args.task_name
        current_task_count = 0
    
    if current_task_count < max_seeds_per_task:
        final_env_args.append(env_args)
        current_task_count += 1

benchmark.env_args_list = final_env_args

print(f"Running {len(final_env_args)} experiments:")
for env_args in final_env_args:
    print(f"  - {env_args.task_name} (seed {env_args.task_seed})")

for env_args in benchmark.env_args_list:
    env_args.headless = True # for seeing the task


# ## select the benchmark to run on
# benchmark = "miniwob_tiny_test"
# benchmark = "miniwob"
# benchmark = "workarena_l1"
# benchmark = "workarena_l2"
# benchmark = "workarena_l3"
# benchmark = "webarena"

# Set reproducibility_mode = True for reproducibility
# this will "ask" agents to be deterministic. Also, it will prevent you from launching if you have
# local changes. For your custom agents you need to implement set_reproducibility_mode
reproducibility_mode = False

# Set relaunch = True to relaunch an existing study, this will continue incomplete
# experiments and relaunch errored experiments
relaunch = False

## Number of parallel jobs
n_jobs = 6  # Make sure to use 1 job when debugging in VSCode
# n_jobs = -1  # to use all available cores

agent_args[0].config.task_hint.use_task_hint = args.use_task_hint

if __name__ == "__main__":  # necessary for dask backend

    if reproducibility_mode:
        [a.set_reproducibility_mode() for a in agent_args]

    if relaunch:
        #  relaunch an existing study
        study = Study.load_most_recent(contains=None)
        study.find_incomplete(include_errors=True)

    else:
        study = Study(agent_args, benchmark, logging_level_stdout=logging.WARNING)

    study.run(
        n_jobs=n_jobs,
        parallel_backend="ray",
        strict_reproducibility=reproducibility_mode,
        n_relaunch=3,
    )

    if reproducibility_mode:
        study.append_to_journal(strict_reproducibility=True)
