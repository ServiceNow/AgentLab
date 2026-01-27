"""
Run CheatingAgent with cheat_custom adapters on a small WorkArena subset.
"""

import os

# This is very import for runtime. DO NOT remove!
os.environ.pop("SNOW_INSTANCE_PWD", None)
os.environ.pop("SNOW_INSTANCE_URL", None)
os.environ.pop("SNOW_INSTANCE_UNAME", None)

import logging

import bgym

from agentlab.agents import CHEATING_AGENT
from agentlab.cheat_custom.workarena_adapters import register_workarena_cheat_customs
from agentlab.experiments.study import make_study

TASK_IDS = [
    # L1 tasks (kept for later):
    # "workarena.servicenow.all-menu",
    # "workarena.servicenow.filter-incident-list",
    # "workarena.servicenow.create-incident",
    # "workarena.servicenow.order-apple-watch",
    # L3 tasks (navigate + do):
    "workarena.servicenow.navigate-and-create-incident-l3",
    "workarena.servicenow.navigate-and-filter-incident-list-l3",
    "workarena.servicenow.navigate-and-order-apple-watch-l3",
]

# Number of parallel jobs
n_jobs = 1
parallel_backend = "ray"
avg_step_timeout = 120  # seconds per step used for Ray cancel timeout
max_steps = 50  # override WorkArena default episode length (was 15 in your env)

# Increase WorkArena Playwright default timeout (ms)
CHEATING_AGENT.snow_browser_timeout_ms = 120_000

if __name__ == "__main__":
    register_workarena_cheat_customs()

    benchmark = bgym.DEFAULT_BENCHMARKS["workarena_l3"]()
    benchmark.env_args_list = [
        env_args for env_args in benchmark.env_args_list if env_args.task_name in TASK_IDS
    ]

    for env_args in benchmark.env_args_list:
        env_args.headless = True
        env_args.max_steps = max_steps

    study = make_study(
        benchmark=benchmark,
        agent_args=[CHEATING_AGENT],
        comment="cheat_custom L3 subset",
    )
    study.avg_step_timeout = avg_step_timeout
    study.run(
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        strict_reproducibility=False,
        n_relaunch=3,
    )
