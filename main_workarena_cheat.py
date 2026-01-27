"""
Convenience script to run the CheatingAgent on WorkArena L1-L3.

Copy and modify locally as needed; don't push changes upstream.
"""

import os

# This is very import for runtime. DO NOT remove!
os.environ.pop("SNOW_INSTANCE_PWD", None)
os.environ.pop("SNOW_INSTANCE_URL", None)
os.environ.pop("SNOW_INSTANCE_UNAME", None)
# Do not remove or override: keep experiment outputs local to this repo.
os.environ.setdefault(
    "AGENTLAB_EXP_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "agentlab_results"),
)

import logging

import bgym

from agentlab.agents import CHEATING_AGENT
from agentlab.experiments.loop import log_reasoning_effort_reminder
from agentlab.experiments.study import make_study

# Force DEBUG logging (including CheatingAgent internals)
# logging.basicConfig(level=logging.DEBUG)
# root_logger = logging.getLogger()
# root_logger.setLevel(logging.DEBUG)
# for handler in root_logger.handlers:
#     handler.setLevel(logging.DEBUG)
# logging.getLogger("agentlab.agents.cheating_agent").setLevel(logging.DEBUG)

benchmarks = ["workarena_l1", "workarena_l2_agent_curriculum_eval", "workarena_l3_agent_curriculum_eval"]

# Number of parallel jobs
n_jobs = 50
parallel_backend = "ray"
avg_step_timeout = 1200  # seconds per step used for Ray cancel timeout
max_steps = 50  # override WorkArena default episode length (was 15 in your env)

# Increase WorkArena Playwright default timeout (ms)
CHEATING_AGENT.snow_browser_timeout_ms = 120_000

if __name__ == "__main__":
    log_reasoning_effort_reminder(CHEATING_AGENT)
    for benchmark in benchmarks:
        benchmark_obj = bgym.DEFAULT_BENCHMARKS[benchmark]()
        for env_args in benchmark_obj.env_args_list:
            env_args.headless = True
            env_args.max_steps = max_steps

        study = make_study(
            benchmark=benchmark_obj,
            agent_args=[CHEATING_AGENT],
            comment="cheat trajectories",
        )
        study.avg_step_timeout = avg_step_timeout
        study.run(
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            strict_reproducibility=False,
            n_relaunch=3,
        )
