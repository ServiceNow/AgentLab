"""
Convenience script to run the CheatingAgent on WorkArena L1-L3.

Copy and modify locally as needed; don't push changes upstream.
"""

import logging

import bgym

from agentlab.agents import CHEATING_AGENT
from agentlab.experiments.study import make_study

# Force DEBUG logging (including CheatingAgent internals)
# logging.basicConfig(level=logging.DEBUG)
# root_logger = logging.getLogger()
# root_logger.setLevel(logging.DEBUG)
# for handler in root_logger.handlers:
#     handler.setLevel(logging.DEBUG)
# logging.getLogger("agentlab.agents.cheating_agent").setLevel(logging.DEBUG)

benchmarks = ["workarena_l3_agent_curriculum_eval"]

# Number of parallel jobs
n_jobs = 1
parallel_backend = "ray"

if __name__ == "__main__":
    for benchmark in benchmarks:
        benchmark_obj = bgym.DEFAULT_BENCHMARKS[benchmark]()
        for env_args in benchmark_obj.env_args_list:
            env_args.headless = True

        study = make_study(
            benchmark=benchmark_obj,
            agent_args=[CHEATING_AGENT],
            comment="cheat trajectories",
        )
        study.run(
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            strict_reproducibility=False,
            n_relaunch=3,
        )
