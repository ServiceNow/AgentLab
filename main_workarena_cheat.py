"""
Convenience script to run the CheatingAgent on WorkArena L1-L3.

Copy and modify locally as needed; don't push changes upstream.
"""

import logging

from agentlab.agents import CHEATING_AGENT
from agentlab.experiments.study import make_study

logging.getLogger().setLevel(logging.INFO)

benchmarks = ["workarena_l1", "workarena_l2", "workarena_l3"]

# Number of parallel jobs
n_jobs = 1
parallel_backend = "ray"

if __name__ == "__main__":
    for benchmark in benchmarks:
        study = make_study(
            benchmark=benchmark,
            agent_args=[CHEATING_AGENT],
            comment="cheat trajectories",
        )
        study.run(
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            strict_reproducibility=False,
            n_relaunch=3,
        )
