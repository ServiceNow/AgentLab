import logging
import os

from agentlab.agents.tool_use_agent.tool_use_agent import OSWORLD_CLAUDE
from agentlab.benchmarks.osworld import OsworldBenchmark
from agentlab.experiments.study import make_study

fmt = "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(funcName)s() - %(message)s"
logging.basicConfig(level=logging.INFO, force=True, format=fmt, handlers=[logging.StreamHandler()])


def main():
    n_jobs = 1
    os.environ["AGENTLAB_DEBUG"] = "1"
    study = make_study(
        benchmark=OsworldBenchmark(),  # type: ignore
        agent_args=[OSWORLD_CLAUDE],
        comment="osworld debug 1",
        logging_level=logging.INFO,
        logging_level_stdout=logging.INFO,
    )

    if os.environ.get("AGENTLAB_DEBUG"):
        study.exp_args_list = study.exp_args_list[:1]
        study.run(n_jobs=1, n_relaunch=1, parallel_backend="sequential")
    else:
        study.run(n_jobs=n_jobs, n_relaunch=1, parallel_backend="ray")


if __name__ == "__main__":
    main()
