import logging
import os

from agentlab.agents.tapeagent.agent import TapeAgentArgs, load_config
from agentlab.benchmarks.gaia import GaiaBenchmark, stop_old_sandbox
from agentlab.experiments.study import make_study

fmt = "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(funcName)s() - %(message)s"
logging.basicConfig(level=logging.INFO, force=True, format=fmt, handlers=[logging.StreamHandler()])

if __name__ == "__main__":
    config = load_config("gaia_l1")
    study = make_study(
        benchmark=GaiaBenchmark.from_config(config),  # type: ignore
        agent_args=TapeAgentArgs(agent_name=config.name, config=config),
        comment=config.comment,
        logging_level=logging.INFO,
        logging_level_stdout=logging.INFO,
    )
    stop_old_sandbox()
    if os.environ.get("AGENTLAB_DEBUG"):
        study.exp_args_list = study.exp_args_list[:3]
        study.run(n_jobs=1, n_relaunch=1, parallel_backend="sequential")
    else:
        study.run(n_jobs=config.n_jobs, n_relaunch=1, parallel_backend=config.parallel_backend)
