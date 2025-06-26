import logging
import os

from agentlab.agents.generic_agent import AGENT_4o_MINI
from agentlab.benchmarks.osworld import OsworldBenchmark
from agentlab.experiments.study import make_study

fmt = "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(funcName)s() - %(message)s"
logging.basicConfig(level=logging.INFO, force=True, format=fmt, handlers=[logging.StreamHandler()])

n_jobs = 1
os.environ["AGENTLAB_DEBUG"] = "1"
os.environ["PROXY_CONFIG_FILE"] = "OSWorld/evaluation_examples/settings/proxy/dataimpulse.json"
if __name__ == "__main__":
    
    study = make_study(
        benchmark= OsworldBenchmark(),
        agent_args=[AGENT_4o_MINI],
        comment="osworld debug 1",
        logging_level=logging.INFO,
        logging_level_stdout=logging.INFO,
    )

    if os.environ.get("AGENTLAB_DEBUG"):
        study.exp_args_list = study.exp_args_list[:3]
        study.run(n_jobs=1, n_relaunch=1, parallel_backend="sequential")
    else:
        study.run(n_jobs=n_jobs, n_relaunch=1, parallel_backend="ray")
