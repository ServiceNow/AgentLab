import logging
import os

from agentlab.agents.tapeagent.agent import TapeAgentArgs, load_config
from agentlab.backends.browser.mcp_playwright import MCPPlaywright
from agentlab.benchmarks.miniwob import MiniWobBenchmark
from agentlab.experiments.study import make_study

fmt = "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(funcName)s() - %(message)s"
logging.basicConfig(level=logging.INFO, force=True, format=fmt, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    config = load_config("gaia_l1")
    study = make_study(
        benchmark=MiniWobBenchmark(backend=MCPPlaywright()),
        agent_args=TapeAgentArgs(agent_name=config.name, config=config),
        comment=config.comment,
        logging_level=logging.INFO,
        logging_level_stdout=logging.INFO,
    )
    if os.environ.get("AGENTLAB_DEBUG"):
        study.exp_args_list = study.exp_args_list[:3]
        study.run(n_jobs=1, n_relaunch=1, parallel_backend="sequential")
    else:
        study.run(n_jobs=config.n_jobs, n_relaunch=1, parallel_backend=config.parallel_backend)
