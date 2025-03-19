import logging

from agentlab.agents.tapeagent.agent import TapeAgentArgs
from agentlab.benchmarks.gaia import GaiaBenchmark
from agentlab.experiments.study import make_study

logging.basicConfig(
    level=logging.INFO,
    force=True,
    format="%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(funcName)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

if __name__ == "__main__":
    study = make_study(
        benchmark=GaiaBenchmark(split="validation", level="1"),
        agent_args=TapeAgentArgs("gaia_agent"),
        comment="Gaia eval",
        logging_level=logging.INFO,
        logging_level_stdout=logging.INFO,
    )
    # study.exp_args_list = study.exp_args_list[:1]
    study.run(n_jobs=5, n_relaunch=1)
