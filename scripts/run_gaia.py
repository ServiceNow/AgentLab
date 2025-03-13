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
    agent_args = TapeAgentArgs("gaia_agent")
    study = make_study(
        benchmark=GaiaBenchmark(split="validation"),
        agent_args=[agent_args],
        comment="Gaia eval",
    )
    print(f"Exp args list len: {len(study.exp_args_list)}")
    study.exp_args_list = study.exp_args_list[:1]
    print(f"Exp args list len: {len(study.exp_args_list)}")
    study.run(n_jobs=1, n_relaunch=1, parallel_backend="sequential")
