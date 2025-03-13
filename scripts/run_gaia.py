from agentlab.agents.tapeagent import TapeAgentArgs
from agentlab.benchmarks.gaia import GaiaBenchmark
from agentlab.experiments.study import make_study

exp_dir = "./outputs/gaia/debug1"
agent_args = TapeAgentArgs("gaia_agent")
study = make_study(
    benchmark=GaiaBenchmark(split="validation", exp_dir=exp_dir),
    agent_args=[agent_args],
    comment="Gaia eval",
)

study.run(n_jobs=1)
