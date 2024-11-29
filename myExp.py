from agentlab.agents.generic_agent import AGENT_4o_MINI
from agentlab.experiments.study import make_study

study = make_study(
    benchmark="miniwob_tiny_test", agent_args=[AGENT_4o_MINI], comment="Test avec MiniWoB"
)

study.run(n_jobs=1)
