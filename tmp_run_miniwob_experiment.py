# Import your agent configuration extending bgym.AgentArgs class
# Make sure this object is imported from a module accessible in PYTHONPATH to properly unpickle
from agentlab.agents.generic_agent import AGENT_AZURE_4o_MINI
import logging
from agentlab.experiments.study import make_study

study = make_study(
    benchmark="miniwob",  # or "webarena", "workarena_l1" ...
    agent_args=[AGENT_AZURE_4o_MINI],
    comment="My third study",
    logging_level=logging.DEBUG,
    logging_level_stdout=logging.DEBUG,
)

study.run(n_jobs=5)
