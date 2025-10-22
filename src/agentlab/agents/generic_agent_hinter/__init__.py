from agentlab.agents.hint_use_agent import *
import warnings
import sys

warnings.warn(
    "generic_agent_hinter is renamed to hint_use_agent.",
    DeprecationWarning,
    stacklevel=2,
)

# Create module alias - redirect old module to new module
import agentlab.agents.hint_use_agent as new_module

sys.modules["agentlab.agents.generic_agent_hinter"] = new_module

# Re-export everything from the new location
from agentlab.agents.hint_use_agent import *
