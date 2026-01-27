"""
AgentLab's pre-implemented agents.

This module contains the agent implementations for AgentLab. With currently:

- GenericAgent: Our baseline agent for evaluation

- MostBasicAgent: A basic agent for learning our framework

- TapeAgent: An agent that uses the Tape data structure to perform actions

- VisualWebArenaAgent: An implementation of the agent used in WebArena and VisualWebArena
"""

from agentlab.agents.cheating_agent import CHEATING_AGENT, CheatingAgentArgs
from agentlab.agents.cheating_custom_agent import (
    CHEATING_CUSTOM_AGENT,
    CheatingCustomAgentArgs,
)
