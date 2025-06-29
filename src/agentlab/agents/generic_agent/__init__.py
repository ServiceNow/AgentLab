"""
Baseline agent for all ServiceNow papers

This module contains the GenericAgent class, which is the baseline agent for all ServiceNow papers. \
It is a simple agent that can be ran OOB on all BrowserGym environments. It is also shipped with \
a few configurations that can be used to run it on different environments.
"""

from .agent_configs import (
    AGENT_3_5,
    AGENT_8B,
    AGENT_CUSTOM,
    AGENT_LLAMA4_17B_INSTRUCT,
    AGENT_LLAMA3_70B,
    AGENT_LLAMA31_70B,
    RANDOM_SEARCH_AGENT,
    AGENT_4o,
    AGENT_4o_MINI,
    AGENT_CLAUDE_SONNET_35,
    AGENT_37_SONNET,
    AGENT_CLAUDE_SONNET_35_VISION,
    AGENT_4o_VISION,
    AGENT_4o_MINI_VISION,
    AGENT_o3_MINI,
    AGENT_o1_MINI,
)

__all__ = [
    "AGENT_3_5",
    "AGENT_4o",
    "AGENT_4o_MINI",
    "AGENT_4o_VISION",
    "AGENT_o3_MINI",
    "AGENT_o1_MINI",
    "AGENT_LLAMA4_17B_INSTRUCT",
    "AGENT_LLAMA3_70B",
    "AGENT_LLAMA31_70B",
    "AGENT_8B",
    "RANDOM_SEARCH_AGENT",
    "AGENT_CUSTOM",
    "AGENT_CLAUDE_SONNET_35",
    "AGENT_37_SONNET",
    "AGENT_4o_VISION",
    "AGENT_4o_MINI_VISION",
    "AGENT_CLAUDE_SONNET_35_VISION",
]
