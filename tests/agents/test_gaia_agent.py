from agentlab.agents.tapeagent import TapeAgent, TapeAgentArgs


def test_agent_creation():
    args = TapeAgentArgs(agent_name="gaia_agent")
    agent = args.make_agent()
    assert isinstance(agent, TapeAgent)
    assert agent.agent.name == "gaia_agent"
