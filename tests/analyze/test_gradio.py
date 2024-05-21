from agentlab.analyze import agent_xray


def test_convert_to_markdown():
    md = agent_xray.convert_to_markdown({"a": 1, "b": 2, "c": 3})
    assert "a" in md and "b" in md and "c" in md
