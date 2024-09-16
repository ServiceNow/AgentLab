from agentlab.experiments import reproducibility_util
from agentlab.agents.generic_agent import AGENT_4o_MINI
import pytest
import json


def test_set_temp():
    agent_args = reproducibility_util.set_temp(AGENT_4o_MINI)
    assert agent_args.chat_model_args.temperature == 0


@pytest.mark.parametrize(
    "benchmark_name",
    ["miniwob", "workarena.l1", "webarena", "visualwebarena"],
)
def test_get_reproducibility_info(benchmark_name):
    info = reproducibility_util.get_reproducibility_info(benchmark_name, ignore_changes=True)

    print("reproducibility info:")
    print(json.dumps(info, indent=4))

    # assert keys in info
    assert "git_user" in info
    assert "benchmark" in info
    assert "benchmark_version" in info
    assert "agentlab_version" in info
    assert "agentlab_git_hash" in info
    assert "agentlab__local_modifications" in info
    assert "browsergym_version" in info
    assert "browsergym_git_hash" in info
    assert "browsergym__local_modifications" in info


if __name__ == "__main__":
    # test_set_temp()
    for benchmark_name in [
        "miniwob",
        "workarena.l1",
        "webarena",
    ]:
        test_get_reproducibility_info(benchmark_name)
