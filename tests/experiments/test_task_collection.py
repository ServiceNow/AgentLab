import pytest

from agentlab.experiments.task_collections import get_benchmark_env_args


@pytest.mark.pricy
@pytest.mark.parametrize(
    "benchmark_name, expected_length",
    [
        ("workarena.l1", 330),
        ("workarena.l2", 235),
        ("workarena.l3", 235),
        ("webarena", 812),
        ("miniwob", 625),
    ],
)
def test_get_benchmark_env_args(benchmark_name, expected_length):
    result = get_benchmark_env_args(benchmark_name)
    assert len(result) == expected_length


if __name__ == "__main__":
    test_get_benchmark_env_args("workarena.l1", 5)
    test_get_benchmark_env_args("workarena.l2", 5)
    test_get_benchmark_env_args("workarena.l3", 5)
    test_get_benchmark_env_args("webarena", 5)
    test_get_benchmark_env_args("miniwob", 5)
