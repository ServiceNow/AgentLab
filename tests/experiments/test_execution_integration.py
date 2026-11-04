"""Exercise native experiment persistence and real local Ray without a browser."""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import ray

from agentlab.experiments.graph_execution_ray import execute_task_graph
from agentlab.experiments.launch_exp import find_incomplete, run_experiments
from agentlab.experiments.loop import AbstractAgentArgs, EnvArgs, ExpArgs


@dataclass
class LocalAgentArgs(AbstractAgentArgs):
    resource_log: str = None

    def prepare(self):
        with open(self.resource_log, "a") as stream:
            stream.write("prepare\n")

    def close(self):
        with open(self.resource_log, "a") as stream:
            stream.write("close\n")

    def make_agent(self):
        raise AssertionError("The scheduler integration test never creates a browser agent")


@dataclass
class FileExperiment(ExpArgs):
    """Persist scheduler effects using the real ExpArgs preparation contract."""

    marker_dir: str = None
    delay: float = 0
    fail: bool = False
    episode_timeout: float = 20

    def run(self):
        markers = Path(self.marker_dir)
        for dependency in self.depends_on:
            assert (markers / dependency).exists(), f"dependency {dependency} has not finished"
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError(f"experiment {self.exp_id} failed")
        (markers / self.exp_id).write_text("finished")
        if self.exp_dir:
            (Path(self.exp_dir) / "summary_info.json").write_text(
                json.dumps({"terminated": True, "truncated": False, "err_msg": None})
            )
        return self.exp_id


def file_experiment(exp_id, marker_dir, *dependencies, agent=None, **kwargs):
    return FileExperiment(
        agent_args=agent,
        env_args=EnvArgs(task_name=f"offline.{exp_id}", task_seed=1, max_steps=100),
        exp_id=exp_id,
        depends_on=dependencies,
        marker_dir=str(marker_dir),
        **kwargs,
    )


def test_native_exp_args_persist_reordered_dependency_execution(tmp_path):
    markers = tmp_path / "markers"
    markers.mkdir()
    agent = LocalAgentArgs(resource_log=str(tmp_path / "resources.log"))
    experiments = [
        file_experiment("child", markers, "parent", agent=agent),
        file_experiment("parent", markers, agent=agent),
    ]
    run_experiments(1, experiments, tmp_path / "study", parallel_backend="sequential")
    assert agent.resource_log and Path(agent.resource_log).read_text() == "prepare\nclose\n"
    assert {path.name for path in markers.iterdir()} == {"child", "parent"}
    assert all((exp.exp_dir / "exp_args.pkl").exists() for exp in experiments)
    restored = find_incomplete(tmp_path / "study")
    assert len(restored) == 2
    assert all(exp.is_dummy for exp in restored)
    assert {exp.exp_id for exp in restored} == {"child", "parent"}


def test_relaunch_keeps_completed_dependency_as_dummy(tmp_path):
    markers = tmp_path / "markers"
    markers.mkdir()
    agent = LocalAgentArgs(resource_log=str(tmp_path / "resources.log"))
    parent = file_experiment("parent", markers, agent=agent)
    child = file_experiment("child", markers, "parent", agent=agent)
    run_experiments(1, [parent, child], tmp_path / "study", parallel_backend="sequential")
    (child.exp_dir / "summary_info.json").unlink()
    (markers / "child").unlink()
    restored = find_incomplete(tmp_path / "study")
    assert {exp.exp_id: exp.is_dummy for exp in restored} == {"parent": True, "child": False}
    parent_marker_time = (markers / "parent").stat().st_mtime_ns
    # Reversing the persisted launch order must still leave the dummy dependency first.
    restored.sort(key=lambda exp: exp.exp_id)
    run_experiments(1, restored, tmp_path / "study", parallel_backend="sequential")
    assert (markers / "child").exists()
    assert (markers / "parent").stat().st_mtime_ns == parent_marker_time
    assert len(find_incomplete(tmp_path / "study")) == 2


def test_native_exp_args_get_ids_before_directory_preparation(tmp_path):
    markers = tmp_path / "markers"
    markers.mkdir()
    agent = LocalAgentArgs(resource_log=str(tmp_path / "resources.log"))
    exp = file_experiment(None, markers, agent=agent)
    run_experiments(1, [exp], tmp_path / "study", parallel_backend="sequential")
    assert isinstance(exp.exp_id, str)
    assert (markers / exp.exp_id).exists()


@pytest.fixture(scope="module")
def local_ray():
    # Workers must receive the test class by value, not import pytest's module name.
    ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
    ray.init(num_cpus=2, include_dashboard=True, log_to_driver=False)
    try:
        yield
    finally:
        ray.shutdown()
        ray.cloudpickle.unregister_pickle_by_value(sys.modules[__name__])


def test_real_ray_out_of_order_diamond_and_result_identity(tmp_path, local_ray):
    experiments = [
        file_experiment("finish", tmp_path, "left", "right"),
        file_experiment("right", tmp_path, "start", delay=0.05),
        file_experiment("left", tmp_path, "start", delay=0.2),
        file_experiment("start", tmp_path),
    ]
    result = execute_task_graph(experiments)
    assert result == {exp.exp_id: exp.exp_id for exp in experiments}
    assert {path.name for path in tmp_path.iterdir()} == {"finish", "right", "left", "start"}


def test_real_ray_parent_failure_blocks_child_but_not_other_branch(tmp_path, local_ray):
    experiments = [
        file_experiment("child", tmp_path, "failed"),
        file_experiment("failed", tmp_path, fail=True),
        file_experiment("independent", tmp_path),
    ]
    result = execute_task_graph(experiments)
    assert isinstance(result["failed"], Exception)
    assert isinstance(result["child"], Exception)
    assert result["independent"] == "independent"
    assert not (tmp_path / "child").exists()
    assert not (tmp_path / "failed").exists()


def test_real_ray_per_episode_timeout_does_not_inherit_long_sibling_budget(tmp_path, local_ray):
    experiments = [
        file_experiment("short", tmp_path, delay=5, episode_timeout=0.5),
        file_experiment("long", tmp_path, delay=1, episode_timeout=20),
    ]
    started = time.monotonic()
    result = execute_task_graph(experiments)
    assert isinstance(result["short"], Exception)
    assert result["long"] == "long"
    assert not (tmp_path / "short").exists()
    assert time.monotonic() - started < 15


def test_real_ray_dependency_wait_does_not_use_child_runtime_budget(tmp_path, local_ray):
    experiments = [
        file_experiment("child", tmp_path, "parent", delay=0.01, episode_timeout=0.5),
        file_experiment("parent", tmp_path, delay=1.2, episode_timeout=20),
    ]
    assert execute_task_graph(experiments) == {"child": "child", "parent": "parent"}


def test_invalid_native_graph_preserves_existing_experiment_directory(tmp_path):
    markers = tmp_path / "markers"
    markers.mkdir()
    agent = LocalAgentArgs(resource_log=str(tmp_path / "resources.log"))
    exp = file_experiment("existing", markers, agent=agent)
    exp.prepare(tmp_path / "study")
    saved_directory = exp.exp_dir
    (saved_directory / "keep.txt").write_text("original results")
    exp.depends_on = ("missing",)
    with pytest.raises(ValueError, match="missing"):
        run_experiments(1, [exp], tmp_path / "study", parallel_backend="sequential")
    assert (saved_directory / "keep.txt").read_text() == "original results"
    assert not saved_directory.with_name("_" + saved_directory.name).exists()
    assert not Path(agent.resource_log).exists()
