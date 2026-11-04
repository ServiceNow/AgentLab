"""Provider-free regressions for dependency execution and resource ownership."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from agentlab.experiments import graph_execution_ray as graph_ray
from agentlab.experiments import launch_exp


@dataclass
class AgentResources:
    events: list
    name: str = "agent"
    prepare_error: BaseException = None
    close_error: BaseException = None

    def prepare(self):
        self.events.append(("prepare", self.name))
        if self.prepare_error:
            raise self.prepare_error

    def close(self):
        self.events.append(("close", self.name))
        if self.close_error:
            raise self.close_error


@dataclass
class Experiment:
    exp_id: str
    depends_on: tuple = ()
    agent_args: AgentResources = None
    events: list = field(default_factory=list)
    run_error: BaseException = None
    prepare_error: BaseException = None
    env_args: object = field(default_factory=lambda: SimpleNamespace(max_steps=1))

    @property
    def exp_name(self):
        return self.exp_id

    def prepare(self, exp_root):
        self.events.append(("directory", self.exp_id))
        if self.prepare_error:
            raise self.prepare_error

    def run(self):
        self.events.append(("run", self.exp_id))
        if self.run_error:
            raise self.run_error
        return self.exp_id


def test_sequential_backend_honors_reversed_dependency_order(tmp_path):
    events = []
    agent = AgentResources(events)
    child = Experiment("child", ("parent",), agent, events)
    parent = Experiment("parent", agent_args=agent, events=events)
    launch_exp.run_experiments(1, [child, parent], tmp_path, parallel_backend="sequential")
    assert [event for event in events if event[0] == "run"] == [
        ("run", "parent"),
        ("run", "child"),
    ]


def test_shared_agent_has_one_resource_lifecycle(tmp_path):
    events = []
    agent = AgentResources(events)
    experiments = [Experiment(str(i), agent_args=agent, events=events) for i in range(3)]
    launch_exp.run_experiments(1, experiments, tmp_path, parallel_backend="sequential")
    assert events.count(("prepare", "agent")) == 1
    assert events.count(("close", "agent")) == 1


def test_partial_preparation_closes_every_acquired_agent(tmp_path):
    events = []
    first = AgentResources(events, "first")
    second = AgentResources(events, "second", prepare_error=RuntimeError("acquisition failed"))
    experiments = [Experiment("a", agent_args=first), Experiment("b", agent_args=second)]
    with pytest.raises(RuntimeError, match="acquisition failed"):
        launch_exp.run_experiments(1, experiments, tmp_path, parallel_backend="sequential")
    assert ("close", "first") in events
    assert ("close", "second") in events


def test_cleanup_failure_does_not_skip_other_agents(tmp_path):
    events = []
    agents = [
        AgentResources(events, "first", close_error=RuntimeError("close failed")),
        AgentResources(events, "second"),
    ]
    with pytest.raises(Exception, match="close failed"):
        launch_exp.run_experiments(
            1,
            [Experiment(str(i), agent_args=agent) for i, agent in enumerate(agents)],
            tmp_path,
            parallel_backend="sequential",
        )
    assert ("close", "second") in events


def test_ray_ready_order_does_not_change_experiment_identity(monkeypatch):
    first, second = object(), object()
    monkeypatch.setattr(graph_ray.ray, "wait", lambda *args, **kwargs: ([second, first], []))
    monkeypatch.setattr(graph_ray.ray, "get", lambda ref: "first" if ref is first else "second")
    assert graph_ray.poll_for_timeout({"a": first, "b": second}, 10) == {
        "a": "first",
        "b": "second",
    }


def test_invalid_graph_submits_nothing(monkeypatch):
    submitted = []

    class Remote:
        def options(self, **kwargs):
            return self

        def remote(self, exp, *args, **kwargs):
            submitted.append(exp.exp_id)
            return object()

    monkeypatch.setattr(graph_ray, "run_exp", Remote())
    with pytest.raises(ValueError, match="missing"):
        graph_ray.execute_task_graph([Experiment("valid"), Experiment("bad", ("missing",))])
    assert submitted == []


@pytest.mark.parametrize("backend", ["sequential", "ray", "joblib"])
def test_unknown_dependency_fails_before_resource_or_directory_side_effects(tmp_path, backend):
    events = []
    agent = AgentResources(events)
    exp = Experiment("child", ("missing",), agent, events)
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="missing"):
        launch_exp.run_experiments(1, [exp], destination, parallel_backend=backend)
    assert events == []
    assert not destination.exists()


def test_unknown_backend_does_not_prepare_agents(tmp_path):
    events = []
    exp = Experiment("a", agent_args=AgentResources(events), events=events)
    with pytest.raises(ValueError, match="Unknown parallel_backend"):
        launch_exp.run_experiments(1, [exp], tmp_path, parallel_backend="dask")
    assert events == []


def test_joblib_rejects_dependencies_before_preparation(tmp_path):
    events = []
    agent = AgentResources(events)
    experiments = [Experiment("a", agent_args=agent), Experiment("b", ("a",), agent)]
    with pytest.raises(ValueError, match="joblib.*dependencies"):
        launch_exp.run_experiments(2, experiments, tmp_path, parallel_backend="joblib")
    assert events == []


def test_failed_experiment_preparation_closes_agents(tmp_path):
    events = []
    agent = AgentResources(events)
    exp = Experiment("a", agent_args=agent, prepare_error=OSError("directory unavailable"))
    with pytest.raises(OSError, match="directory unavailable"):
        launch_exp.run_experiments(1, [exp], tmp_path, parallel_backend="sequential")
    assert events == [("prepare", "agent"), ("close", "agent")]


def test_agents_not_reached_during_preparation_are_not_closed(tmp_path):
    events = []
    first = AgentResources(events, "first", prepare_error=RuntimeError("prepare failed"))
    second = AgentResources(events, "second")
    with pytest.raises(RuntimeError, match="prepare failed"):
        launch_exp.run_experiments(
            1,
            [Experiment("a", agent_args=first), Experiment("b", agent_args=second)],
            tmp_path,
            parallel_backend="sequential",
        )
    assert events == [("prepare", "first"), ("close", "first")]


def test_primary_run_error_survives_cleanup_error(tmp_path):
    events = []
    agent = AgentResources(events, close_error=RuntimeError("cleanup failed"))
    failure = ValueError("execution failed")
    exp = Experiment("a", agent_args=agent, run_error=failure)
    with pytest.raises(ValueError, match="execution failed") as error:
        launch_exp.run_experiments(1, [exp], tmp_path, parallel_backend="sequential")
    assert error.value is failure
    assert any("cleanup failed" in note for note in error.value.__notes__)


def test_all_cleanup_failures_are_reported(tmp_path):
    events = []
    agents = [
        AgentResources(events, "first", close_error=RuntimeError("first close failed")),
        AgentResources(events, "second", close_error=OSError("second close failed")),
    ]
    with pytest.raises(ExceptionGroup) as error:
        launch_exp.run_experiments(
            1,
            [Experiment(str(i), agent_args=agent) for i, agent in enumerate(agents)],
            tmp_path,
            parallel_backend="sequential",
        )
    assert {str(exc) for exc in error.value.exceptions} == {
        "first close failed",
        "second close failed",
    }
    assert len([event for event in events if event[0] == "close"]) == 2


def test_keyboard_interrupt_during_prepare_closes_partial_agent(tmp_path):
    events = []
    agent = AgentResources(events, prepare_error=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        launch_exp.run_experiments(
            1, [Experiment("a", agent_args=agent)], tmp_path, parallel_backend="sequential"
        )
    assert ("close", "agent") in events


def test_keyboard_interrupt_during_run_is_not_masked_by_close(tmp_path):
    events = []
    agent = AgentResources(events, close_error=RuntimeError("cleanup failed"))
    with pytest.raises(KeyboardInterrupt):
        launch_exp.run_experiments(
            1,
            [Experiment("a", agent_args=agent, run_error=KeyboardInterrupt())],
            tmp_path,
            parallel_backend="sequential",
        )
    assert ("close", "agent") in events


def test_empty_study_has_no_directory_side_effect(tmp_path):
    destination = tmp_path / "empty"
    assert launch_exp.run_experiments(1, [], destination) is None
    assert not destination.exists()


def test_distinct_equal_agent_objects_have_separate_lifecycles(tmp_path):
    events = []
    first, second = AgentResources(events), AgentResources(events)
    assert first == second
    launch_exp.run_experiments(
        1,
        [Experiment("a", agent_args=first), Experiment("b", agent_args=second)],
        tmp_path,
        parallel_backend="sequential",
    )
    assert events.count(("prepare", "agent")) == 2
    assert events.count(("close", "agent")) == 2


def test_joblib_sequential_partition_does_not_lose_cleanup_ownership(tmp_path, monkeypatch):
    import joblib

    events = []
    first, second = AgentResources(events, "first"), AgentResources(events, "second")
    sequential = Experiment("a", agent_args=first)
    sequential.sequential = True
    parallel = Experiment("b", agent_args=second)

    class InlineParallel:
        def __init__(self, **kwargs):
            pass

        def __call__(self, jobs):
            return [fn(*args, **kwargs) for fn, args, kwargs in jobs]

    monkeypatch.setattr(joblib, "Parallel", InlineParallel)
    launch_exp.run_experiments(2, [sequential, parallel], tmp_path, parallel_backend="joblib")
    assert ("close", "first") in events
    assert ("close", "second") in events


@pytest.fixture
def ray_launcher(monkeypatch):
    from unittest.mock import Mock

    init, shutdown, execute = Mock(), Mock(), Mock()
    initialized = Mock(return_value=False)
    monkeypatch.setattr(graph_ray.ray, "is_initialized", initialized)
    monkeypatch.setattr(graph_ray.ray, "init", init)
    monkeypatch.setattr(graph_ray.ray, "shutdown", shutdown)
    monkeypatch.setattr(graph_ray, "execute_task_graph", execute)
    return SimpleNamespace(init=init, shutdown=shutdown, execute=execute, initialized=initialized)


def test_ray_startup_failure_still_closes_agents(tmp_path, ray_launcher):
    events = []
    ray_launcher.init.side_effect = RuntimeError("startup failed")
    with pytest.raises(RuntimeError, match="startup failed"):
        launch_exp.run_experiments(
            2, [Experiment("a", agent_args=AgentResources(events))], tmp_path
        )
    assert ("close", "agent") in events
    ray_launcher.execute.assert_not_called()
    ray_launcher.shutdown.assert_called_once()


def test_ray_run_error_survives_shutdown_and_agent_cleanup_failures(tmp_path, ray_launcher):
    events = []
    failure = RuntimeError("graph failed")
    ray_launcher.execute.side_effect = failure
    ray_launcher.shutdown.side_effect = OSError("shutdown failed")
    agent = AgentResources(events, close_error=RuntimeError("agent cleanup failed"))
    with pytest.raises(RuntimeError, match="graph failed") as error:
        launch_exp.run_experiments(2, [Experiment("a", agent_args=agent)], tmp_path)
    assert error.value is failure
    assert any("shutdown failed" in note for note in failure.__notes__)
    assert any("agent cleanup failed" in note for note in failure.__notes__)
    assert ("close", "agent") in events
    ray_launcher.shutdown.assert_called_once()


def test_ray_shutdown_failure_is_reported_after_successful_run(tmp_path, ray_launcher):
    events = []
    ray_launcher.shutdown.side_effect = OSError("shutdown failed")
    with pytest.raises(OSError, match="shutdown failed"):
        launch_exp.run_experiments(
            2, [Experiment("a", agent_args=AgentResources(events))], tmp_path
        )
    assert ("close", "agent") in events


def test_invalid_ray_deadline_does_not_start_agents_or_cluster(tmp_path, ray_launcher):
    events = []
    exp = Experiment("a", agent_args=AgentResources(events))
    with pytest.raises(ValueError, match="avg_step_timeout"):
        launch_exp.run_experiments(2, [exp], tmp_path, avg_step_timeout=0)
    assert events == []
    ray_launcher.init.assert_not_called()


def test_preexisting_ray_runtime_is_not_closed_on_reinitialization_failure(tmp_path, ray_launcher):
    events = []
    ray_launcher.initialized.return_value = True
    ray_launcher.init.side_effect = RuntimeError("already initialized")
    with pytest.raises(RuntimeError, match="already initialized"):
        launch_exp.run_experiments(
            2, [Experiment("a", agent_args=AgentResources(events))], tmp_path
        )
    ray_launcher.shutdown.assert_not_called()
    assert ("close", "agent") in events


def test_partial_ray_startup_error_survives_shutdown_failure(tmp_path, ray_launcher):
    events = []
    failure = RuntimeError("partial startup failed")
    ray_launcher.init.side_effect = failure
    ray_launcher.shutdown.side_effect = OSError("partial shutdown failed")
    with pytest.raises(RuntimeError, match="partial startup failed") as error:
        launch_exp.run_experiments(
            2, [Experiment("a", agent_args=AgentResources(events))], tmp_path
        )
    assert error.value is failure
    ray_launcher.shutdown.assert_called_once()
    assert any("partial shutdown failed" in note for note in failure.__notes__)
    assert ("close", "agent") in events
