"""Graph dispatch is validated atomically before it can create remote work."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentlab.experiments import graph_execution_ray as execution


def experiment(exp_id, *dependencies, steps=1, episode_timeout=1000):
    return SimpleNamespace(
        exp_id=exp_id,
        exp_name=f"experiment-{exp_id}",
        depends_on=dependencies,
        env_args=SimpleNamespace(max_steps=steps),
        episode_timeout=episode_timeout,
    )


@pytest.fixture
def dispatch(monkeypatch):
    submitted = []
    refs = {}
    deadlines = {}

    class Remote:
        def options(self, **kwargs):
            self.name = kwargs["name"]
            return self

        def remote(self, exp, *dependencies, **kwargs):
            submitted.append((exp.exp_id, dependencies, self.name, kwargs))
            refs[exp.exp_id] = object()
            return refs[exp.exp_id]

    def collect(tasks, timeouts, **kwargs):
        deadlines.update(timeouts)
        return dict(tasks)

    api = SimpleNamespace(submitted=submitted, refs=refs, deadlines=deadlines)
    monkeypatch.setattr(execution, "run_exp", Remote())
    monkeypatch.setattr(execution, "poll_for_timeout", collect)
    monkeypatch.setattr(execution.ray, "cancel", Mock())
    return api


def test_out_of_order_diamond_submits_parent_references_once(dispatch):
    experiments = [
        experiment("finish", "left", "right"),
        experiment("right", "start"),
        experiment("start"),
        experiment("left", "start"),
    ]
    result = execution.execute_task_graph(experiments, avg_step_timeout=3)
    assert [entry[0] for entry in dispatch.submitted] == ["start", "right", "left", "finish"]
    assert dispatch.submitted[-1][1] == (dispatch.refs["left"], dispatch.refs["right"])
    assert dispatch.submitted[1][1] == (dispatch.refs["start"],)
    assert dispatch.submitted[2][1] == (dispatch.refs["start"],)
    assert list(result) == ["finish", "right", "start", "left"]
    assert all(entry[2] == f"experiment-{entry[0]}" for entry in dispatch.submitted)
    assert all(entry[3] == {"avg_step_timeout": 3} for entry in dispatch.submitted)


def test_episode_deadlines_are_not_replaced_by_study_maximum(dispatch):
    execution.execute_task_graph(
        [
            experiment("short", steps=2),
            experiment("long", steps=20),
            experiment("capped", steps=20, episode_timeout=4),
        ],
        avg_step_timeout=3,
    )
    assert dispatch.deadlines == {"short": 6, "long": 60, "capped": 4}


def test_no_max_steps_retains_ten_hour_default(dispatch):
    exp = experiment("default", steps=None, episode_timeout=1e9)
    execution.execute_task_graph([exp])
    assert dispatch.deadlines == {"default": 36000}


@pytest.mark.parametrize(
    "experiments, message",
    [
        ([experiment("a"), experiment("a")], "Duplicate"),
        ([experiment("ok"), experiment("bad", "missing")], "missing"),
        ([experiment("ok"), experiment("a", "b"), experiment("b", "a")], "cycle"),
    ],
)
def test_no_valid_prefix_is_submitted_when_later_component_is_invalid(
    dispatch, experiments, message
):
    with pytest.raises(ValueError, match=message):
        execution.execute_task_graph(experiments)
    assert dispatch.submitted == []


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), None, "1", True])
def test_invalid_episode_deadline_submits_no_work(dispatch, timeout):
    with pytest.raises(ValueError, match="timeout"):
        execution.execute_task_graph([experiment("ok"), experiment("bad", episode_timeout=timeout)])
    assert dispatch.submitted == []


@pytest.mark.parametrize("average", [0, -1, None, "1", True])
def test_invalid_average_submits_no_work(dispatch, average):
    with pytest.raises(ValueError, match="avg_step_timeout"):
        execution.execute_task_graph([experiment("a")], avg_step_timeout=average)
    assert dispatch.submitted == []


def test_empty_graph_returns_empty_mapping(dispatch):
    assert execution.execute_task_graph([]) == {}
    assert dispatch.submitted == []


@pytest.mark.parametrize("failure", [RuntimeError("submission failed"), KeyboardInterrupt()])
def test_partial_submission_failure_cancels_created_work(monkeypatch, failure):
    reference = object()
    remote = Mock()
    remote.options.return_value = remote
    remote.remote.side_effect = [reference, failure]
    cancel = Mock()
    monkeypatch.setattr(execution, "run_exp", remote)
    monkeypatch.setattr(execution.ray, "cancel", cancel)
    with pytest.raises(type(failure)) as raised:
        execution.execute_task_graph([experiment("a"), experiment("b", "a")])
    assert raised.value is failure
    cancel.assert_called_once_with(reference, force=True, recursive=False)


def test_submission_cleanup_does_not_replace_original_failure(monkeypatch):
    remote = Mock()
    remote.options.return_value = remote
    remote.remote.side_effect = [object(), RuntimeError("submission failed")]
    monkeypatch.setattr(execution, "run_exp", remote)
    monkeypatch.setattr(execution.ray, "cancel", Mock(side_effect=RuntimeError("cancel failed")))
    with pytest.raises(RuntimeError, match="submission failed"):
        execution.execute_task_graph([experiment("a"), experiment("b")])


def test_large_reversed_graph_submits_without_recursion(dispatch):
    experiments = [experiment(str(i), str(i + 1)) for i in range(1500)]
    experiments.append(experiment("1500"))
    result = execution.execute_task_graph(experiments)
    assert len(result) == 1501
    assert dispatch.submitted[0][0] == "1500"
    assert dispatch.submitted[-1][0] == "0"
