"""Deterministic tests of Ray result identity, deadlines and cancellation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentlab.experiments import graph_execution_ray as execution


class Ref:
    def __init__(self, name):
        self.name = name

    def task_id(self):
        return self

    def hex(self):
        return self.name


@pytest.fixture
def ray_api(monkeypatch):
    api = SimpleNamespace(wait=Mock(), get=Mock(), cancel=Mock())
    for name, method in vars(api).items():
        monkeypatch.setattr(execution.ray, name, method)
    return api


def test_empty_monitor_does_not_call_ray(ray_api):
    assert execution.poll_for_timeout({}, 5) == {}
    ray_api.wait.assert_not_called()


def test_results_are_keyed_by_reference_instead_of_ready_position(ray_api):
    a, b, c = Ref("a"), Ref("b"), Ref("c")
    ray_api.wait.side_effect = [([b], [a, c]), ([c, a], [])]
    ray_api.get.side_effect = lambda ref: ref.name
    result = execution.poll_for_timeout({"A": a, "B": b, "C": c}, 10)
    assert result == {"A": "a", "B": "b", "C": "c"}
    assert list(result) == ["A", "B", "C"]
    assert ray_api.wait.call_args_list[1].args[0] == [a, c]
    assert ray_api.get.call_count == 3


def test_task_failure_is_returned_without_stopping_independent_results(ray_api):
    a, b = Ref("a"), Ref("b")
    failure = RuntimeError("worker failed")
    ray_api.wait.return_value = ([b, a], [])
    ray_api.get.side_effect = lambda ref: (_ for _ in ()).throw(failure) if ref is a else 12
    result = execution.poll_for_timeout({"a": a, "b": b}, 5)
    assert result["a"] is failure
    assert result["b"] == 12


def test_each_task_uses_its_own_deadline(ray_api, monkeypatch):
    short, long = Ref("short"), Ref("long")
    ray_api.wait.side_effect = [([], [short, long]), ([long, short], [])]
    ray_api.get.side_effect = lambda ref: ref.name
    monkeypatch.setattr(execution, "get_elapsed_time", lambda ref: 15)
    execution.poll_for_timeout({"s": short, "l": long}, {"s": 10, "l": 100})
    ray_api.cancel.assert_called_once_with(short, force=False, recursive=False)


def test_queued_tasks_do_not_consume_runtime_budget(ray_api, monkeypatch):
    queued = Ref("queued")
    ray_api.wait.side_effect = [([], [queued]), ([queued], [])]
    monkeypatch.setattr(execution, "get_elapsed_time", lambda ref: None)
    execution.poll_for_timeout({"q": queued}, 0.01)
    ray_api.cancel.assert_not_called()


def test_timeout_boundary_is_not_cancelled_prematurely(ray_api, monkeypatch):
    task = Ref("task")
    ray_api.wait.side_effect = [([], [task]), ([task], [])]
    monkeypatch.setattr(execution, "get_elapsed_time", lambda ref: 5)
    execution.poll_for_timeout({"task": task}, 5)
    ray_api.cancel.assert_not_called()


def test_cancel_happens_once_then_force_after_grace_even_if_state_disappears(ray_api, monkeypatch):
    task = Ref("task")
    ray_api.wait.side_effect = [([], [task])] * 4 + [([task], [])]
    elapsed = iter([11, None, None, None])
    monkeypatch.setattr(execution, "get_elapsed_time", lambda ref: next(elapsed))
    times = iter([100, 101, 161, 162])
    monkeypatch.setattr(execution.time, "monotonic", lambda: next(times))
    execution.poll_for_timeout({"task": task}, 10, cancel_grace_period=60)
    assert ray_api.cancel.call_args_list == [
        ((task,), {"force": False, "recursive": False}),
        ((task,), {"force": True, "recursive": False}),
    ]


def test_force_cancellation_can_be_immediate(ray_api, monkeypatch):
    task = Ref("task")
    ray_api.wait.side_effect = [([], [task]), ([], [task]), ([task], [])]
    monkeypatch.setattr(execution, "get_elapsed_time", lambda ref: 11)
    execution.poll_for_timeout({"task": task}, 10, cancel_grace_period=0)
    assert [call.kwargs["force"] for call in ray_api.cancel.call_args_list] == [False, True]


def test_interruption_cancels_only_unfinished_tasks(ray_api):
    done, pending = Ref("done"), Ref("pending")
    interruption = KeyboardInterrupt()
    ray_api.wait.side_effect = [([done], [pending]), interruption]
    with pytest.raises(KeyboardInterrupt):
        execution.poll_for_timeout({"done": done, "pending": pending}, 5)
    ray_api.cancel.assert_called_once_with(pending, force=True, recursive=False)


def test_cleanup_continues_when_cancelling_one_reference_fails(ray_api):
    a, b = Ref("a"), Ref("b")
    ray_api.wait.side_effect = RuntimeError("cluster disconnected")
    ray_api.cancel.side_effect = [RuntimeError("already gone"), None]
    with pytest.raises(RuntimeError, match="cluster disconnected"):
        execution.poll_for_timeout({"a": a, "b": b}, 5)
    assert ray_api.cancel.call_count == 2


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), None, "5", True])
def test_invalid_timeout_is_rejected_before_polling(ray_api, value):
    with pytest.raises(ValueError, match="timeout"):
        execution.poll_for_timeout({"a": Ref("a")}, value)
    ray_api.wait.assert_not_called()


@pytest.mark.parametrize("timeouts", [{}, {"other": 5}, {"a": 5, "extra": 1}, {"a": -1}])
def test_timeout_mapping_must_cover_exactly_the_submitted_tasks(ray_api, timeouts):
    with pytest.raises(ValueError, match="timeout"):
        execution.poll_for_timeout({"a": Ref("a")}, timeouts)
    ray_api.wait.assert_not_called()


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_poll_interval_fails_before_polling(ray_api, value):
    with pytest.raises(ValueError, match="poll_interval"):
        execution.poll_for_timeout({"a": Ref("a")}, 5, poll_interval=value)
    ray_api.wait.assert_not_called()


def test_duplicate_references_are_rejected(ray_api):
    ref = Ref("shared")
    with pytest.raises(ValueError, match="reference"):
        execution.poll_for_timeout({"a": ref, "b": ref}, 5)
    ray_api.wait.assert_not_called()


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), "60"])
def test_invalid_cancellation_grace_fails_before_polling(ray_api, value):
    with pytest.raises(ValueError, match="cancel_grace_period"):
        execution.poll_for_timeout({"a": Ref("a")}, 5, cancel_grace_period=value)
    ray_api.wait.assert_not_called()


def test_task_state_lookup_error_is_nonfatal(monkeypatch, caplog):
    monkeypatch.setattr(execution.state, "get_task", Mock(side_effect=RuntimeError("unavailable")))
    assert execution.get_elapsed_time(Ref("task")) is None
    assert "unavailable" in caplog.text


def test_task_id_lookup_error_does_not_mask_original_error(caplog):
    class BrokenRef:
        def task_id(self):
            raise RuntimeError("invalid reference")

    assert execution.get_elapsed_time(BrokenRef()) is None
    assert "invalid reference" in caplog.text


@pytest.mark.parametrize("task_info", [None, [], SimpleNamespace(start_time_ms=None)])
def test_pending_or_missing_task_state_has_no_elapsed_time(monkeypatch, task_info):
    monkeypatch.setattr(execution.state, "get_task", lambda *args, **kwargs: task_info)
    assert execution.get_elapsed_time(Ref("task")) is None


def test_running_task_uses_wall_clock_start_time(monkeypatch):
    monkeypatch.setattr(
        execution.state, "get_task", lambda *args, **kwargs: SimpleNamespace(start_time_ms=12500)
    )
    monkeypatch.setattr(execution.time, "time", lambda: 20)
    assert execution.get_elapsed_time(Ref("task")) == 7.5


def test_latest_retry_start_time_is_used(monkeypatch):
    attempts = [
        SimpleNamespace(start_time_ms=1000, attempt_number=0),
        SimpleNamespace(start_time_ms=15000, attempt_number=1),
    ]
    monkeypatch.setattr(execution.state, "get_task", lambda *args, **kwargs: attempts)
    monkeypatch.setattr(execution.time, "time", lambda: 20)
    assert execution.get_elapsed_time(Ref("task")) == 5


def test_retry_waiting_to_start_does_not_inherit_previous_attempt_deadline(monkeypatch):
    attempts = [
        SimpleNamespace(start_time_ms=1000, attempt_number=0),
        SimpleNamespace(start_time_ms=None, attempt_number=1),
    ]
    monkeypatch.setattr(execution.state, "get_task", lambda *args, **kwargs: attempts)
    assert execution.get_elapsed_time(Ref("task")) is None
