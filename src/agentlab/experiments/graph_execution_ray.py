"""Dependency-aware Ray execution and runtime timeout monitoring."""

import logging
import math
import time
from collections.abc import Mapping
from numbers import Real

import bgym
import ray
from agentlab.experiments.exp_utils import _episode_timeout, run_exp
from agentlab.experiments.task_graph import TaskGraph
from ray.util import state

logger = logging.getLogger(__name__)

run_exp = ray.remote(run_exp)


def _validate_timeout(value, name, allow_zero=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
        or (value == 0 and not allow_zero)
    ):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} number, got {value!r}")
    return float(value)


def _experiment_timeouts(experiments, avg_step_timeout):
    _validate_timeout(avg_step_timeout, "avg_step_timeout")
    timeouts = {}
    for exp in experiments:
        if hasattr(exp, "episode_timeout"):
            _validate_timeout(exp.episode_timeout, f"episode_timeout for {exp.exp_id!r}")
        timeouts[exp.exp_id] = _validate_timeout(
            _episode_timeout(exp, avg_step_timeout), f"timeout for {exp.exp_id!r}"
        )
    return timeouts


def execute_task_graph(exp_args_list: list[bgym.ExpArgs], avg_step_timeout=60):
    """Execute a validated experiment DAG, returning results by experiment ID.

    The complete graph and all deadlines are checked before submitting any work.
    Dependencies are top-level Ray object references, so Ray does not run a child
    until its parents finish. A raised parent exception prevents its descendants
    from running; independent branches are still collected. Exceptions are values
    in the returned mapping, as in ``poll_for_timeout``.

    Args:
        exp_args_list: Experiments with assigned IDs and their dependency IDs.
        avg_step_timeout: Runtime budget per environment step, in seconds.

    Returns:
        A mapping from experiment ID to its result or exception.

    Raises:
        BaseException: If graph validation, submission, or monitoring fails.
    """
    graph = TaskGraph(exp_args_list)
    timeouts = _experiment_timeouts(graph.experiments, avg_step_timeout)
    tasks = {}
    try:
        for exp_arg in graph.ordered_experiments:
            dependencies = [tasks[exp_id] for exp_id in graph.dependencies[exp_arg.exp_id]]
            tasks[exp_arg.exp_id] = run_exp.options(name=f"{exp_arg.exp_name}").remote(
                exp_arg, *dependencies, avg_step_timeout=avg_step_timeout
            )
    except BaseException:
        _cancel_tasks(tasks.values())
        raise

    # Keep the result mapping in caller order, independently of submission order.
    tasks = {exp.exp_id: tasks[exp.exp_id] for exp in graph.experiments}
    poll_interval = min(1.0, min(timeouts.values()) * 0.1) if timeouts else 1.0
    return poll_for_timeout(tasks, timeouts, poll_interval=poll_interval)


def _cancel_tasks(tasks):
    """Best-effort cleanup that never hides an execution or interruption error."""
    for task in tasks:
        try:
            ray.cancel(task, force=True, recursive=False)
        except Exception:
            logger.exception("Could not cancel outstanding experiment task %r", task)


def poll_for_timeout(
    tasks: dict[str, ray.ObjectRef],
    timeout: float | Mapping[str, float],
    poll_interval: float = 1.0,
    *,
    cancel_grace_period: float = 60.0,
):
    """Collect results and cancel tasks that exceed their own running time limit.

    ``timeout`` can be one positive number for every task, or a mapping covering
    exactly the experiment IDs in ``tasks``. Time waiting for dependencies or an
    available worker is excluded. Ray task-state information is best effort; when
    unavailable, no runtime deadline is inferred from submission time.

    A timed-out task receives one cooperative cancellation, then one force kill
    if it is still pending after ``cancel_grace_period`` seconds. The grace period
    uses a monotonic clock and continues even when task-state queries fail. Result
    collection waits for Ray to acknowledge completion/cancellation so dependent
    references reach a terminal state. Ordinary task errors are returned alongside
    successful results. Driver errors and interrupts cancel outstanding work and
    propagate to the caller.

    Args:
        tasks: Mapping of experiment IDs to unique Ray object references.
        timeout: Shared runtime limit or a per-experiment mapping, in seconds.
        poll_interval: Maximum time to wait for each batch of ready references.
        cancel_grace_period: Seconds between cooperative cancellation and force kill.

    Returns:
        A mapping in input order containing each task result or exception.

    Raises:
        ValueError: If deadlines, intervals, or object references are invalid.
        BaseException: If the driver is interrupted or Ray monitoring fails.
    """
    _validate_timeout(poll_interval, "poll_interval")
    _validate_timeout(cancel_grace_period, "cancel_grace_period", allow_zero=True)
    if isinstance(timeout, Mapping):
        if set(timeout) != set(tasks):
            raise ValueError("timeout mapping must cover exactly the submitted experiment IDs")
        timeouts = {
            task_id: _validate_timeout(value, f"timeout for {task_id!r}")
            for task_id, value in timeout.items()
        }
    else:
        limit = _validate_timeout(timeout, "timeout")
        timeouts = dict.fromkeys(tasks, limit)

    pending = {ref: task_id for task_id, ref in tasks.items()}
    if len(pending) != len(tasks):
        raise ValueError("Each experiment must have a distinct Ray object reference")
    results = {}
    cancelled_at = {}
    force_cancelled = set()
    try:
        while pending:
            ready, not_ready = ray.wait(
                list(pending), num_returns=len(pending), timeout=poll_interval
            )
            for task in ready:
                task_id = pending.pop(task)
                try:
                    results[task_id] = ray.get(task)
                except Exception as error:
                    logger.exception("Experiment %s failed", task_id)
                    results[task_id] = error

            for task in not_ready:
                task_id = pending[task]
                if task in cancelled_at:
                    if (
                        task not in force_cancelled
                        and time.monotonic() - cancelled_at[task] >= cancel_grace_period
                    ):
                        logger.warning(
                            "Force killing experiment %s after cancellation grace", task_id
                        )
                        ray.cancel(task, force=True, recursive=False)
                        force_cancelled.add(task)
                    continue
                elapsed_time = get_elapsed_time(task)
                if elapsed_time is not None and elapsed_time > timeouts[task_id]:
                    logger.warning(
                        "Experiment %s exceeded its timeout (%ss > %ss); cancelling",
                        task_id,
                        elapsed_time,
                        timeouts[task_id],
                    )
                    ray.cancel(task, force=False, recursive=False)
                    cancelled_at[task] = time.monotonic()
    except BaseException:
        _cancel_tasks(pending)
        raise
    return {task_id: results[task_id] for task_id in tasks}


def get_elapsed_time(task_ref: ray.ObjectRef):
    """Return seconds since the latest Ray task attempt started, or None."""
    task_id = "<unknown>"
    try:
        task_id = task_ref.task_id().hex()
        task_info = state.get_task(task_id, address="auto")
        if not task_info:
            return None
        attempts = task_info if isinstance(task_info, list) else [task_info]
        latest_attempt = max(attempts, key=lambda task: getattr(task, "attempt_number", 0))
        start_time_ms = getattr(latest_attempt, "start_time_ms", None)
        if start_time_ms is None or start_time_ms <= 0:
            return None
        return max(0, time.time() - start_time_ms / 1000.0)
    except Exception as error:
        logger.warning("Could not get elapsed time for task %s: %s", task_id, error)
        return None
