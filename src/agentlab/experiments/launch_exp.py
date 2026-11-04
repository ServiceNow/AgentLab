import logging
import os
from importlib import import_module
from pathlib import Path

import bgym

from agentlab.experiments.exp_utils import run_exp
from agentlab.experiments.loop import ExpArgs, yield_all_exp_results
from agentlab.experiments.task_graph import TaskGraph

RAY_PUBLIC_DASHBOARD = os.environ.get("RAY_PUBLIC_DASHBOARD", "false") == "true"
RAY_DASHBOARD_PORT = os.environ.get("RAY_DASHBOARD_PORT")


def run_experiments(
    n_jobs,
    exp_args_list: list[ExpArgs],
    study_dir,
    parallel_backend="ray",
    avg_step_timeout=60,
):
    """Run experiments with validated dependencies and managed agent resources.

    Ray and sequential execution support ``ExpArgs.depends_on`` regardless of
    input order. Joblib supports independent experiments only; a dependency graph
    is rejected before preparing agents or creating experiment directories.
    Shared agent argument objects are prepared and closed once per invocation.

    Args:
        n_jobs: Number of parallel workers (ignored for sequential execution).
        exp_args_list: Experiments to prepare and run.
        study_dir: Directory where experiments will be saved.
        parallel_backend: One of "joblib", "ray", or "sequential".
        avg_step_timeout: Ray runtime budget per environment step. Each episode
            uses its own max_steps and optional episode_timeout cap. Other
            backends do not enforce episode timeouts.

    Raises:
        ValueError: If the backend, dependency graph, or Ray deadlines are invalid.
        BaseException: If preparation, execution, or resource cleanup fails.
    """
    if not exp_args_list:
        logging.warning("No experiments to run.")
        return
    if parallel_backend not in ("joblib", "ray", "sequential"):
        raise ValueError(f"Unknown parallel_backend: {parallel_backend}")

    # Assign identities without preparing directories or starting model servers.
    for exp_args in exp_args_list:
        if exp_args.exp_id is None:
            exp_args.make_id()
    graph = TaskGraph(exp_args_list)
    if parallel_backend == "joblib" and graph.has_dependencies:
        raise ValueError("joblib does not support experiment dependencies; use ray or sequential")
    if parallel_backend == "ray":
        from agentlab.experiments.graph_execution_ray import _experiment_timeouts

        _experiment_timeouts(graph.experiments, avg_step_timeout)

    study_dir = Path(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving experiments to {study_dir}")
    agents = []
    agent_ids = set()
    primary_error = None
    try:
        for exp_args in graph.experiments:
            agent = exp_args.agent_args
            if id(agent) not in agent_ids:
                agent_ids.add(id(agent))
                # Register before preparation: a failing prepare may already
                # have started a server and still needs its matching close.
                agents.append(agent)
                agent.prepare()
            exp_args.prepare(exp_root=study_dir)

        if parallel_backend == "joblib":
            from joblib import Parallel, delayed

            sequential, parallel = _split_sequential_exp(graph.experiments)
            for exp_args in sequential:
                run_exp(exp_args, avg_step_timeout=avg_step_timeout)
            Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(run_exp)(exp_args, avg_step_timeout=avg_step_timeout)
                for exp_args in parallel
            )
        elif parallel_backend == "ray":
            _run_ray_experiments(n_jobs, graph.experiments, avg_step_timeout)
        else:
            for exp_args in graph.ordered_experiments:
                run_exp(exp_args, avg_step_timeout=avg_step_timeout)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        logging.info("Closing prepared agent resources...")
        _cleanup_resources([agent.close for agent in reversed(agents)], primary_error)
        logging.info(f"Experiment finished and saved in {study_dir}.")


def _cleanup_resources(callbacks, primary_error):
    """Attempt all releases, preserving the original failure when there is one."""
    errors = []
    for close in callbacks:
        try:
            close()
        except BaseException as error:
            logging.exception("Experiment resource cleanup failed")
            errors.append(error)
    if primary_error is not None:
        for error in errors:
            primary_error.add_note(f"Resource cleanup also failed: {error!r}")
    elif len(errors) == 1:
        raise errors[0]
    elif errors:
        raise BaseExceptionGroup("Experiment resource cleanup failed", errors)


def _run_ray_experiments(n_jobs, experiments, avg_step_timeout):
    from agentlab.experiments.graph_execution_ray import execute_task_graph, ray

    # shutdown() also cleans partially initialized local processes. Never take
    # ownership of a runtime that existed before this launcher was called.
    owns_runtime = not ray.is_initialized()
    primary_error = None
    try:
        ray.init(
            num_cpus=n_jobs,
            dashboard_host="0.0.0.0" if RAY_PUBLIC_DASHBOARD else "127.0.0.1",
            dashboard_port=None if RAY_DASHBOARD_PORT is None else int(RAY_DASHBOARD_PORT),
        )
        execute_task_graph(experiments, avg_step_timeout=avg_step_timeout)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if owns_runtime:
            _cleanup_resources([ray.shutdown], primary_error)


def find_incomplete(study_dir: str | Path, include_errors=True):
    """Find all incomplete experiments for relaunching.

    Note: completed experiments are kept but are replaced by dummy exp_args
    with nothing to run. This help keeping the dependencies between tasks.

    Args:
        study_dir: Path
            The directory where the experiments are saved.
        include_errors: str
            Find all incomplete experiments and relaunch them.
            - "incomplete_only": relaunch only the incomplete experiments.
            - "incomplete_or_error": relaunch incomplete or errors.

    Returns:
        list[ExpArgs]
            List of ExpArgs objects to relaunch.

    Raises:
        ValueError: If the study_dir does not exist.
    """
    study_dir = Path(study_dir)

    if not study_dir.exists():
        raise ValueError(
            f"You asked to relaunch an existing experiment but {study_dir} does not exist."
        )

    exp_result_list = list(yield_all_exp_results(study_dir, progress_fn=None))
    exp_args_list = [_hide_completed(exp_result, include_errors) for exp_result in exp_result_list]
    # sort according to exp_args.order
    exp_args_list.sort(key=lambda exp_args: exp_args.order if exp_args.order is not None else 0)

    job_count = non_dummy_count(exp_args_list)

    if job_count == 0:
        logging.info(f"No incomplete experiments found in {study_dir}.")
        return exp_args_list
    else:
        logging.info(f"Found {job_count} incomplete experiments in {study_dir}.")

    message = "Make sure the processes that were running are all stopped. Otherwise, "
    "there will be concurrent writing in the same directories.\n"

    logging.info(message)

    return exp_args_list


def non_dummy_count(exp_args_list: list[ExpArgs]) -> int:
    return sum([not exp_args.is_dummy for exp_args in exp_args_list])


def noop(*args, **kwargs):
    pass


def _hide_completed(exp_result: bgym.ExpResult, include_errors: bool = True):
    """Hide completed experiments from the list.

    This little hack, allows an elegant way to keep the task dependencies for e.g. webarena
    while skipping the tasks that are completed when relaunching.

    Args:
        exp_result: bgym.ExpResult
            The experiment result to hide.
        include_errors: bool
            If True, include experiments that errored.

    Returns:
        ExpArgs
            The ExpArgs object hidden if the experiment is completed.
    """

    hide = False
    if exp_result.status == "done":
        hide = True
    if exp_result.status == "error" and (not include_errors):
        hide = True

    exp_args = exp_result.exp_args
    exp_args.is_dummy = hide  # just to keep track
    exp_args.status = exp_result.status
    if hide:
        # make those function do nothing since they are finished.
        exp_args.run = noop
        exp_args.prepare = noop

    return exp_args


# TODO remove this function once ray backend is stable
def _split_sequential_exp(exp_args_list: list[ExpArgs]) -> tuple[list[ExpArgs], list[ExpArgs]]:
    """split exp_args that are flagged as sequential from those that are not"""
    sequential_exp_args = []
    parallel_exp_args = []
    for exp_args in exp_args_list:
        if getattr(exp_args, "sequential", False):
            sequential_exp_args.append(exp_args)
        else:
            parallel_exp_args.append(exp_args)

    return sequential_exp_args, parallel_exp_args


def _split_path(path: str):
    """Split a path into a module name and an object name."""
    if "/" in path:
        path = path.replace("/", ".")
    module_name, obj_name = path.rsplit(".", 1)
    return module_name, obj_name


def import_object(path: str):
    module_name, obj_name = _split_path(path)
    try:
        module = import_module(module_name)
        obj = getattr(module, obj_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Error importing {path}: {e}")
    return obj
