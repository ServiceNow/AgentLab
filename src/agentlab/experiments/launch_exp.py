import logging
from importlib import import_module
from pathlib import Path

from browsergym.experiments.loop import ExpArgs, yield_all_exp_results


def import_object(path: str):
    module_name, obj_name = split_path(path)
    try:
        module = import_module(module_name)
        obj = getattr(module, obj_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Error importing {path}: {e}")
    return obj


def run_experiments(
    n_jobs,
    exp_args_list: list[ExpArgs],
    study_dir,
    parallel_backend="joblib",
):
    """Run a list of ExpArgs in parallel.

    To ensure optimal parallelism, make sure ExpArgs.depend_on is set correctly
    and the backend is set to dask.

    Args:
        n_jobs: int
            Number of parallel jobs.
        exp_args_list: list[ExpArgs]
            List of ExpArgs objects.
        exp_dir: Path
            Directory where the experiments will be saved.
        parallel_backend: str
            Parallel backend to use. Either "joblib", "dask" or "sequential".
    """

    if len(exp_args_list) == 0:
        logging.warning("No experiments to run.")
        return

    study_dir = Path(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)

    if n_jobs == 1 and parallel_backend != "sequential":
        logging.warning("Only 1 job, switching to sequential backend.")
        parallel_backend = "sequential"

    logging.info(f"Saving experiments to {study_dir}")
    for exp_args in exp_args_list:
        exp_args.agent_args.prepare()
        exp_args.prepare(exp_root=study_dir)
    try:
        if parallel_backend == "joblib":
            from joblib import Parallel, delayed

            Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(exp_args.run)() for exp_args in exp_args_list
            )

        elif parallel_backend == "dask":
            from agentlab.experiments.graph_execution import execute_task_graph, make_dask_client

            with make_dask_client(n_worker=n_jobs):
                execute_task_graph(exp_args_list)
        elif parallel_backend == "sequential":
            for exp_args in exp_args_list:
                exp_args.run()
        else:
            raise ValueError(f"Unknown parallel_backend: {parallel_backend}")
    finally:
        # will close servers even if there is an exception or ctrl+c
        # servers won't be closed if the script is killed with kill -9 or segfaults.
        logging.info("All jobs are finished. Calling agent_args.close() on all agents...")
        for exp_args in exp_args_list:
            exp_args.agent_args.close()
        logging.info("Experiment finished.")


def relaunch_study(study_dir: str | Path, relaunch_mode="incomplete_only"):
    """Return exp_args_list and study_dir

    Args:
        study_dir: Path
            The directory where the experiments are saved.
        relaunch_mode: str
            Find all incomplete experiments and relaunch them.
            - "incomplete_only": relaunch only the incomplete experiments.
            - "incomplete_or_error": relaunch incomplete or errors.
    """
    study_dir = Path(study_dir)

    if not study_dir.exists():
        raise ValueError(
            f"You asked to relaunch an existing experiment but {study_dir} does not exist."
        )
    exp_args_list = list(_yield_incomplete_experiments(study_dir, relaunch_mode=relaunch_mode))

    if len(exp_args_list) == 0:
        logging.info(f"No incomplete experiments found in {study_dir}.")
        return [], study_dir

    message = f"Make sure the processes that were running are all stopped. Otherwise, "
    f"there will be concurrent writing in the same directories.\n"

    logging.info(message)

    return exp_args_list, study_dir


def _yield_incomplete_experiments(exp_root, relaunch_mode="incomplete_only"):
    """Find all incomplete experiments and relaunch them."""
    # TODO(make relanch_mode a callable, for flexibility)
    for exp_result in yield_all_exp_results(exp_root, progress_fn=None):  # type: ExpArgs
        try:
            # TODO  implement has_finished instead of dealing with FileNotFoundError
            summary_info = exp_result.summary_info

        except FileNotFoundError:
            yield exp_result.exp_args
            continue

        if relaunch_mode == "incomplete_only":
            continue

        err_msg = summary_info.get("err_msg", None)

        if err_msg is not None:
            if relaunch_mode == "incomplete_or_error":
                yield exp_result.exp_args
            else:
                raise ValueError(f"Unknown relaunch_mode: {relaunch_mode}")


def split_path(path: str):
    """Split a path into a module name and an object name."""
    if "/" in path:
        path = path.replace("/", ".")
    module_name, obj_name = path.rsplit(".", 1)
    return module_name, obj_name
