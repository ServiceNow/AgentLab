from dask import compute, delayed
from dask.distributed import Client
from browsergym.experiments.loop import ExpArgs
import logging


def run_experiments(n_workers, exp_args_list: list[ExpArgs], exp_dir):
    """Run a list of experiments in parallel while respecting dependencies."""

    logging.info(f"Saving experiments to {exp_dir}")
    for exp_args in exp_args_list:
        exp_args.agent_args.prepare()
        exp_args.prepare(exp_root=exp_dir)

    try:
        execute_task_graph(Client(n_workers=n_workers), exp_args_list)
    finally:
        logging.info("All jobs are finished. Calling agent_args.close() on all agents...")
        for exp_args in exp_args_list:
            exp_args.agent_args.close()
        logging.info("Experiment finished.")


def _run(exp_arg: ExpArgs, *dependencies):
    """Capture dependencies to ensure they are run before the current task."""
    return exp_arg.run()


def execute_task_graph(dask_client, exp_args_list: list[ExpArgs]):
    """Execute a task graph in parallel while respecting dependencies."""
    exp_args_map = {exp_args.exp_id: exp_args for exp_args in exp_args_list}

    with dask_client:
        tasks = {}

        def get_task(exp_arg: ExpArgs):
            if exp_arg.exp_id not in tasks:
                dependencies = [get_task(exp_args_map[dep_key]) for dep_key in exp_arg.depends_on]
                tasks[exp_arg.exp_id] = delayed(_run)(exp_arg, *dependencies)
            return tasks[exp_arg.exp_id]

        for exp_arg in exp_args_list:
            get_task(exp_arg)

        task_ids, task_list = zip(*tasks.items())
        results = compute(*task_list)

    return {task_id: result for task_id, result in zip(task_ids, results)}


def add_dependencies(exp_args_list: list[ExpArgs], task_dependencies: dict[list] = None):
    """Add dependencies to a list of ExpArgs.

    Args:
        exp_args_list: list[ExpArgs]
            A list of experiments to run.
        task_dependencies: dict
            A dictionary mapping task names to a list of task names that they
            depend on. If None or empty, no dependencies are added.

    Returns:
        list[ExpArgs]
            The modified exp_args_list with dependencies added.
    """

    if task_dependencies is None or all([len(dep) == 0 for dep in task_dependencies.values()]):
        # nothing to be done
        return exp_args_list

    exp_args_map = {exp_args.env_args.task_name: exp_args for exp_args in exp_args_list}
    if len(exp_args_map) != len(exp_args_list):
        raise ValueError(
            (
                "Task names are not unique in exp_args_map, "
                "you can't run multiple seeds with task dependencies."
            )
        )

    for task_name in exp_args_map.keys():
        if task_name not in task_dependencies:
            raise ValueError(f"Task {task_name} is missing from task_dependencies")

    # turn dependencies from task names to exp_ids
    for task_name, exp_args in exp_args_map.items():

        exp_args.depends_on = [
            exp_args_map[dep_name].exp_id
            for dep_name in task_dependencies[task_name]
            if dep_name in exp_args_map  # ignore dependencies that are not to be run
        ]

    return exp_args_list
