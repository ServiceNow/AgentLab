from dask import compute, delayed
from browsergym.experiments.loop import ExpArgs
from distributed import LocalCluster, Client


def _run(exp_arg: ExpArgs, *dependencies):
    return exp_arg.run()


def make_dask_client(n_worker):
    """Create a Dask client with a LocalCluster backend.

    I struggled to find an appropriate configuration.
    I believe it has to do with the interplay of playwright async loop (even if
    used in sync mode) and the fact that dask uses asyncio under the hood.
    Making sure we use processes and 1 thread per worker seems to work.

    Args:
        n_worker: int
            Number of workers to create.

    Returns:
        A Dask client object.
    """
    cluster = LocalCluster(
        n_workers=n_worker,
        processes=True,
        threads_per_worker=1,
    )

    return Client(cluster)


def execute_task_graph(exp_args_list: list[ExpArgs]):
    """Execute a task graph in parallel while respecting dependencies."""
    exp_args_map = {exp_args.exp_id: exp_args for exp_args in exp_args_list}

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

        exp_args.depends_on = tuple(
            exp_args_map[dep_name].exp_id
            for dep_name in task_dependencies[task_name]
            if dep_name in exp_args_map  # ignore dependencies that are not to be run
        )

    return exp_args_list
