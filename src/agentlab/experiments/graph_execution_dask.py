from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from contextlib import contextmanager
import threading
from dask import compute, delayed
from bgym import ExpArgs
from distributed import LocalCluster, Client
from agentlab.experiments.exp_utils import _episode_timeout

# from agentlab.experiments.exp_utils import run_exp


def run_exp(exp_arg: ExpArgs, *dependencies, avg_step_timeout=60):
    """Run exp_args.run() with a timeout and handle dependencies."""
    # dask can't use the timeout_manager define in exp_utils.py
    # ValueError: signal only works in main thread of the main interpreter
    # most alternative I try doesn't work
    episode_timeout = _episode_timeout(exp_arg, avg_step_timeout=avg_step_timeout)
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
            tasks[exp_arg.exp_id] = delayed(run_exp)(exp_arg, *dependencies)
        return tasks[exp_arg.exp_id]

    for exp_arg in exp_args_list:
        get_task(exp_arg)

    task_ids, task_list = zip(*tasks.items())
    results = compute(*task_list)

    return {task_id: result for task_id, result in zip(task_ids, results)}
