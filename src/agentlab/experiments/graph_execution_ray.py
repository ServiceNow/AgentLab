# import os

# # Disable Ray log deduplication
# os.environ["RAY_DEDUP_LOGS"] = "0"

import ray
import bgym
from agentlab.experiments.exp_utils import run_exp


run_exp = ray.remote(run_exp)


def execute_task_graph(exp_args_list: list[bgym.ExpArgs], avg_step_timeout=30):
    """Execute a task graph in parallel while respecting dependencies using Ray."""

    exp_args_map = {exp_args.exp_id: exp_args for exp_args in exp_args_list}
    tasks = {}

    def get_task(exp_arg: bgym.ExpArgs):
        if exp_arg.exp_id not in tasks:
            # Get all dependency tasks first
            dependency_tasks = [get_task(exp_args_map[dep_key]) for dep_key in exp_arg.depends_on]

            # Create new task that depends on the dependency results
            tasks[exp_arg.exp_id] = run_exp.remote(
                exp_arg, *dependency_tasks, avg_step_timeout=avg_step_timeout
            )
        return tasks[exp_arg.exp_id]

    # Build task graph
    for exp_arg in exp_args_list:
        get_task(exp_arg)

    # Execute all tasks and gather results
    task_ids = list(tasks.keys())
    results = ray.get(list(tasks.values()))

    return {task_id: result for task_id, result in zip(task_ids, results)}
