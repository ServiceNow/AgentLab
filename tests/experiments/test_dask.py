from agentlab.experiments.graph_execution_dask import execute_task_graph, make_dask_client
from agentlab.experiments.exp_utils import MockedExpArgs

TASK_TIME = 3


def test_execute_task_graph():
    # Define a list of ExpArgs with dependencies
    exp_args_list = [
        MockedExpArgs(exp_id="task1", depends_on=[]),
        MockedExpArgs(exp_id="task2", depends_on=["task1"]),
        MockedExpArgs(exp_id="task3", depends_on=["task1"]),
        MockedExpArgs(exp_id="task4", depends_on=["task2", "task3"]),
    ]

    with make_dask_client(n_worker=5):
        results = execute_task_graph(exp_args_list)

    exp_args_list = [results[task_id] for task_id in ["task1", "task2", "task3", "task4"]]

    # Verify that all tasks were executed in the proper order
    assert exp_args_list[0].start_time < exp_args_list[1].start_time
    assert exp_args_list[0].start_time < exp_args_list[2].start_time
    assert exp_args_list[1].end_time < exp_args_list[3].start_time
    assert exp_args_list[2].end_time < exp_args_list[3].start_time

    # # Verify that parallel tasks (task2 and task3) started within a short time of each other
    # parallel_start_diff = abs(exp_args_list[1].start_time - exp_args_list[2].start_time)
    # print(f"parallel_start_diff: {parallel_start_diff}")
    # assert parallel_start_diff < 1.5  # Allow for a small delay

    # Ensure that the entire task graph took the expected amount of time
    total_time = exp_args_list[-1].end_time - exp_args_list[0].start_time
    assert (
        total_time >= TASK_TIME * 3
    )  # Since the critical path involves at least 1.5 seconds of work


if __name__ == "__main__":
    test_execute_task_graph()
    # test_add_dependencies()
