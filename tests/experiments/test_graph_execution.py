import pytest
from agentlab.experiments.graph_execution import (
    execute_task_graph,
    add_dependencies,
    make_dask_client,
)
from time import time, sleep
from browsergym.experiments.loop import ExpArgs, EnvArgs

TASK_TIME = 3


# Mock implementation of the ExpArgs class with timestamp checks
class MockedExpArgs:
    def __init__(self, exp_id, depends_on=None):
        self.exp_id = exp_id
        self.depends_on = depends_on if depends_on else []
        self.start_time = None
        self.end_time = None

    def run(self):
        self.start_time = time()

        # # simulate playright code, (this was causing issues due to python async loop)
        # import playwright.sync_api

        # pw = playwright.sync_api.sync_playwright().start()
        # pw.selectors.set_test_id_attribute("mytestid")
        sleep(TASK_TIME)  # Simulate task execution time
        self.end_time = time()
        return self


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

    # Verify that parallel tasks (task2 and task3) started within a short time of each other
    # parallel_start_diff = abs(exp_args_list[1].start_time - exp_args_list[2].start_time)
    # assert parallel_start_diff < 1.5  # Allow for a small delay

    # Ensure that the entire task graph took the expected amount of time
    total_time = exp_args_list[-1].end_time - exp_args_list[0].start_time
    assert (
        total_time >= TASK_TIME * 3
    )  # Since the critical path involves at least 1.5 seconds of work


def test_add_dependencies():
    # Prepare a simple list of ExpArgs

    def make_exp_args(task_name, exp_id):
        return ExpArgs(agent_args=None, env_args=EnvArgs(task_name=task_name), exp_id=exp_id)

    exp_args_list = [
        make_exp_args("task1", "1"),
        make_exp_args("task2", "2"),
        make_exp_args("task3", "3"),
    ]

    # Define simple task_dependencies
    task_dependencies = {"task1": ["task2"], "task2": [], "task3": ["task1"]}

    # Call the function
    modified_list = add_dependencies(exp_args_list, task_dependencies)

    # Verify dependencies
    assert modified_list[0].depends_on == ("2",)  # task1 depends on task2
    assert modified_list[1].depends_on == ()  # task2 has no dependencies
    assert modified_list[2].depends_on == ("1",)  # task3 depends on task1

    # assert raise if task_dependencies is wrong
    task_dependencies = {"task1": ["task2"], "task2": [], "task4": ["task3"]}
    with pytest.raises(ValueError):
        add_dependencies(exp_args_list, task_dependencies)


if __name__ == "__main__":
    test_execute_task_graph()
    # test_add_dependencies()
