from dataclasses import dataclass
from agentlab.benchmarks.abstract_env import AbstractEnv, AbstractEnvArgs
import bgym


@dataclass
class TauBenchEnvArgs(AbstractEnvArgs):
    """All arguments parameterizing a task in tau-bench"""

    task_name: str
    task_seed: int  # is there any seeds or tasks are deterministic?

    def __init__(self):
        super().__init__()

    def make_env(self, action_mapping, exp_dir, exp_task_kwargs) -> "AbstractEnv":
        # TODO look at how bgym does it. You need to register tasks and do gym.make(task_name)
        pass


class TauBenchEnv(AbstractEnv):
    def __init__(self):
        super().__init__()

    def reset(self, seed=None):
        pass

    def step(self, action: str):
        pass

    def close(self):
        pass


@dataclass
class TauBenchActionSetArgs:
    """Holds hyperparameters for the TauBenchActionSet"""

    def make_action_set(self):
        return TauBenchActionSet()


class TauBenchActionSet(bgym.AbstractActionSet):
    # TODO: Get inspiration from bgym's HighLevelActionSet, perhaps reusing code there, TBD

    def describe(self, with_long_description: bool = True, with_examples: bool = True) -> str:
        # TODO: Implement this method
        pass

    def example_action(self, abstract: bool) -> str:
        # TODO: Implement this method

        pass

    def to_python_code(self, action) -> str:
        # TODO: Implement this method

        pass


def _make_env_args_list():
    # TODO generate all evn_args for the benchmark, get inspiration from bgym's task_list_from_metadata and make_env_args_list_from_repeat_tasks
    return [TauBenchEnvArgs()]


def _task_metadata():
    # load a dataframe containing configuration for all tasks
    pass


def make_tau_benchmark():
    return bgym.Benchmark(
        name="tau-bench",
        high_level_action_set_args=TauBenchActionSet(),
        is_multi_tab=False,
        supports_parallel_seeds=True,
        backends=[
            "taubench"
        ],  # TODO this is not an implemented backend yet and bgym's make_backed implementation with match case needs to be revised
        env_args_list=_make_env_args_list(),  # TODO adapt
        task_metadata=_task_metadata(),  # TODO adapt
    )
