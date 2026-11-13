from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import benchmarks
import numpy as np
import pandas as pd
from bgym import Benchmark, EnvArgs, HighLevelActionSetArgs
from browsergym.experiments.benchmark.base import BenchmarkBackend
from browsergym.experiments.benchmark.utils import make_env_args_list_from_repeat_tasks
from dataclasses_json import DataClassJsonMixin, config
from torch import threshold

from agentlab.analyze.inspect_results import load_result_df
from agentlab.experiments.study import Study


@dataclass
class ResampleBenchmark(Benchmark):
    exp_dir: Path = None
    name: str = None
    high_level_action_set_args: HighLevelActionSetArgs = None
    is_multi_tab: bool = None
    supports_parallel_seeds: bool = None
    env_args_list: list[EnvArgs] = None
    backends: list[BenchmarkBackend] = None
    task_metadata: Optional[pd.DataFrame] = field(
        default_factory=lambda: None,
        metadata=config(
            encoder=lambda df: df.to_dict(orient="records") if df is not None else None,
            decoder=lambda items: pd.DataFrame(items) if items is not None else None,
        ),
    )

    def __post_init__(self):
        assert self.exp_dir is not None
        study = Study.load(self.exp_dir)
        benchmark = study.benchmark

        self.name = f"resample-{benchmark.name}"
        self.high_level_action_set_args = benchmark.high_level_action_set_args
        self.is_multi_tab = benchmark.is_multi_tab
        self.supports_parallel_seeds = benchmark.supports_parallel_seeds
        self.backends = benchmark.backends
        # we discard the task_metadata to create new ones in post_init

        values = self.evaluate(study, benchmark.env_args_list)
        selected_env_args = self.select(values, benchmark.env_args_list)

        if len(selected_env_args) == 0:
            raise ValueError("No env_args selected, lower restrictions")

        self.env_args_list = selected_env_args

        super().__post_init__()

    @abstractmethod
    def evaluate(self, study, env_args_list):
        pass

    @abstractmethod
    def select(self, values, env_args_list):
        pass


@dataclass
class AllTasksBenchmark(ResampleBenchmark):
    def evaluate(self, study, env_args_list):
        return [0] * len(env_args_list)

    def select(self, values, env_args_list):
        return env_args_list


@dataclass
class HighVarianceBenchmark(ResampleBenchmark):
    threshold: float = 0.2

    def evaluate(self, study: Study, env_args_list):
        result_df = load_result_df(study.dir)
        return dict(result_df.groupby("env.task_name")["cum_reward"].std())

    def select(self, values, env_args_list):
        selected_env_args = []
        for env_args in env_args_list:
            if values[env_args.task_name] > self.threshold:
                selected_env_args.append(env_args)
        return selected_env_args


@dataclass
class StochasticHighVarianceBenchmark(ResampleBenchmark):
    regulation_threshold: float = 0.1
    total_seeds = 600
    min_seeds = 2
    random_seed = 42

    def evaluate(self, study: Study, env_args_list):
        result_df = load_result_df(study.dir)
        var = result_df.groupby("env.task_name")["cum_reward"].var()
        probs = dict((var + self.regulation_threshold) / (var + self.regulation_threshold).sum())
        return probs

    def select(self, values, env_args_list: list[EnvArgs]):
        selected_env_args = []
        max_steps = env_args_list[0].max_steps
        for task_name, p in values.items():
            # ceil to avoid missing seeds
            n_seeds = np.random.RandomState(self.random_seed).poisson(p * self.total_seeds)
            n_seeds = max(n_seeds, self.min_seeds)
            for seed in np.random.RandomState(self.random_seed).randint(0, 2**32, n_seeds):
                selected_env_args.append(
                    EnvArgs(
                        task_name=task_name,
                        task_seed=int(seed),
                        max_steps=max_steps,
                        headless=True,
                        record_video=False,
                        wait_for_user_message=False,
                        viewport=None,
                        slow_mo=None,
                        storage_state=None,
                        task_kwargs=None,
                    )
                )
        return selected_env_args


if __name__ == "__main__":
    exp_dir = Path(
        "/Users/t.lesellierdechezell/agentlab_results/2025-03-04_14-43-48_genericagent-gpt-4o-mini-2024-07-18-on-miniwob"
    )
    benchmark = StochasticHighVarianceBenchmark(exp_dir=exp_dir)
    print(benchmark.env_args_list)
