"""
Console launcher for the Human-in-the-Loop Generic Agent UI.

Usage (installed entry point):
    agentlab-mentor --benchmark miniwob --task-name miniwob.book-flight --seed 123 --no-headless

This will run a Study with the MultipleProposalGenericAgent and the selected task.
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

import bgym

from agentlab.agents.hitl_agent.generic_human_guided_agent import (
    HUMAN_GUIDED_GENERIC_AGENT,
)
from agentlab.experiments.study import Study


def build_benchmark(
    benchmark_name: str, task_name: Optional[str], seed: Optional[int], headless: bool
):
    # Instantiate benchmark by name using BrowserGym registry
    try:
        benchmark = bgym.DEFAULT_BENCHMARKS[benchmark_name.lower()]()
    except KeyError as e:
        choices = ", ".join(sorted(bgym.DEFAULT_BENCHMARKS.keys()))
        raise SystemExit(f"Unknown benchmark '{benchmark_name}'. Choose one of: {choices}") from e

    if task_name:
        # If a fully-qualified name is provided, filter by exact match; otherwise, allow glob
        if any(ch in task_name for ch in "*?[]"):
            benchmark = benchmark.subset_from_glob("task_name", task_name)
        else:
            benchmark = benchmark.subset_from_glob("task_name", task_name)

    # If a specific seed is provided, set it on all env args
    if seed is not None:
        for env_args in benchmark.env_args_list:
            env_args.task_seed = seed

    # Reasonable defaults for interactive UI
    for env_args in benchmark.env_args_list:
        env_args.max_steps = env_args.max_steps or 100
        env_args.headless = headless

    return benchmark


def parse_args():
    p = argparse.ArgumentParser(description="Run HITL Generic Agent UI on a benchmark task")
    p.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark name as registered in BrowserGym, e.g., miniwob, workarena_l1, webarena, visualwebarena",
    )
    p.add_argument(
        "--task-name",
        dest="task_name",
        default=None,
        help="Task name or glob to filter tasks within the benchmark (e.g., 'miniwob.*book*')",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Task seed to use for all selected tasks. If omitted, tasks keep their configured/random seed.",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (UI agent typically runs sequentially)",
    )
    p.add_argument(
        "--parallel-backend",
        default="sequential",
        choices=["sequential", "ray", "joblib"],
        help="Parallel backend to use",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Number of relaunch attempts for incomplete experiments",
    )
    p.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    p.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the browser headless (default: True). Use --no-headless to show the browser.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    logging_level = getattr(logging, args.log_level)

    benchmark = build_benchmark(args.benchmark, args.task_name, args.seed, args.headless)
    agent_configs = [HUMAN_GUIDED_GENERIC_AGENT]

    study = Study(
        agent_configs,
        benchmark,
        logging_level=logging_level,
        logging_level_stdout=logging_level,
    )

    study.run(
        n_jobs=args.jobs,
        parallel_backend=args.parallel_backend,
        n_relaunch=args.retries,
    )


if __name__ == "__main__":
    main()
