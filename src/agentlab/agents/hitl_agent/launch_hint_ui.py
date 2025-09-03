"""
Console launcher for the Human-in-the-Loop Generic Agent UI.

Usage (installed entry point):
    agentlab-mentor --benchmark miniwob --task-name miniwob.book-flight --seed 123 --no-headless

This will run a Study with the MultipleProposalGenericAgent and the selected task.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import bgym

from agentlab.agents.hitl_agent.generic_human_guided_agent import get_base_agent
from agentlab.experiments.exp_utils import RESULTS_DIR
from agentlab.experiments.study import Study


def build_benchmark(benchmark_name: str, task_name: str, seed: int, headless: bool):
    # Instantiate benchmark by name using BrowserGym registry
    try:
        benchmark = bgym.DEFAULT_BENCHMARKS[benchmark_name.lower()]()
    except KeyError as e:
        choices = ", ".join(sorted(bgym.DEFAULT_BENCHMARKS.keys()))
        raise SystemExit(f"Unknown benchmark '{benchmark_name}'. Choose one of: {choices}") from e

    filtered_env_args = [
        env_args for env_args in benchmark.env_args_list if env_args.task_name == task_name
    ]
    if not filtered_env_args:
        raise SystemExit(f'No tasks found matching "{task_name}"')
    filtered_env_args = filtered_env_args[:1]  # take the first one
    benchmark.env_args_list = filtered_env_args

    # Reasonable defaults for interactive UI
    for env_args in benchmark.env_args_list:
        env_args.task_seed = seed
        env_args.max_steps = env_args.max_steps or 200
        env_args.headless = headless

    return benchmark


def extract_hints_from_experiment_trace(exp_dir):
    """Extracts hints from every step of each episode in a exp_dir and returns a df with each row containing a hint.

    Args:
        exp_dir: Path-like to a study/experiment directory whose results should be scanned.

    Returns:
        pandas.DataFrame: One row per hint with metadata columns.
    """
    import pandas as pd

    from agentlab.analyze import inspect_results
    from agentlab.experiments.exp_utils import RESULTS_DIR
    from agentlab.experiments.loop import ExpResult

    output = []
    # Use provided exp_dir if set; otherwise default to <$AGENTLAB_EXP_ROOT>/agentlab_mentor
    result_df = inspect_results.load_result_df(exp_dir or (RESULTS_DIR / "agentlab_mentor"))
    if result_df is None:
        # No results to parse; return empty dataframe with expected columns
        return pd.DataFrame(
            columns=[
                "exp_id",
                "agent_name",
                "benchmark",
                "task_name",
                "episode_reward",
                "hint",
            ]
        )
    result_df = result_df.reset_index()
    for _, row in result_df.iterrows():
        result = ExpResult(row.exp_dir)
        episode = result.steps_info
        episode_reward = max([step.reward for step in episode])
        for step_info in episode:
            step_hints = step_info.agent_info.get("extra_info", {}).get("step_hints", None)
            if step_hints:
                for hint in step_hints:
                    output.append(
                        {
                            "exp_id": row["exp_id"],
                            "agent_name": row["agent.agent_name"],
                            "benchmark": row["env.task_name"].split(".")[0],
                            "task_name": row["env.task_name"],
                            "episode_reward": episode_reward,
                            "hint": hint,
                        }
                    )
    output = pd.DataFrame(output)
    output = output.dropna()
    return output


def parse_args():
    p = argparse.ArgumentParser(description="Run HITL Generic Agent UI on a benchmark task")
    p.add_argument(
        "--benchmark",
        required=False,
        help="Benchmark name as registered in BrowserGym, e.g., miniwob, workarena_l1, webarena, visualwebarena",
    )
    p.add_argument(
        "--task-name",
        dest="task_name",
        required=False,
        help="Exact task name within the benchmark (e.g., 'miniwob.book-flight')",
    )
    p.add_argument(
        "--seed",
        type=int,
        required=False,
        help="Task seed to use for the selected task.",
    )
    p.add_argument(
        "--llm-config",
        dest="llm_config",
        default="openai/gpt-5-mini-2025-08-07",
        help="LLM configuration to use for the agent (e.g., 'azure/gpt-5-mini-2025-08-07').",
    )
    p.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the browser headless (default: True). Use --no-headless to show the browser.",
    )
    p.add_argument(
        "--download-hints",
        nargs="?",
        const="extracted_hints.csv",
        required=False,
        default=None,
        metavar="[OUTPUT_CSV]",
        help=(
            "Extract hints from the default study directory and save to OUTPUT_CSV. "
            "If OUTPUT_CSV is omitted, saves to 'extracted_hints.csv'. When provided, other args are ignored."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    save_dir = RESULTS_DIR / "agentlab_mentor"
    if args.download_hints:
        df = extract_hints_from_experiment_trace(save_dir)
        out_path = Path(args.download_hints)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(str(out_path))
        return
    # Validate required args only when not downloading hints
    if not args.benchmark or not args.task_name or args.seed is None:
        raise SystemExit(
            "--benchmark, --task-name, and --seed are required unless using --download-hints"
        )
    benchmark = build_benchmark(args.benchmark, args.task_name, args.seed, args.headless)
    agent_configs = [get_base_agent(args.llm_config)]
    # study is needed to run the 'set_benchmark' method which sets appropriate agent parameters.
    study = Study(agent_args=agent_configs, benchmark=benchmark, logging_level=logging.WARNING)
    study.run(
        n_jobs=1,
        parallel_backend="sequential",
        n_relaunch=1,
        exp_root=save_dir,
    )


if __name__ == "__main__":
    main()
