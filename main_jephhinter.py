"""
Note: This script is a convenience script to launch experiments instead of using
the command line.

Copy this script and modify at will, but don't push your changes to the
repository.
"""

import logging
import os
from typing import cast

from agentlab.agents.generic_agent import (
    AGENT_LLAMA3_70B,
    AGENT_LLAMA31_70B,
    RANDOM_SEARCH_AGENT,
    AGENT_4o,
    AGENT_4o_MINI,
    AGENT_o3_MINI,
    AGENT_o1_MINI,
    AGENT_37_SONNET,
    AGENT_CLAUDE_SONNET_35,
)
from browsergym.experiments.benchmark.utils import make_env_args_list_from_repeat_tasks
from browsergym.experiments.benchmark.metadata.utils import task_list_from_metadata, task_metadata
from bgym import DEFAULT_BENCHMARKS
import numpy as np
from agentlab.experiments.study import Study
from agentlab.agents.tool_use_agent.tool_use_agent import AGENT_CONFIG
from agentlab_configs import AgentLabRunConfig

class AgentLabRun:
    def __init__(self, config: AgentLabRunConfig):
        self.config = config
        self.exp_root = config.exp_root
        self.use_task_hint = config.use_task_hint
        self.hint_db_path = config.hint_db_path or f"{self.exp_root}/hint_db_updated.csv"
        logging.getLogger().setLevel(logging.INFO)
        self._setup_hint_database()
        self._configure_agent()
        self._configure_benchmark()
    def _setup_hint_database(self):
        hint_db_dir = os.path.dirname(self.hint_db_path)
        if not os.path.exists(hint_db_dir):
            os.makedirs(hint_db_dir, exist_ok=True)
            print(f"Created hint database directory: {hint_db_dir}")
        if not os.path.exists(self.hint_db_path):
            import pandas as pd
            empty_db = pd.DataFrame({
                "time_stamp": [],
                "task_name": [],
                "task_seed": [],
                "base_llm": [],
                "agent_name": [],
                "domain_name": [],
                "user_name": [],
                "source": [],
                "semantic_keys": [],
                "hint": []
            })
            empty_db.to_csv(self.hint_db_path, index=False)
            print(f"Created empty hint database at: {self.hint_db_path}")
    def _configure_agent(self):
        AGENT_CONFIG.config.task_hint.hint_db_rel_path = self.hint_db_path
        self.agent_args = cast(list, [AGENT_CONFIG])
        self.agent_args[0].config.task_hint.use_task_hint = self.use_task_hint
    def _configure_benchmark(self):
        benchmark_name = "miniwob"
        self.benchmark = DEFAULT_BENCHMARKS[benchmark_name]()
        self.benchmark.env_args_list = make_env_args_list_from_repeat_tasks(
            task_list=task_list_from_metadata(metadata=task_metadata("miniwob")),
            max_steps=10,
            n_repeats=10,
            seeds_rng=np.random.RandomState(42),
        )
        selected_tasks = [
            "miniwob.book-flight",
            "miniwob.count-shape",
            "miniwob.form-sequence-2",
            "miniwob.number-checkboxes",
            "miniwob.search-engine",
            "miniwob.stock-market",
            "miniwob.use-colorwheel-2",
            "miniwob.bisect-angle",
            "miniwob.click-menu",
            "miniwob.click-scroll-list",
            "miniwob.daily-calendar",
            "miniwob.drag-items-grid",
            "miniwob.grid-coordinate",
            "miniwob.hot-cold",
            "miniwob.right-angle",
            "miniwob.social-media-all",
        ]
        filtered_env_args = []
        for env_args in self.benchmark.env_args_list:
            task_name = env_args.task_name
            if task_name in selected_tasks:
                filtered_env_args.append(env_args)
        filtered_env_args.sort(key=lambda x: (x.task_name, x.task_seed))
        final_env_args = []
        current_task = None
        current_task_count = 0
        max_seeds_per_task = 5
        for env_args in filtered_env_args:
            if env_args.task_name != current_task:
                current_task = env_args.task_name
                current_task_count = 0
            if current_task_count < max_seeds_per_task:
                final_env_args.append(env_args)
                current_task_count += 1
        self.benchmark.env_args_list = final_env_args
        print(f"Running {len(final_env_args)} experiments:")
        for env_args in final_env_args:
            print(f"  - {env_args.task_name} (seed {env_args.task_seed})")
        for env_args in self.benchmark.env_args_list:
            env_args.headless = True
    def run(self):
        if self.config.reproducibility_mode:
            [a.set_reproducibility_mode() for a in self.agent_args]
        if self.config.relaunch:
            study = Study.load_most_recent(contains=None)
            study.find_incomplete(include_errors=True)
        else:
            study = Study(self.agent_args, self.benchmark, logging_level_stdout=logging.WARNING)
        study.run(
            n_jobs=self.config.n_jobs,
            parallel_backend="ray",
            strict_reproducibility=self.config.reproducibility_mode,
            n_relaunch=3,
        )
        if self.config.reproducibility_mode:
            study.append_to_journal(strict_reproducibility=True)

# Example usage
if __name__ == "__main__":
    config = AgentLabRunConfig()
    agentlab_run = AgentLabRun(config)
    agentlab_run.run()
