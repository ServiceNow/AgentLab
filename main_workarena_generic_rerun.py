"""
Run GenericAgent on a user-specified WorkArena task list with a custom model.
"""

import os

# This is very import for runtime. DO NOT remove!
os.environ.pop("SNOW_INSTANCE_PWD", None)
os.environ.pop("SNOW_INSTANCE_URL", None)
os.environ.pop("SNOW_INSTANCE_UNAME", None)
# Do not remove or override: keep experiment outputs local to this repo.
os.environ.setdefault(
    "AGENTLAB_EXP_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "agentlab_results"),
)

import bgym

from copy import deepcopy

from agentlab.agents.generic_agent.agent_configs import AGENT_GPT5_MINI
from agentlab.experiments.study import make_study
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

# Update this list with the exact task base IDs you want to run.
TASK_IDS = [
    # "workarena.servicenow.some-task-l2",
]

# Choose the model to rerun with (must exist in CHAT_MODEL_ARGS_DICT).
MODEL_NAME = "openai/gpt-5-mini-2025-08-07"

# Number of parallel jobs
n_jobs = 50
parallel_backend = "ray"
avg_step_timeout = 1200  # seconds per step used for Ray cancel timeout
max_steps = 50  # override WorkArena default episode length (was 15 in your env)

# Benchmark to run (change as needed)
BENCHMARK_NAME = "workarena_l2_agent_curriculum_eval"

if __name__ == "__main__":
    if MODEL_NAME not in CHAT_MODEL_ARGS_DICT:
        raise ValueError(
            f"MODEL_NAME={MODEL_NAME!r} not found in CHAT_MODEL_ARGS_DICT. "
            "Update MODEL_NAME to a supported key."
        )

    generic_agent = deepcopy(AGENT_GPT5_MINI)
    generic_agent.chat_model_args = CHAT_MODEL_ARGS_DICT[MODEL_NAME]
    generic_agent.agent_name = f"GenericAgent-{generic_agent.chat_model_args.model_name}".replace(
        "/", "_"
    )

    benchmark = bgym.DEFAULT_BENCHMARKS[BENCHMARK_NAME]()

    if TASK_IDS:
        benchmark.env_args_list = [
            env_args
            for env_args in benchmark.env_args_list
            if env_args.task_name in TASK_IDS
        ]

    print(f"Running {len(benchmark.env_args_list)} tasks after filter")
    if not benchmark.env_args_list:
        raise SystemExit("No experiments to run after filtering.")

    for env_args in benchmark.env_args_list:
        env_args.headless = True
        env_args.max_steps = max_steps

    study = make_study(
        benchmark=benchmark,
        agent_args=[generic_agent],
        comment=f"generic rerun ({MODEL_NAME})",
    )
    study.avg_step_timeout = avg_step_timeout
    study.run(
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        strict_reproducibility=False,
        n_relaunch=3,
    )
