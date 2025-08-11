from bgym import DEFAULT_BENCHMARKS

from agentlab.agents.generic_agent.tmlr_config import (
    BASE_FLAGS,
    CHAT_MODEL_ARGS_DICT,
    GenericAgentArgs,
)
from agentlab.experiments.study import Study

agent_config = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-5-nano-2025-08-07"],
    flags=BASE_FLAGS,
)

# choose a list of agents to evaluate
agent_configs = [agent_config]

# chose your benchmark
benchmark = DEFAULT_BENCHMARKS["miniwob_tiny_test"]()

# benchmark = DEFAULT_BENCHMARKS["miniwob"]()  # 125 tasks
# benchmark = benchmark.subset_from_glob(column="task_name", glob="*enter*")

## Number of parallel jobs
n_jobs = 4  # Make sure to use 1 job when debugging in VSCode

if __name__ == "__main__":  # necessary for dask backend

    # A study evaluates multiple agents on a benchmark
    study = Study(agent_configs, benchmark)

    study.run(
        n_jobs=n_jobs,
        parallel_backend="ray",  # "ray", "joblib" or "sequential"
        n_relaunch=3,  # will automatically relaunch tasks with system error or incomplete tasks.
    )
