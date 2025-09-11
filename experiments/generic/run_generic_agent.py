import argparse

from dotenv import load_dotenv

load_dotenv()

import argparse
import logging

from agentlab.agents.generic_agent.tmlr_config import get_base_agent
from agentlab.experiments.study import Study
from bgym import DEFAULT_BENCHMARKS

logging.getLogger().setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--llm-config", required=True)
    parser.add_argument("--relaunch", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=5)
    parser.add_argument("--n-relaunch", type=int, default=3)
    parser.add_argument("--parallel-backend", type=str, default="ray")
    parser.add_argument("--reproducibility-mode", action="store_true")

    args = parser.parse_args()

    # instantiate agent
    agent_args = [get_base_agent(args.llm_config)]
    benchmark = DEFAULT_BENCHMARKS[args.benchmark]()

    ##################### Shuffle env args list, pick subset
    import numpy as np
    rng = np.random.default_rng(42)
    rng.shuffle(benchmark.env_args_list)
    benchmark.env_args_list = benchmark.env_args_list[:33]
    #####################

    # for env_args in benchmark.env_args_list:
        # env_args.max_steps = 100

    if args.relaunch:
        #  relaunch an existing study
        study = Study.load_most_recent(contains=None)
        study.find_incomplete(include_errors=True)

    else:
        study = Study(
            agent_args,
            benchmark,
            logging_level=logging.WARNING,
            logging_level_stdout=logging.WARNING,
        )

    study.run(
        n_jobs=args.n_jobs,
        parallel_backend="ray",
        strict_reproducibility=args.reproducibility_mode,
        n_relaunch=args.n_relaunch,
    )


if __name__ == "__main__":
    main()
