import json
import logging
import os

from agentlab.agents.tool_use_agent.tool_use_agent import OSWORLD_CLAUDE
from agentlab.benchmarks.osworld import OsworldBenchmark
from agentlab.experiments.study import make_study

fmt = "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(funcName)s() - %(message)s"
logging.basicConfig(level=logging.INFO, force=True, format=fmt, handlers=[logging.StreamHandler()])


def get_task_ids() -> set[str]:
    with open("experiments/osworld_debug_task_ids.json", "r") as f:
        task_ids = json.load(f)
    return set([task["id"] for task in task_ids])


def main():
    n_jobs = 1
    os.environ["AGENTLAB_DEBUG"] = "1"
    study = make_study(
        benchmark=OsworldBenchmark(test_set_name="test_small.json"),  # type: ignore
        agent_args=[OSWORLD_CLAUDE],
        comment="osworld debug 2",
        logging_level=logging.INFO,
        logging_level_stdout=logging.INFO,
    )

    if os.environ.get("AGENTLAB_DEBUG"):
        task_ids = get_task_ids()
        study.exp_args_list = [exp_args for exp_args in study.exp_args_list if exp_args.env_args.task["id"] in task_ids]
        print(f"Debug on {len(study.exp_args_list)} experiments")
        study.run(n_jobs=2, n_relaunch=1, parallel_backend="ray")
    else:
        study.run(n_jobs=n_jobs, n_relaunch=1, parallel_backend="ray")


if __name__ == "__main__":
    main()
