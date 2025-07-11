import json
import logging
import os

from agentlab.agents.tool_use_agent.tool_use_agent import OSWORLD_CLAUDE
from agentlab.benchmarks.osworld import OsworldBenchmark
from agentlab.experiments.study import Study, make_study

fmt = "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(funcName)s() - %(message)s"
logging.basicConfig(level=logging.INFO, force=True, format=fmt, handlers=[logging.StreamHandler()])


def get_most_recent_incomplete_study() -> Study:
    """
    Relaunch an existing study, this will continue incomplete experiments and relaunch errored experiments.
    """
    study = Study.load_most_recent()
    study.find_incomplete(include_errors=True)
    return study


def get_task_ids() -> set[str]:
    with open("experiments/osworld_debug_task_ids.json", "r") as f:
        task_ids = json.load(f)
    return set([task["id"] for task in task_ids])


def main():
    n_jobs = 4
    use_vmware = True
    relaunch = True
    agent_args = [
        OSWORLD_CLAUDE,
        #    OSWORLD_OAI # performs poorly.
    ]  # type: ignore
    parallel_backend = "ray"
    os.environ["AGENTLAB_DEBUG"] = os.environ.get("AGENTLAB_DEBUG", "1")

    study = make_study(
        benchmark=OsworldBenchmark(
            test_set_name="test_small.json"
        ),  # or test_all.json (Exper)  # type: ignore
        agent_args=agent_args,  # type: ignore
        comment="osworld debug 2",
        logging_level=logging.INFO,
        logging_level_stdout=logging.INFO,
    )

    if use_vmware:
        for exp_args in study.exp_args_list:
            exp_args.env_args.provider_name = "vmware"  # type: ignore
            exp_args.env_args.path_to_vm = "OSWorld/vmware_vm_data/Ubuntu0/Ubuntu0.vmx"  # type: ignore
        parallel_backend = "sequential"

    if os.environ.get("AGENTLAB_DEBUG"):
        task_ids = get_task_ids()
        study.exp_args_list = [exp_args for exp_args in study.exp_args_list if exp_args.env_args.task["id"] in task_ids]  # type: ignore
        print(f"Debug on {len(study.exp_args_list)} experiments")
        n_jobs = 1  # Make sure to use 1 job when debugging in VS

    study = get_most_recent_incomplete_study() if relaunch else study
    study.run(n_jobs=n_jobs, n_relaunch=1, parallel_backend=parallel_backend)


if __name__ == "__main__":
    main()
