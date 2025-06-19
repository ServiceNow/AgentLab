from agentlab.agents.vl_agent.config import VL_AGENT_ARGS_DICT
from agentlab.experiments.study import Study
import logging
import os

logging.getLogger().setLevel(logging.INFO)
os.environ["MINIWOB_URL"] = "file:///mnt/home/miniwob-plusplus/miniwob/html/miniwob/"

reproducibility_mode = False
relaunch = False
vl_agent_args_list = [VL_AGENT_ARGS_DICT["ui_agent"]]
benchmark = "miniwob"
parallel_backend = "sequential"
n_jobs = 1
n_relaunch = 1


if __name__ == "__main__":
    if reproducibility_mode:
        for vl_agent_args in vl_agent_args_list:
            vl_agent_args.set_reproducibility_mode()
    if relaunch:
        study = Study.load_most_recent(contains=None)
        study.find_incomplete(include_errors=True)
    else:
        study = Study(vl_agent_args_list, benchmark=benchmark, logging_level_stdout=logging.WARNING)
    study.run(
        parallel_backend=parallel_backend,
        strict_reproducibility=reproducibility_mode,
        n_jobs=n_jobs,
        n_relaunch=n_relaunch,
    )
    if reproducibility_mode:
        study.append_to_journal(strict_reproducibility=True)
