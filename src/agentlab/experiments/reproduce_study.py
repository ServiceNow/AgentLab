import logging

from agentlab.agents.generic_agent.reproducibility_agent import reproduce_study
from agentlab.experiments.exp_utils import RESULTS_DIR
from agentlab.experiments.launch_exp import make_study_dir, run_experiments
from agentlab.experiments.reproducibility_util import (
    write_reproducibility_info,
    add_experiment_to_journal,
    infer_agent,
    infer_benchmark,
)


logging.getLogger().setLevel(logging.INFO)


if __name__ == "__main__":

    # study_dir = RESULTS_DIR / "2024-06-02_18-16-17_final_run"
    old_study_dir = (
        RESULTS_DIR / "2024-09-12_08-39-16_GenericAgent-gpt-4o-mini_on_miniwob_tiny_test"
    )
    study_name, exp_args_list = reproduce_study(old_study_dir)
    study_dir = make_study_dir(RESULTS_DIR, study_name)
    n_jobs = 1

    write_reproducibility_info(
        study_dir=study_dir,
        agent_name=infer_agent(exp_args_list),
        benchmark_name=infer_benchmark(exp_args_list),
        ignore_changes=True,
    )

    # run the experiments

    run_experiments(n_jobs, exp_args_list, study_dir, parallel_backend="joblib")
    # finally:
    #     # will try to gather info at the end even if run_experiments failed
    #     add_experiment_to_journal(study_dir)
