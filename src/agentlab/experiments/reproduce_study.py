"""
This script will leverage an old study to reproduce it on the same tasks and
same seeds. Instead of calling the LLM it will reuse the responses from the old
llm. Load the study in agent-xray and look at the Agent Info HTML to compare
the diff in HTML format.
"""

import logging

from agentlab.agents.generic_agent.reproducibility_agent import reproduce_study
from agentlab.experiments.exp_utils import RESULTS_DIR

logging.getLogger().setLevel(logging.INFO)


if __name__ == "__main__":

    old_study = "2024-06-02_18-16-17_final_run"
    # old_study = "2024-09-12_08-39-16_GenericAgent-gpt-4o-mini_on_miniwob_tiny_test"

    study = reproduce_study(RESULTS_DIR / old_study)
    n_jobs = 1

    study.run(n_jobs=n_jobs, parallel_backend="joblib", strict_reproducibility=False)
