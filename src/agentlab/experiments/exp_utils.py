import os
from pathlib import Path
from browsergym.experiments.loop import _move_old_exp, yield_all_exp_results
from tqdm import tqdm
import logging
from browsergym.experiments.loop import ExpArgs


# TODO move this to a more appropriate place
RESULTS_DIR = os.environ.get("AGENTLAB_EXP_ROOT", None)
if RESULTS_DIR is None:
    RESULTS_DIR = os.environ.get("UI_COPILOT_RESULTS_DIR", None)
if RESULTS_DIR is None:
    logging.info("$AGENTLAB_EXP_ROOT is not defined, Using $HOME/agentlab_results.")
    RESULTS_DIR = Path.home() / "agentlab_results"
else:
    RESULTS_DIR = Path(RESULTS_DIR)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def hide_some_exp(base_dir, filter: callable, just_test):
    """Move all experiments that match the filter to a new name."""
    exp_list = list(yield_all_exp_results(base_dir, progress_fn=None))

    msg = f"Searching {len(exp_list)} experiments to move to _* expriments where `filter(exp_args)` is True."
    if just_test:
        msg += f"\nNote: This is a just a test, no experiments will be moved. Set `just_test=False` to move them."

    logging.info(msg)

    exp_list = tqdm(exp_list, desc=f"Filtering experiments.")

    filtered_out = []
    for exp in exp_list:
        if filter(exp):
            if not just_test:
                _move_old_exp(exp.exp_dir)
            filtered_out.append(exp)
    return filtered_out


def make_seeds(n, offset=42):
    return [seed + offset for seed in range(n)]


def order(exp_args_list: list[ExpArgs]):
    """Store the order of the list of experiments to be able to sort them back.

    This is important for progression or ablation studies.
    """
    for i, exp_args in enumerate(exp_args_list):
        exp_args.order = i
    return exp_args_list
