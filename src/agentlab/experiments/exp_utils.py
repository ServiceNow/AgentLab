import os
from pathlib import Path
from browsergym.experiments.loop import _move_old_exp, yield_all_exp_results
from tqdm import tqdm
import logging
from browsergym.experiments.loop import ExpArgs
from contextlib import contextmanager
import signal
import sys
from time import time, sleep

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


def run_exp(exp_arg: ExpArgs, *dependencies, avg_step_timeout=30):
    """Run exp_args.run() with a timeout and handle dependencies."""
    episode_timeout = _episode_timeout(exp_arg, avg_step_timeout=avg_step_timeout)
    with timeout_manager(seconds=episode_timeout):
        return exp_arg.run()


def _episode_timeout(exp_arg: ExpArgs, avg_step_timeout=30):
    """Some logic to determine the episode timeout."""
    max_steps = getattr(exp_arg.env_args, "max_steps", None)
    if max_steps is None:
        episode_timeout_global = 10 * 60 * 60  # 10 hours
    else:
        episode_timeout_global = exp_arg.env_args.max_steps * avg_step_timeout

    episode_timeout_exp = getattr(exp_arg, "episode_timeout", episode_timeout_global)

    return min(episode_timeout_global, episode_timeout_exp)


@contextmanager
def timeout_manager(seconds: int = None):
    """Context manager to handle timeouts."""

    # Check if we're on Windows
    if seconds is None or sys.platform == "win32":
        try:
            yield
        finally:
            pass
    else:
        seconds = max(1, int(seconds))

        def handler(signum, frame):
            print("before raising timeout")
            raise TimeoutError(f"Operation timed out after {seconds} seconds")

        print(f"Setting timeout to {seconds} seconds.")
        # Register the signal handler
        previous_handler = signal.signal(signal.SIGALRM, handler)
        # Set the alarm
        signal.alarm(seconds)

        try:
            yield
        finally:
            # Cleanup: cancel alarm and restore previous handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)


# Mock implementation of the ExpArgs class with timestamp checks for unit testing
class MockedExpArgs:
    def __init__(self, exp_id, depends_on=None):
        self.exp_id = exp_id
        self.depends_on = depends_on if depends_on else []
        self.start_time = None
        self.end_time = None
        self.env_args = None

    def run(self):
        self.start_time = time()

        # # simulate playright code, (this was causing issues due to python async loop)
        # import playwright.sync_api

        # pw = playwright.sync_api.sync_playwright().start()
        # pw.selectors.set_test_id_attribute("mytestid")
        sleep(3)  # Simulate task execution time
        self.end_time = time()
        return self


def make_seeds(n, offset=42):
    raise DeprecationWarning("This function will be removed. Comment out this error if needed.")
    return [seed + offset for seed in range(n)]


def order(exp_args_list: list[ExpArgs]):
    raise DeprecationWarning("This function will be removed. Comment out this error if needed.")
    """Store the order of the list of experiments to be able to sort them back.

    This is important for progression or ablation studies.
    """
    for i, exp_args in enumerate(exp_args_list):
        exp_args.order = i
    return exp_args_list


# This was an old function for filtering some issue with the experiments.
def hide_some_exp(base_dir, filter: callable, just_test):
    """Move all experiments that match the filter to a new name."""
    raise DeprecationWarning("This function will be removed. Comment out this error if needed.")
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
