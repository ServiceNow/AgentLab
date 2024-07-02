from datetime import datetime
import logging
from pathlib import Path
import random
from joblib import Parallel, delayed
from agentlab.analyze import error_categorization
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT
from browsergym.experiments.loop import ExpArgs, yield_all_exp_results
from agentlab.webarena_setup.check_webarena_servers import check_webarena_servers
import argparse


def main(
    exp_root,
    n_jobs,
    exp_group_name=None,
    benchmark: str = None,
    model_name: str = None,
    exp_args_list=None,
    shuffle_jobs=False,
    auto_accept=False,
    use_threads_instead_of_processes=False,
    relaunch_mode=None,
):
    """Launch a group of experiments.

    Args:
        exp_group_name: name of the experiment group to launch as defined in
            exp_configs.EXP_GROUPS
        exp_root: folder where experiments will be saved
        n_jobs: number of parallel jobs in joblib
        exp_args_list: list of ExpArgs to launch. If None, will use the list
            from exp_configs.EXP_GROUPS[exp_group_name]
        shuffle_jobs: shuffle the order of the experiments
        auto_accept: skip the prompt to accept the experiment
        use_threads_instead_of_processes: prefer threads over processes in
            joblib, useful for debugging.
        relaunch_mode: choice of None, 'incomplete_only', 'all_errors', 'server_error',
    """
    if exp_group_name:
        logging.info(f"Launching experiment group: {exp_group_name}")
        if benchmark or model_name:
            logging.info(f"Overriding benchmark: {benchmark} and model_name: {model_name}")
        extra_kwargs = {}

    elif benchmark and model_name:
        logging.info(f"Launching benchmark: {benchmark} with model: {model_name}")
        exp_group_name = "final_run"
        extra_kwargs = {"benchmark": benchmark, "model_name": model_name}

    exp_args_list, exp_dir = _validate_launch_mode(
        exp_root, exp_group_name, exp_args_list, relaunch_mode, auto_accept, extra_kwargs
    )

    if shuffle_jobs:
        random.shuffle(exp_args_list)

    # if webarena, check if the server is running
    if any("webarena" in exp_args.env_args.task_name for exp_args in exp_args_list):
        logging.info("Checking webarena servers...")
        check_webarena_servers()

    # launch servers if needed
    registry = {}

    logging.info(f"Saving experiments to {exp_dir}")
    for exp_args in exp_args_list:
        exp_args.agent_args.prepare(registry)
        exp_args.prepare(exp_root=exp_dir)

    try:
        prefer = "threads" if use_threads_instead_of_processes else "processes"
        Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(exp_args.run)() for exp_args in exp_args_list
        )
    finally:
        # will close servers even if there is an exception or ctrl+c
        # servers won't be closed if the script is killed with kill -9 or segfaults.
        # TODO: it would be convinient to have a way to close servers in that case.
        logging.info("Closing all LLM servers...")
        for exp_args in exp_args_list:
            exp_args.agent_args.close(registry)  # TODO: get rid of that
        logging.info("LLM servers closed.")

    return exp_group_name


def _validate_launch_mode(
    exp_root, exp_group_name, exp_args_list, relaunch_mode, auto_accept, extra_kwargs
) -> tuple[list[ExpArgs], Path]:
    if relaunch_mode is not None:
        # dig into an existing experiment group and relaunch all incomplete experiments
        exp_dir = Path(exp_root) / exp_group_name
        if not exp_dir.exists():
            raise ValueError(
                f"You asked to relaunch an existing experiment but {exp_group_name} does not exist."
            )

        exp_args_list = list(_yield_incomplete_experiments(exp_dir, relaunch_mode=relaunch_mode))

        if len(exp_args_list) == 0:
            logging.info(f"No incomplete experiments found in {exp_dir}.")
            return

        message = (
            f"\nHey, You are about to relaunch {len(exp_args_list)} incomplete or errored experiments in {exp_dir}. "
            f"Make sure the processes that were running are all stopped. Otherwise, "
            f"there will be concurrent writing in the same directories.\n"
            f"Press Y to continue.\n"
        )

        # overwrtting the model_url just in case
        for exp_args in exp_args_list:
            exp_args.agent_args.chat_model_args.model_url = CHAT_MODEL_ARGS_DICT[
                exp_args.agent_args.chat_model_args.model_name
            ].model_url

    else:
        if exp_args_list is None:
            from agentlab.experiments import exp_configs

            exp_group_name, exp_args_list = exp_configs.get_exp_args_list(
                exp_group_name, **extra_kwargs
            )

        # overwriting exp_group_name for the recursive call
        exp_group_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{exp_group_name}"
        exp_dir = Path(exp_root) / exp_group_name
        message = (
            f"\nYou are about to launch {len(exp_args_list)} experiments in {exp_dir}.\n"
            f"Press Y to continue.\n"
        )

    if auto_accept:
        logging.info(message)
        answer = "y"
    else:
        answer = input(message)

    if answer.lower() != "y":
        logging.info("Aborting.")
        return

    return exp_args_list, exp_dir


def _yield_incomplete_experiments(exp_root, relaunch_mode="incomplete_only"):
    """Find all incomplete experiments and relaunch them."""
    # TODO(make relanch_mode a callable, for flexibility)
    for exp_result in yield_all_exp_results(exp_root, progress_fn=None):  # type: ExpArgs
        try:
            summary_info = exp_result.summary_info
        except FileNotFoundError:
            yield exp_result.exp_args
            continue

        if relaunch_mode == "incomplete_only":
            continue

        err_msg = summary_info.get("err_msg", None)
        stack_trace = summary_info.get("stack_trace", None)

        if err_msg is not None:
            if relaunch_mode == "all_errors":
                yield exp_result.exp_args
            elif relaunch_mode == "server_errors":
                critical_server_error = error_categorization.is_critical_server_error(
                    err_msg, stack_trace
                )
                minor_server_error = error_categorization.is_minor_server_error(
                    err_msg, stack_trace
                )
                if critical_server_error or minor_server_error:
                    yield exp_result.exp_args
            else:
                raise ValueError(f"Unknown relaunch_mode: {relaunch_mode}")


if __name__ == "__main__":
    from agentlab.experiments.exp_utils import RESULTS_DIR

    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_root",
        default=RESULTS_DIR,
        help="folder where experiments will be saved",
    )
    parser.add_argument(
        "--n_jobs",
        default=1,
        type=int,
        help="number of parallel jobs",
    )
    parser.add_argument(
        "--exp_group_name",
        type=str,
        default=None,
        help="Name of the experiment group to launch as defined in exp_configs.py",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="miniwob",
        choices=["miniwob", "workarena.l1", "workarena.l2", "workarena.l3"],
        help="Benchmark to launch",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama3-8b",
        choices=["gpt-3.5", "llama3-8b", "llama3-70b", "gpt-4o", "gpt-4o-vision"],
        help="Model to launch",
    )
    parser.add_argument(
        "--relaunch_mode",
        default=None,
        type=str,
        choices=[None, "incomplete_only", "all_errors", "server_errors"],
        help="Find all incomplete experiments and relaunch them.",
    )

    args, unknown = parser.parse_known_args()
    main(
        exp_root=args.exp_root,
        n_jobs=args.n_jobs,
        exp_group_name=args.exp_group_name,
        benchmark=args.benchmark,
        model_name=args.model_name,
        relaunch_mode=args.relaunch_mode,
    )
