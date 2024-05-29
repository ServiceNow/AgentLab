from datetime import datetime
import logging
import multiprocessing
from pathlib import Path
import random
from joblib import Parallel, delayed
from agentlab.analyze import error_categorization
from agentlab.llm.llm_servers import LLMServers
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT
from browsergym.experiments.loop import ExpArgs, yield_all_exp_results
from agentlab.webarena_setup.check_webarena_servers import check_webarena_servers
import agentlab
import argparse

logging.getLogger().setLevel(logging.INFO)


def run_exp(exp_args: ExpArgs, server_error_flag: bool, llm_servers: LLMServers):
    if server_error_flag is not None and server_error_flag.value:
        logging.info("Skipping job because of server error.")
        return
    llm_servers.wait_for_server(exp_args.agent_args.chat_model_args.key())
    exp_args.run()


def main(
    exp_root,
    exp_group_name,
    n_jobs,
    exp_args_list=None,
    shuffle_jobs=False,
    auto_accept=False,
    use_threads_instead_of_processes=False,
    relaunch_mode=None,
    server_error_flag=None,
):
    """Launch a group of experiments.

    Args:
        exp_root: folder where experiments will be saved
        exp_group_name: name of the experiment group to launch as defined in
            exp_configs.EXP_GROUPS
        n_jobs: number of parallel jobs in joblib
        exp_args_list: list of ExpArgs to launch. If None, will use the list
            from exp_configs.EXP_GROUPS[exp_group_name]
        shuffle_jobs: shuffle the order of the experiments
        auto_accept: skip the prompt to accept the experiment
        use_threads_instead_of_processes: prefer threads over processes in
            joblib, useful for debugging.
        relaunch_mode: choice of None, 'incomplete_only', 'all_errors', 'server_error',
    """
    exp_args_list, exp_dir = _validate_launch_mode(
        exp_root, exp_group_name, exp_args_list, relaunch_mode, auto_accept
    )

    if shuffle_jobs:
        random.shuffle(exp_args_list)

    # if webarena, check if the server is running
    if any("webarena" in exp_args.env_args.task_name for exp_args in exp_args_list):
        logging.info("Checking webarena servers...")
        check_webarena_servers()

    # launch servers if needed
    llm_servers = LLMServers(exp_args_list)
    logging.info(f"Saving experiments to {exp_dir}")
    for exp_args in exp_args_list:
        exp_args.prepare(exp_root=exp_dir)

    try:
        prefer = "threads" if use_threads_instead_of_processes else "processes"
        Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(run_exp)(exp_args, server_error_flag, llm_servers) for exp_args in exp_args_list
        )
    finally:
        # will close servers even if there is an exception or ctrl+c
        # servers won't be closed if the script is killed with kill -9 or segfaults.
        # TODO: it would be convinient to have a way to close servers in that case.
        logging.info("Closing all LLM servers...")
        llm_servers.close_all_servers()
        logging.info("LLM servers closed.")

    return exp_group_name


def _validate_launch_mode(
    exp_root, exp_group_name, exp_args_list, relaunch_mode, auto_accept
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

            exp_group_name, exp_args_list = exp_configs.get_exp_args_list(exp_group_name)

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
            reward_info = exp_result.reward_info
        except FileNotFoundError:
            yield exp_result.exp_args
            continue

        if relaunch_mode == "incomplete_only":
            continue

        err_msg = reward_info.get("err_msg", None)
        stack_trace = reward_info.get("stack_trace", None)

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


def meta_main(
    exp_root,
    exp_group_name,
    n_jobs,
    shuffle_jobs=True,
    auto_accept=False,
    use_threads_instead_of_processes=False,
    relaunch_mode=None,
    n_retry=10,
):
    # TODO: deprecated: server_error_flag isn't used anymore

    manager = multiprocessing.Manager()
    server_error_flag = manager.Value("i", 0)

    itr = 0
    while itr < n_retry:
        exp_group_name = main(
            exp_root=exp_root,
            exp_group_name=exp_group_name,
            n_jobs=n_jobs,
            shuffle_jobs=shuffle_jobs,
            auto_accept=auto_accept,
            use_threads_instead_of_processes=use_threads_instead_of_processes,
            relaunch_mode=relaunch_mode,
            server_error_flag=server_error_flag,
        )
        if not server_error_flag.value:
            return

        # values to overwrite for the next iterations
        server_error_flag.value = 0
        relaunch_mode = "server_errors"
        auto_accept = True

        itr += 1
        logging.info("\n-----------------------------------")
        logging.info(f"Server error occurred. Retrying {itr}/{n_retry}...")
        logging.info("-----------------------------------\n")

    logging.info("Server error occurred too many times. Aborting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_group_name",
        default="generic_agent_test",
    )
    parser.add_argument(
        "--exp_root",
        default=Path(agentlab.__file__).parent.parent.parent.parent / "results",
        help="folder where experiments will be saved",
    )
    parser.add_argument(
        "--n_jobs",
        default=-1,
        type=int,
        help="number of parallel jobs",
    )
    parser.add_argument(
        "--relaunch_mode",
        default=None,
        type=str,
        choices=[None, "incomplete_only", "all_errors", "server_errors"],
        help="Find all incomplete experiments and relaunch them.",
    )

    args, unknown = parser.parse_known_args()
    main(args.exp_root, args.exp_group_name, args.n_jobs, relaunch_mode=args.relaunch_mode)
