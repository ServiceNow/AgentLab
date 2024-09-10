import logging
from datetime import datetime
from importlib import import_module
from pathlib import Path

from browsergym.experiments.loop import ExpArgs, yield_all_exp_results
from joblib import Parallel, delayed


def import_object(path: str):
    module_name, obj_name = split_path(path)
    try:
        module = import_module(module_name)
        obj = getattr(module, obj_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Error importing {path}: {e}")
    return obj


def run_experiments(n_jobs, exp_args_list: list[ExpArgs], exp_dir):
    logging.info(f"Saving experiments to {exp_dir}")
    for exp_args in exp_args_list:
        exp_args.agent_args.prepare()
        exp_args.prepare(exp_root=exp_dir)

    try:
        prefer = "processes"
        Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(exp_args.run)() for exp_args in exp_args_list
        )
    finally:
        # will close servers even if there is an exception or ctrl+c
        # servers won't be closed if the script is killed with kill -9 or segfaults.
        logging.info("All jobs are finished. Calling agent_args.close() on all agents...")
        for exp_args in exp_args_list:
            exp_args.agent_args.close()
        logging.info("Experiment finished.")


def make_study_dir(exp_root, study_name, add_date=True):
    if add_date:
        study_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{study_name}"
    return Path(exp_root) / study_name


# def study_agent_on_benchmark(exp_root, study_func, agent, benchmark, extra_kwargs={}):
#     exp_args_list = study_func(agent, benchmark, **extra_kwargs)
#     study_name = f"{study_func.__name__}_{agent.__class__.__name__}_on_{benchmark}"
#     return exp_args_list, make_study_dir(exp_root, study_name)


# def make_study(exp_root, study_func, extra_kwargs={}):
#     exp_args_list = study_func(**extra_kwargs)
#     return exp_args_list, make_study_dir(exp_root, f"{study_func.__name__}")


def relaunch_study(study_dir: str | Path, relaunch_mode="incomplete_only"):
    """Return exp_args_list and study_dir

    Args:
        study_dir: Path
            The directory where the experiments are saved.
        relaunch_mode: str
            Find all incomplete experiments and relaunch them.
            - "incomplete_only": relaunch only the incomplete experiments.
            - "incomplete_or_error": relaunch incomplete or errors.
    """
    study_dir = Path(study_dir)

    if not study_dir.exists():
        raise ValueError(
            f"You asked to relaunch an existing experiment but {study_dir} does not exist."
        )
    exp_args_list = list(_yield_incomplete_experiments(study_dir, relaunch_mode=relaunch_mode))

    if len(exp_args_list) == 0:
        logging.info(f"No incomplete experiments found in {study_dir}.")
        return

    message = f"Make sure the processes that were running are all stopped. Otherwise, "
    f"there will be concurrent writing in the same directories.\n"

    logging.info(message)

    return exp_args_list, Path(study_dir)


def _yield_incomplete_experiments(exp_root, relaunch_mode="incomplete_only"):
    """Find all incomplete experiments and relaunch them."""
    # TODO(make relanch_mode a callable, for flexibility)
    for exp_result in yield_all_exp_results(exp_root, progress_fn=None):  # type: ExpArgs
        try:
            # TODO  implement has_finished instead of dealing with FileNotFoundError
            summary_info = exp_result.summary_info

        except FileNotFoundError:
            yield exp_result.exp_args
            continue

        if relaunch_mode == "incomplete_only":
            continue

        err_msg = summary_info.get("err_msg", None)

        if err_msg is not None:
            if relaunch_mode == "incomplete_or_error":
                yield exp_result.exp_args
            else:
                raise ValueError(f"Unknown relaunch_mode: {relaunch_mode}")


# def str2dict(arg):
#     try:
#         return json.loads(arg)
#     except json.JSONDecodeError as e:
#         raise argparse.ArgumentTypeError(f"Invalid dictionary format: {e}")


def split_path(path: str):
    """Split a path into a module name and an object name."""
    if "/" in path:
        path = path.replace("/", ".")
    module_name, obj_name = path.rsplit(".", 1)
    return module_name, obj_name


# def main():
#     from agentlab.experiments.exp_utils import RESULTS_DIR

#     logging.getLogger().setLevel(logging.INFO)

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--exp_root",
#         default=RESULTS_DIR,
#         help="folder where experiments will be saved",
#     )
#     parser.add_argument(
#         "--n_jobs",
#         default=1,
#         type=int,
#         help="number of parallel jobs",
#     )
#     parser.add_argument(
#         "--exp_config",
#         type=str,
#         default="final_run",
#         help="Python path to the experiment function to launch",
#     )
#     parser.add_argument(
#         "--benchmark",
#         type=str,
#         default="miniwob",
#         choices=["miniwob", "workarena.l1", "workarena.l2", "workarena.l3"],
#         help="Benchmark to launch",
#     )
#     parser.add_argument(
#         "--agent_config",
#         type=str,
#         default=None,
#         help="Python path to the agent config",
#     )
#     parser.add_argument(
#         "--relaunch_mode",
#         default=None,
#         type=str,
#         choices=[None, "incomplete_only", "all_errors", "server_errors"],
#         help="Find all incomplete experiments and relaunch them.",
#     )
#     parser.add_argument(
#         "--extra_kwargs",
#         default="{}",
#         type=str2dict,
#         help="Extra arguments to pass to the experiment group.",
#     )

#     parser.add_argument(
#         "-y", "--auto_accept", action="store_true", help="Skip the prompt to accept the experiment"
#     )

#     parser.add_argument("--shuffle_jobs", action="store_true", help="Shuffle the jobs")

#     args, unknown = parser.parse_known_args()

#     # if relaunch_mode is not None, we will relaunch the experiments
#     if args.relaunch_mode is not None:
#         assert args.exp_root is not None, "You must specify an exp_root to relaunch experiments."
#         exp_args_list, exp_dir = relaunch_study(args.exp_config, args.relaunch_mode)
#     else:
#         # we launch an experiment using the exp_config
#         assert args.exp_config is not None, "You must specify an exp_config."
#         study_func = import_object(args.exp_config)
#         if args.agent_config is not None:
#             agent = import_object(args.agent_config)
#             exp_args_list, exp_dir = study_agent_on_benchmark(
#                 args.exp_root, study_func, agent, args.benchmark, args.extra_kwargs
#             )
#         else:
#             exp_args_list, exp_dir = make_study(args.exp_root, study_func, args.extra_kwargs)

#     message = f"\nYou are about to launch {len(exp_args_list)} experiments in {exp_dir}.\nPress Y to continue.\n"

#     if args.shuffle_jobs:
#         logging.info("Shuffling jobs")
#         random.shuffle(exp_args_list)

#     if args.auto_accept:
#         logging.info(message)
#         answer = "y"
#     else:
#         answer = input(message)

#     if answer.lower() != "y":
#         logging.info("Aborting.")
#     else:
#         run_experiments(args.n_jobs, exp_args_list, exp_dir)


# if __name__ == "__main__":
#     main()
