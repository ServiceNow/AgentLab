import copy
import os
from pathlib import Path
from browsergym.experiments.loop import _move_old_exp, yield_all_exp_results
from tqdm import tqdm
import logging


# TODO move this to a more appropriate place
RESULTS_DIR = os.environ.get("UI_COPILOT_RESULTS_DIR", None)
if RESULTS_DIR is None:
    logging.info("$UI_COPILOT_RESULTS_DIR is not defined, Using $HOME/agentlab_results.")
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


def get_ckpt_list(
    chat_model_args,
    add_base_model=True,
    keep_every=None,
    to_keep=None,
):
    """
    Returns a list of arguments for each checkpoint iteration directory in the given model path.

    Args:
        chat_model_args (object): The arguments for the chat model.
        add_base_model (bool, optional): Whether to add the base model directory to the list. Defaults to True.
        keep_every (int, optional): If provided, only checkpoints with iteration numbers divisible by this value will be included. Defaults to None.
        to_keep (list, optional): If provided, only checkpoints with iteration numbers in this list will be included. Defaults to None.

    Returns:
        list: A list of arguments for each checkpoint iteration directory.
    """

    args_list = []
    ckpt_dir = chat_model_args.model_path
    ckpt_itr_dirs = [d for d in os.listdir(ckpt_dir) if d.startswith("ckpt_itr_")]
    for ckpt_itr_dir in ckpt_itr_dirs:
        if keep_every:
            if int(ckpt_itr_dir.split("_")[-1]) % keep_every != 0:
                continue
        elif to_keep:
            if int(ckpt_itr_dir.split("_")[-1]) not in to_keep:
                continue
        args = copy.deepcopy(chat_model_args)
        args.model_path = os.path.join(ckpt_dir, ckpt_itr_dir)
        args_list.append(args)

    if add_base_model:
        base_model_dir = str(Path(chat_model_args.model_path).parent.parent)
        args = copy.deepcopy(chat_model_args)
        args.model_path = base_model_dir
        args_list.append(args)

    return args_list


def overwrite_chat_model_arg(chat_model_args, overwrite_dict):

    for chat_model_arg in chat_model_args:
        for key, value in overwrite_dict.items():
            setattr(chat_model_arg, key, value)

    return chat_model_args
