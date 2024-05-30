from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from browsergym.experiments.loop import ExpArgs
from agentlab.experiments import args
from agentlab.experiments import task_collections as tasks
from agentlab.agents.dynamic_prompting import Flags, BASIC_FLAGS
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

from browsergym.experiments.loop import EnvArgs
from agentlab.experiments.exp_configs import make_seeds
from agentlab.experiments.exp_utils import get_ckpt_list, overwrite_chat_model_arg


def get_exp_args_list(func_name: str):
    """Run func_name and return exp_arg_list"""
    func = globals()[func_name]
    return func_name, func()


model_name_list = [
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    # "microsoft/Phi-3-mini-128k-instruct",
    # "codellama/CodeLlama-34b-Instruct-hf",
    # "Salesforce/xLAM-v0.1-r",
    # "deepseek-ai/deepseek-coder-6.7b-instruct",
    # "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # "microsoft/WizardLM-2-8x22B"
    # "finetuning/Meta-Llama-3-8B-Instruct",
    "finetuning/debug",
]

# set to None or empty dict to keep the default values
overwrite_chat_model_args_dict = {
    # "model_url": "https://abceab17-35da-41a6-90cc-d223145a18d2.job.console.elementai.com",
    # "max_total_tokens": 16_384,
}

MINIWOB_RS_OSS_FLAGS = Flags(
    use_html=args.Choice([True, False], p=[0.75, 0.25]),
    use_ax_tree=args.Choice([True, False], p=[0.75, 0.25]),
    use_plan=args.Choice([True, False], p=[0.25, 0.75]),
    use_criticise=args.Choice([True, False], p=[0.25, 0.75]),
    use_thinking=args.Choice([True, False], p=[0.5, 0.5]),
    use_think_history=args.Choice([True, False], p=[0.5, 0.5]),
    use_error_logs=args.Choice([True, False], p=[0.8, 0.2]),
    use_past_error_logs=args.Choice([True, False], p=[0.5, 0.5]),
    use_history=args.Choice([True, False], p=[0.8, 0.2]),
    use_action_history=args.Choice([True, False], p=[0.8, 0.2]),
    use_memory=args.Choice([True, False], p=[0.25, 0.75]),
    use_diff=args.Choice([True, False], p=[0.5, 0.5]),
    use_concrete_example=True,
    use_abstract_example=True,
    multi_actions=args.Choice([True, False], p=[0.5, 0.5]),
    action_space=args.Choice(["bid", "python"]),
    use_hints=args.Choice([True, False], p=[0.25, 0.75]),
    use_screenshot=False,
    max_prompt_tokens=args.Choice([16_384, 8_192], p=[0.5, 0.5]),
)

RS_OSS_FLAGS = Flags(
    # use_html=args.Choice([True, False], p=[0.75, 0.25]),
    use_html=False,
    # use_ax_tree=args.Choice([True, False], p=[0.75, 0.25]),
    use_ax_tree=False,
    use_plan=args.Choice([True, False], p=[0.25, 0.75]),
    use_criticise=args.Choice([True, False], p=[0.25, 0.75]),
    use_thinking=args.Choice([True, False], p=[0.5, 0.5]),
    use_think_history=args.Choice([True, False], p=[0.5, 0.5]),
    use_error_logs=args.Choice([True, False], p=[0.5, 0.5]),
    use_past_error_logs=args.Choice([True, False], p=[0.5, 0.5]),
    use_history=args.Choice([True, False], p=[0.8, 0.2]),
    use_action_history=args.Choice([True, False], p=[0.8, 0.2]),
    use_memory=args.Choice([True, False], p=[0.25, 0.75]),
    use_diff=args.Choice([True, False], p=[0.5, 0.5]),
    use_concrete_example=True,
    use_abstract_example=True,
    multi_actions=args.Choice([True, False], p=[0.5, 0.5]),
    action_space=args.Choice(["bid", "python"]),
    use_hints=args.Choice([True, False], p=[0.25, 0.75]),
    use_screenshot=False,
    max_prompt_tokens=args.Choice([16_384, 8_192], p=[0.5, 0.5]),
)

FINETUNING_FLAGS = Flags(
    use_html=False,
    use_ax_tree=True,
    use_plan=False,
    use_criticise=False,
    use_thinking=False,
    use_think_history=False,
    use_error_logs=False,
    use_past_error_logs=False,
    use_history=True,
    use_action_history=True,
    use_memory=False,
    use_diff=True,
    use_concrete_example=True,
    use_abstract_example=True,
    multi_actions=False,
    action_space="bid",
    use_hints=False,
    use_screenshot=False,
)


def OSS_eval(benchmark: str = "workarena", task_name: str = "AllMenuTask"):

    flags = FINETUNING_FLAGS

    if task_name:
        # TODO: automate this
        if task_name == "AllMenuTask":
            task_list = ["workarena.servicenow.all-menu"]

        n_seeds = 10
    else:
        if benchmark == "miniwob":
            task_list = tasks.miniwob_all
        elif benchmark == "workarena":
            task_list = tasks.workarena_tasks
        elif benchmark == "webarena":
            task_list = tasks.webarena_tasks
            n_seeds = 1  # webarena doesn't have any randomness for a given task
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")

    chat_model_args_list = [CHAT_MODEL_ARGS_DICT[k] for k in model_name_list]
    if overwrite_chat_model_args_dict:
        chat_model_args_list = overwrite_chat_model_arg(
            chat_model_args_list, overwrite_chat_model_args_dict
        )

    return args.expand_cross_product(
        ExpArgs(
            agent_args=GenericAgentArgs(
                chat_model_args=args.CrossProd(chat_model_args_list),
                flags=flags,
            ),
            env_args=EnvArgs(
                max_steps=10,
                task_seed=args.CrossProd(make_seeds(n_seeds)),
                task_name=args.CrossProd(task_list),
            ),
            enable_debug=False,
        )
    )


def OSS_random_search(benchmark: str = "workarena", task_name: str = None):
    """Example of random search. Modify this at will, but don't push your
    changes.

    The variance will usually be relatively high and the search space is soo big
    that the false discovery rate will be particularly high. Make sure to
    analyze the  results with caution and don't actually draw final conclusions
    from these experiments.

    TODO: eventually merge w/ exp_configs'
    """
    if task_name:
        # TODO: automate this
        if task_name == "AllMenuTask":
            task_list = ["workarena.servicenow.all-menu"]

        n_seeds = 10
        flags = RS_OSS_FLAGS

    else:
        n_seeds = 3
        if benchmark == "miniwob":
            flags = MINIWOB_RS_OSS_FLAGS
            task_list = tasks.miniwob_all
        elif benchmark == "workarena":
            flags = RS_OSS_FLAGS
            task_list = tasks.workarena_tasks[1:]
        elif benchmark == "webarena":
            flags = RS_OSS_FLAGS
            task_list = tasks.webarena_tasks
            n_seeds = 1  # webarena doesn't have any randomness for a given task
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")

    chat_model_args_list = [CHAT_MODEL_ARGS_DICT[k] for k in model_name_list]
    if overwrite_chat_model_args_dict:
        chat_model_args_list = overwrite_chat_model_arg(
            chat_model_args_list, overwrite_chat_model_args_dict
        )

    return args.sample_and_expand_cross_product(
        ExpArgs(
            agent_args=GenericAgentArgs(
                chat_model_args=args.Choice(chat_model_args_list),
                flags=flags,
            ),
            env_args=EnvArgs(
                max_steps=10,
                task_seed=args.CrossProd(make_seeds(n_seeds)),
                task_name=args.CrossProd(task_list),
            ),
            enable_debug=False,
        ),
        n_samples=16,  # number of samples
    )


def OSS_finetuning_eval(benchmark: str = "workarena", task_name: str = "AllMenuTask"):
    """Example of random search. Modify this at will, but don't push your
    changes.

    The variance will usually be relatively high and the search space is soo big
    that the false discovery rate will be particularly high. Make sure to
    analyze the  results with caution and don't actually draw final conclusions
    from these experiments.

    TODO: eventually merge w/ exp_configs'
    """

    assert len(model_name_list) == 1, "Only one model is supported for finetuning eval"
    assert model_name_list[0].startswith(
        "finetuning/"
    ), "Only finetuning models are supported for finetuning eval"

    if task_name:
        # TODO: automate this
        if task_name == "AllMenuTask":
            task_list = ["workarena.servicenow.all-menu"]

        n_seeds = 10

    else:
        if benchmark == "miniwob":
            task_list = tasks.miniwob_all
        elif benchmark == "workarena":
            task_list = tasks.workarena_tasks[1:]
        elif benchmark == "webarena":
            task_list = tasks.webarena_tasks
            n_seeds = 1  # webarena doesn't have any randomness for a given task
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")

    return args.expand_cross_product(
        ExpArgs(
            agent_args=GenericAgentArgs(
                chat_model_args=args.CrossProd(
                    get_ckpt_list(CHAT_MODEL_ARGS_DICT[model_name_list[0]])
                ),
                flags=FINETUNING_FLAGS,
            ),
            env_args=EnvArgs(
                max_steps=10,
                task_seed=args.CrossProd(make_seeds(n_seeds)),
                task_name=args.CrossProd(task_list),
            ),
            enable_debug=False,
        )
    )
