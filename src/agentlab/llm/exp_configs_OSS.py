from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from browsergym.experiments.loop import ExpArgs, get_ckpt_list
from agentlab.experiments import args
from agentlab.experiments import task_collections as tasks
from agentlab.agents.dynamic_prompting import Flags, BASIC_FLAGS
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

from browsergym.experiments.loop import EnvArgs
from agentlab.experiments.exp_configs import make_seeds


def get_exp_args_list(func_name: str):
    """Run func_name and return exp_arg_list"""
    func = globals()[func_name]
    return func_name, func()


def tgi_toolkit_test():
    """Minimalistic experiment to test the system."""
    basic_flags = BASIC_FLAGS.copy()
    basic_flags.use_html = False
    basic_flags.use_ax_tree = True
    return args.expand_cross_product(
        ExpArgs(
            agent_args=GenericAgentArgs(
                # NOTE: this model ask for a 12GB GPU - sporadically, it might crash because the CUDA version is not compatible
                chat_model_args=CHAT_MODEL_ARGS_DICT["microsoft/Phi-3-mini-4k-instruct"],
                flags=basic_flags,
            ),
            env_args=EnvArgs(
                max_steps=5,
                task_seed=args.CrossProd([None] * 2),
                # task_name=args.CrossProd(tasks.miniwob_tiny_test),
                task_name=args.CrossProd(tasks.workarena_tasks[:2]),
            ),
            enable_debug=True,
        )
    )


model_name_list = [
    # "meta-llama/Meta-Llama-3-70B-Instruct",
    # "microsoft/Phi-3-mini-128k-instruct",
    # "codellama/CodeLlama-34b-Instruct-hf",
    # "Salesforce/xLAM-v0.1-r",
    # "deepseek-ai/deepseek-coder-6.7b-instruct",
    # "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # "microsoft/WizardLM-2-8x22B"
    # "finetuning/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]


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
)


def OSS_random_search(benchmark: str = "workarena"):
    """Example of random search. Modify this at will, but don't push your
    changes.

    The variance will usually be relatively high and the search space is soo big
    that the false discovery rate will be particularly high. Make sure to
    analyze the  results with caution and don't actually draw final conclusions
    from these experiments.

    TODO: eventually merge w/ exp_configs'
    """
    n_seeds = 3
    if benchmark == "miniwob":
        task_list = tasks.miniwob_all
        flags = MINIWOB_RS_OSS_FLAGS
    elif benchmark == "workarena":
        task_list = tasks.workarena_tasks
        flags = RS_OSS_FLAGS
    elif benchmark == "webarena":
        task_list = tasks.webarena_tasks
        flags = RS_OSS_FLAGS
        n_seeds = 1  # webarena doesn't have any randomness for a given task
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    return args.sample_and_expand_cross_product(
        ExpArgs(
            agent_args=GenericAgentArgs(
                chat_model_args=args.Choice([CHAT_MODEL_ARGS_DICT[k] for k in model_name_list]),
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
    use_concrete_example=False,
    use_abstract_example=False,
    multi_actions=False,
    action_space="bid",
    use_hints=False,
    use_screenshot=False,
)

# # TODO: get the flags automatically
# EXP_GROUPS["OSS_finetuning_eval"] = args.expand_cross_product(
#     ExpArgs(
#         agent_args=GenericAgentArgs(
#             chat_model_args=args.CrossProd(get_ckpt_list(CHAT_MODEL_ARGS_DICT[model_name_list[0]])),
#             # TODO: set these flags permanently
#             flags=Flags(
#                 use_html=False,
#                 use_ax_tree=True,
#                 drop_ax_tree_first=False,
#                 use_plan=False,
#                 use_criticise=False,
#                 use_thinking=False,
#                 use_think_history=False,
#                 use_error_logs=False,
#                 use_past_error_logs=False,
#                 use_history=True,
#                 use_action_history=True,
#                 use_memory=False,
#                 use_diff=True,
#                 use_concrete_example=False,
#                 use_abstract_example=False,
#                 multi_actions=False,
#                 action_space="bid",
#                 use_hints=False,
#                 use_screenshot=False,
#             ),
#         ),
#         max_steps=10,
#         task_seed=args.CrossProd([None] * 3),
#         task_name=args.CrossProd(tasks.workarena_tasks),
#         enable_debug=False,
#     ),
# )
