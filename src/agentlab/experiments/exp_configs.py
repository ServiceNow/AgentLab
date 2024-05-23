from browsergym.experiments.loop import EnvArgs
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.agents import dynamic_prompting as dp
from browsergym.experiments.loop import ExpArgs
from agentlab.experiments import args
from agentlab.experiments import task_collections as tasks
from agentlab.agents.generic_agent.generic_agent_prompt import (
    GenericPromptFlags,
    BASIC_FLAGS,
    ADVANCED_FLAGS,
)
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT


def get_exp_args_list(func_name: str):
    """Run func_name and return exp_arg_list"""
    func = globals()[func_name]
    return func_name, func()


def make_seeds(n, offset=42):
    return [seed + offset for seed in range(n)]


def order(exp_args_list):
    for i, exp_args in enumerate(exp_args_list):
        exp_args.order = i
    return exp_args_list


def generic_agent_test():
    """Minimalistic experiment to test the system."""
    return args.expand_cross_product(
        ExpArgs(
            agent_args=GenericAgentArgs(
                chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-3.5-turbo-0125"],
                flags=BASIC_FLAGS,
            ),
            env_args=EnvArgs(
                max_steps=5,
                task_seed=args.CrossProd([None] * 2),
                task_name=args.CrossProd(tasks.miniwob_tiny_test),
            ),
            enable_debug=True,
        )
    )


def tgi_toolkit_test():
    """Minimalistic experiment to test the system."""
    basic_flags = BASIC_FLAGS.copy()
    basic_flags.obs.use_html = False
    basic_flags.obs.use_ax_tree = True
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


# use list_openai_models.py to get the latest list of models
model_name_list = [
    # "openai/gpt-4-1106-preview",
    # "openai/gpt-4-vision-preview",
    # "openai/gpt-4-1106-vision-preview",
    # "openai/gpt-3.5-turbo-1106",
    # "openai/gpt-3.5-turbo-0125",
    # "openai/gpt-3.5-turbo-0301",
    # "openai/gpt-3.5-turbo-16k-0613",
    # "openai/gpt-4-0314",
    # "openai/gpt-4-0613",
    # "openai/gpt-4-1106-preview",
    # "openai/gpt-4-turbo-2024-04-09",
    # "openai/gpt-4o-2024-05-13",
    # ------------------ OSS ------------------------
    # "finetuning/Meta-Llama-3-8B-Instruct",
    # "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    # "microsoft/Phi-3-mini-128k-instruct",
    # "codellama/CodeLlama-34b-Instruct-hf",
    # "Salesforce/xLAM-v0.1-r",
    # "deepseek-ai/deepseek-coder-6.7b-instruct",
    # "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # "microsoft/WizardLM-2-8x22B"
    # "finetuning/Meta-Llama-3-8B-Instruct",
]


# test GenericAgent with different LLMs
def generic_agent_eval_llm(benchmark="miniwob"):
    """Evaluate GenericAgent with different LLMs on a selected benchmark."""
    flags = ADVANCED_FLAGS.copy()
    n_seeds = 5
    if benchmark == "miniwob":
        flags.obs.use_html = True  # it's better to use HTML for miniwob
        task_list = tasks.miniwob_all
    elif benchmark == "workarena":
        task_list = tasks.workarena_tasks
    elif benchmark == "webarena":
        task_list = tasks.webarena_tasks
        n_seeds = 1  # webearana doesn't have any randomness for a given task
        # TODO(we need to not randomize task list if it's webarena)

    return args.expand_cross_product(
        ExpArgs(
            agent_args=GenericAgentArgs(
                chat_model_args=args.CrossProd([CHAT_MODEL_ARGS_DICT[k] for k in model_name_list]),
                flags=flags,
            ),
            env_args=EnvArgs(
                max_steps=5,
                task_seed=args.CrossProd(make_seeds(n_seeds)),
                task_name=args.CrossProd(task_list),
            ),
            enable_debug=False,
        )
    )


def random_search(benchmark: str = "miniwob"):
    """Example of random search. Modify this at will, but don't push your
    changes.

    The variance will usually be relatively high and the search space is soo big
    that the false discovery rate will be particularly high. Make sure to
    analyze the  results with caution and don't actually draw final conclusions
    from these experiments.
    """
    n_seeds = 3
    if benchmark == "miniwob":
        task_list = tasks.miniwob_all
    elif benchmark == "workarena":
        task_list = tasks.workarena_tasks
    elif benchmark == "webarena":
        task_list = tasks.webarena_tasks
        n_seeds = 1  # webarana doesn't have any randomness for a given task
    return args.sample_and_expand_cross_product(
        ExpArgs(
            agent_args=DEFAULT_RS_FLAGS,
            env_args=EnvArgs(
                max_steps=10,
                task_seed=args.CrossProd(make_seeds(n_seeds)),
                task_name=args.CrossProd(task_list),
            ),
            enable_debug=False,
        ),
        n_samples=20,  # number of samples
    )


def progression_study(benchmark: str = "miniwob"):
    """Example of a progression study. Modify this at will, but don't push your
    changes.

    Progression study are similar to ablation study. They start with a base
    configuration and a sequence of changes are applied to the base
    configuration progressively.
    """
    n_seeds = 10
    if benchmark == "miniwob":
        task_list = tasks.miniwob_all
    elif benchmark == "workarena":
        task_list = tasks.workarena_tasks
    elif benchmark == "webarena":
        task_list = tasks.webarena_tasks
        n_seeds = 1
    start_point = GenericPromptFlags(
        obs=dp.ObsFlags(
            use_html=True,
            use_ax_tree=True,
            use_focused_element=False,
            use_error_logs=False,
            use_history=True,
            use_past_error_logs=False,
            use_action_history=True,
            use_think_history=True,
            use_diff=False,
            html_type="pruned_html",
            use_screenshot=True,
            use_som=False,
            extract_visible_tag=False,
            extract_clickable_tag=True,
            extract_coords="False",
            filter_visible_elements_only=False,
        ),
        action=dp.ActionFlags(
            multi_actions=False,
            action_set="bid",
        ),
        use_plan=False,
        use_criticise=False,
        use_thinking=True,
        use_memory=False,
        use_concrete_example=True,
        use_abstract_example=True,
        use_hints=True,
        enable_chat=False,
        max_prompt_tokens=None,
        be_cautious=True,
        extra_instructions=None,
    )
    return order(
        args.expand_cross_product(
            ExpArgs(
                agent_args=GenericAgentArgs(
                    chat_model_args=args.CrossProd(
                        [CHAT_MODEL_ARGS_DICT[k] for k in model_name_list]
                    ),
                    flags=args.make_progression_study(
                        start_point=start_point,
                        changes=[
                            (".obs.use_error_logs", True),
                            (".obs.use_past_error_logs", True),
                            (".obs.use_ax_tree", True),
                            (".action.multi_actions", True),
                            (".obs.extract_coords", "center"),
                            (".action.action_set", "bid+coord"),
                            (".obs.extract_coords", "box"),
                            (".obs.extract_visible_tag", True),
                        ],
                    ),
                ),
                env_args=EnvArgs(
                    max_steps=10,
                    task_seed=args.CrossProd(make_seeds(n_seeds)),
                    task_name=args.CrossProd(task_list),
                ),
                enable_debug=False,
            )
        )
    )


def ablation_study():

    start_point = GenericPromptFlags(
        obs=dp.ObsFlags(
            use_html=False,
            use_ax_tree=True,
            use_focused_element=True,
            use_error_logs=True,
            use_history=True,
            use_past_error_logs=True,
            use_action_history=True,
            use_think_history=True,
            use_diff=False,
            html_type="pruned_html",
            use_screenshot=False,
            use_som=False,
            extract_visible_tag=True,
            extract_clickable_tag=True,
            extract_coords="False",
            filter_visible_elements_only=False,
        ),
        action=dp.ActionFlags(
            multi_actions=True,
            action_set="bid",
        ),
        use_plan=False,
        use_criticise=False,
        use_thinking=True,
        use_memory=False,
        use_concrete_example=True,
        use_abstract_example=True,
        use_hints=False,
        enable_chat=False,
        max_prompt_tokens=None,
        be_cautious=True,
        extra_instructions=None,
    )

    return order(
        args.expand_cross_product(
            ExpArgs(
                agent_args=GenericAgentArgs(
                    chat_model_args=args.CrossProd(
                        [CHAT_MODEL_ARGS_DICT[k] for k in model_name_list]
                    ),
                    flags=args.make_ablation_study(
                        start_point=start_point,
                        changes=[
                            (".action.multi_actions", False),
                            (".filter_visible_elements_only", True),
                            [
                                (".action.action_set", "bid+coord"),
                                (".obs.extract_coords", "center"),
                            ],
                            [
                                (".action.action_set", "bid+coord"),
                                (".obs.extract_coords", "box"),
                            ],
                            # obs flags
                            (".obs.use_history", False),
                            (".obs.use_screenshot", True),
                            [
                                (".obs.use_screenshot", True),
                                (".obs.use_som", True),
                            ]
                            # agent features
                            (".use_thinking", False),
                        ],
                    ),
                ),
                env_args=EnvArgs(
                    max_steps=10,
                    task_seed=args.CrossProd(make_seeds(5)),
                    task_name=args.CrossProd(tasks.miniwob_all),
                ),
                enable_debug=False,
            )
        )
    )


def demo_maker():
    """Runs in demo mode with video turned on"""
    flags = ADVANCED_FLAGS.copy()
    flags.obs.use_screenshot = True
    flags.action.demo_mode = "all_blue"

    env_args = EnvArgs(
        task_name=args.CrossProd(tasks.workarena_tasks),
        task_seed=args.CrossProd([None] * 3),
        max_steps=15,
        viewport={"width": 1280, "height": 720},
        record_video=True,
        wait_for_user_message=False,
        slow_mo=1000,
    )

    return args.expand_cross_product(
        ExpArgs(
            agent_args=GenericAgentArgs(
                chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-2024-05-13"],
                flags=flags,
            ),
            env_args=env_args,
            enable_debug=False,
        )
    )


DEFAULT_RS_FLAGS = GenericAgentArgs(
    chat_model_args=args.Choice([CHAT_MODEL_ARGS_DICT[k] for k in model_name_list]),
    flags=GenericPromptFlags(
        obs=dp.ObsFlags(
            use_html=True,
            use_ax_tree=args.Choice([True, False]),
            use_focused_element=False,
            use_error_logs=True,
            use_history=True,
            use_past_error_logs=args.Choice([True, False], p=[0.7, 0.3]),
            use_action_history=True,
            use_think_history=args.Choice([True, False], p=[0.7, 0.3]),
            use_diff=args.Choice([True, False], p=[0.3, 0.7]),
            html_type="pruned_html",
            use_screenshot=False,
            use_som=False,
            extract_visible_tag=args.Choice([True, False]),
            extract_clickable_tag=False,
            extract_coords=args.Choice(["center", "box"]),
            filter_visible_elements_only=args.Choice([True, False], p=[0.3, 0.7]),
        ),
        action=dp.ActionFlags(
            multi_actions=args.Choice([True, False], p=[0.7, 0.3]),
            action_set="bid",
            # action_set=args.Choice(["python", "bid", "coord",
            # "bid+coord"]),
        ),
        # drop_ax_tree_first=True, # this flag is no longer active, according to browsergym doc
        use_plan=args.Choice([True, False]),
        use_criticise=args.Choice([True, False], p=[0.7, 0.3]),
        use_thinking=args.Choice([True, False], p=[0.7, 0.3]),
        use_memory=args.Choice([True, False], p=[0.7, 0.3]),
        use_concrete_example=True,
        use_abstract_example=True,
        use_hints=args.Choice([True, False], p=[0.7, 0.3]),
        be_cautious=args.Choice([True, False]),
        enable_chat=False,
        max_prompt_tokens=None,
        extra_instructions=None,
    ),
)


MINIWOB_RS_OSS_FLAGS = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=args.Choice([True, False], p=[0.75, 0.25]),
        use_ax_tree=args.Choice([True, False], p=[0.75, 0.25]),
        use_focused_element=False,
        use_error_logs=args.Choice([True, False], p=[0.8, 0.2]),
        use_history=args.Choice([True, False], p=[0.8, 0.2]),
        use_past_error_logs=args.Choice([True, False], p=[0.5, 0.5]),
        use_action_history=args.Choice([True, False], p=[0.8, 0.2]),
        use_think_history=args.Choice([True, False], p=[0.5, 0.5]),
        use_diff=args.Choice([True, False], p=[0.5, 0.5]),
        html_type="pruned_html",
        use_screenshot=False,
        use_som=False,
        extract_visible_tag=False,
        extract_clickable_tag=False,
        extract_coords="False",
        filter_visible_elements_only=False,
    ),
    action=dp.ActionFlags(
        multi_actions=args.Choice([True, False], p=[0.5, 0.5]),
        action_set=args.Choice(["bid", "python"]),
        # action_set=args.Choice(["python", "bid", "coord",
        # "bid+coord"]),
    ),
    # drop_ax_tree_first=True, # this flag is no longer active, according to browsergym doc
    use_plan=args.Choice([True, False], p=[0.25, 0.75]),
    use_criticise=args.Choice([True, False], p=[0.25, 0.75]),
    use_thinking=args.Choice([True, False], p=[0.5, 0.5]),
    use_memory=args.Choice([True, False], p=[0.25, 0.75]),
    use_concrete_example=True,
    use_abstract_example=True,
    use_hints=args.Choice([True, False], p=[0.25, 0.75]),
    be_cautious=True,
    enable_chat=False,
    max_prompt_tokens=None,
    extra_instructions=None,
)


RS_OSS_FLAGS = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=args.Choice([True, False], p=[0.75, 0.25]),
        use_ax_tree=args.Choice([True, False], p=[0.75, 0.25]),
        use_focused_element=False,
        use_error_logs=args.Choice([True, False], p=[0.5, 0.5]),
        use_history=args.Choice([True, False], p=[0.8, 0.2]),
        use_past_error_logs=args.Choice([True, False], p=[0.5, 0.5]),
        use_action_history=args.Choice([True, False], p=[0.8, 0.2]),
        use_think_history=args.Choice([True, False], p=[0.5, 0.5]),
        use_diff=args.Choice([True, False], p=[0.5, 0.5]),
        html_type="pruned_html",
        use_screenshot=False,
        use_som=False,
        extract_visible_tag=False,
        extract_clickable_tag=False,
        extract_coords="False",
        filter_visible_elements_only=False,
    ),
    action=dp.ActionFlags(
        multi_actions=args.Choice([True, False], p=[0.5, 0.5]),
        action_set=args.Choice(["bid", "python"]),
        # action_set=args.Choice(["python", "bid", "coord",
        # "bid+coord"]),
    ),
    # drop_ax_tree_first=True, # this flag is no longer active, according to browsergym doc
    use_plan=args.Choice([True, False], p=[0.25, 0.75]),
    use_criticise=args.Choice([True, False], p=[0.25, 0.75]),
    use_thinking=args.Choice([True, False], p=[0.5, 0.5]),
    use_memory=args.Choice([True, False], p=[0.25, 0.75]),
    use_concrete_example=True,
    use_abstract_example=True,
    use_hints=args.Choice([True, False], p=[0.25, 0.75]),
    be_cautious=True,
    enable_chat=False,
    max_prompt_tokens=None,
    extra_instructions=None,
)


FINETUNING_FLAGS = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_think_history=False,
        use_error_logs=False,
        use_past_error_logs=False,
        use_history=True,
        use_action_history=True,
        use_diff=True,
        use_screenshot=False,
    ),
    action=dp.ActionFlags(
        multi_actions=False,
        action_set="bid",
    ),
    use_plan=False,
    use_criticise=False,
    use_thinking=False,
    use_memory=False,
    use_concrete_example=False,
    use_abstract_example=False,
    use_hints=False,
)
