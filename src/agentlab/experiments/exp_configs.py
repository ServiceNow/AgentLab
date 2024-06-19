import logging
from agentlab.experiments.exp_utils import get_ckpt_list, overwrite_chat_model_arg
from browsergym.experiments.loop import EnvArgs
import random
from browsergym.experiments.loop import EnvArgs, ExpArgs
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.agents import dynamic_prompting as dp
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

    exp_args_list = func()  # type: list[ExpArgs]

    not_filter_task = []
    filter_task = []
    has_webarena = False
    for exp_args in exp_args_list:
        task_name = exp_args.env_args.task_name

        if task_name.startswith("webarena"):
            has_webarena = True

        if task_name.startswith("workarena") and "sort" in task_name:
            filter_task.append(exp_args)
        else:
            not_filter_task.append(exp_args)

    # shuffle sepearately
    if not has_webarena:
        logging.info("Shuffling the task list.")
        random.shuffle(not_filter_task)
        random.shuffle(filter_task)

    exp_arg_list = not_filter_task + filter_task
    logging.info(f"{len(filter_task)}/{len(exp_arg_list)} are moved to the end.")
    return func_name, exp_arg_list


def make_seeds(n, offset=42):
    return [seed + offset for seed in range(n)]


def order(exp_args_list):
    for i, exp_args in enumerate(exp_args_list):
        exp_args.order = i
    return exp_args_list


def fix_flags(benchmark: str, mode: str = None, flags: GenericPromptFlags = None):
    if not mode:
        mode = "single_eval"
        logging.warning(f"Mode not specified, defaulting to {mode}")
    if mode not in ["single_eval", "rs", "OSS_eval", "OSS_rs", "finetuning_eval"]:
        raise ValueError(f"Unknown mode: {mode}")

    if mode == "single_eval":
        if benchmark == "miniwob":
            flags.obs.use_html = True

    if mode == "OSS_rs":
        if "miniwob" in benchmark:
            flags = MINIWOB_RS_OSS_FLAGS
        else:
            flags = RS_OSS_FLAGS

    if mode == "finetuning_eval":
        if "miniwob" in benchmark:
            flags = FINETUNING_MINIWOB_FLAGS
        else:
            flags = FINETUNING_FLAGS

    return flags


def get_n_seeds(benchmark: str, default_n_seeds: int = 5):
    if benchmark == "webarena":
        return 1
    if benchmark == "workarena.servicenow.all-menu":
        return 10
    if benchmark == "miniwob.click-menu-2":
        return 10

    return default_n_seeds


def get_task_list(benchmark: str):
    if benchmark == "miniwob":
        return tasks.miniwob_all
    elif benchmark == "workarena":
        return tasks.workarena_tasks
    elif benchmark == "webarena":
        return tasks.webarena_tasks
    else:
        # TODO: automate this
        return [benchmark]


def miniwob_add_html(benchmark: str, flags: GenericPromptFlags):
    if benchmark == "miniwob":
        flags.obs.use_html = True
    return flags


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
                chat_model_args=CHAT_MODEL_ARGS_DICT["meta-llama/Meta-Llama-3-8B-Instruct"],
                flags=basic_flags,
            ),
            env_args=EnvArgs(
                max_steps=5,
                task_seed=args.CrossProd([None] * 2),
                # task_name=args.CrossProd(tasks.miniwob_tiny_test),
                task_name=args.CrossProd(tasks.workarena_tasks_l1[:2]),
            ),
            enable_debug=True,
        )
    )


# use list_openai_models.py to get the latest list of models
model_name_list = [
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
    "meta-llama/Meta-Llama-3-8B-Instruct",
    # "meta-llama/Meta-Llama-3-70B-Instruct",
    # "microsoft/Phi-3-mini-128k-instruct",
    # "codellama/CodeLlama-34b-Instruct-hf",
    # "Salesforce/xLAM-v0.1-r",
    # "deepseek-ai/deepseek-coder-6.7b-instruct",
    # "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # "microsoft/WizardLM-2-8x22B"
    # "finetuning/Meta-Llama-3-8B-Instruct",
    # "finetuning/debug0",
    # "finetuning/debug1",
    # "finetuning/debug2",
]
# set to None or empty dict to keep the default values
overwrite_chat_model_args_dict = {
    # "model_url": "https://6be22ad6-b39e-4a6c-9449-c4d67f67e1af.job.console.elementai.com",
    # "max_total_tokens": 16_384,
}


# test GenericAgent with different LLMs
def generic_agent_eval_llm(benchmark="workarena.l1.sort"):
    """Evaluate GenericAgent with different LLMs on a selected benchmark."""
    flags = ADVANCED_FLAGS.copy()
    flags.obs.extract_visible_tag = True
    flags.obs.extract_clickable_tag = False
    flags.obs.use_think_history = False
    flags.obs.use_screenshot = False
    flags.obs.use_focused_element = True
    flags.obs.use_past_error_logs = False
    flags.use_hints = True
    flags.action.is_strict = False
    flags.action.multi_actions = False
    flags.action.action_set = "bid"
    flags.action.individual_examples = False
    flags.action.long_description = False

    # n_seeds = get_n_seeds(benchmark, default_n_seeds=5)
    # task_list = get_task_list(benchmark)

    flags = miniwob_add_html(benchmark, flags)
    env_args_list = tasks.get_benchmark_env_args(benchmark, max_steps=20, n_repeat=20)

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
            env_args=args.CrossProd(env_args_list),
            enable_debug=False,
            logging_level=logging.DEBUG,
        )
    )


def random_search(
    # benchmark: str = "workarena.servicenow.all-menu",
    # benchmark: str = "miniwob.click-menu-2",
    benchmark: str = "workarena.l1",
    mode="OSS_rs",
):
    """Example of random search. Modify this at will, but don't push your
    changes.

    The variance will usually be relatively high and the search space is soo big
    that the false discovery rate will be particularly high. Make sure to
    analyze the  results with caution and don't actually draw final conclusions
    from these experiments.
    """
    flags = fix_flags(benchmark, mode=mode, flags=DEFAULT_RS_FLAGS)
    # flags = miniwob_add_html(benchmark, DEFAULT_RS_FLAGS)

    env_args_list = tasks.get_benchmark_env_args(benchmark)

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
            env_args=args.CrossProd(env_args_list),
            enable_debug=False,
        ),
        n_samples=40,  # number of samples
    )


def progression_study(
    benchmark: str = "miniwob",
):
    """Example of a progression study. Modify this at will, but don't push your
    changes.

    Progression study are similar to ablation study. They start with a base
    configuration and a sequence of changes are applied to the base
    configuration progressively.
    """
    flags = GenericPromptFlags(
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

    flags = miniwob_add_html(benchmark, flags)
    env_args_list = tasks.get_benchmark_env_args(benchmark)

    chat_model_args_list = [CHAT_MODEL_ARGS_DICT[k] for k in model_name_list]
    if overwrite_chat_model_args_dict:
        chat_model_args_list = overwrite_chat_model_arg(
            chat_model_args_list, overwrite_chat_model_args_dict
        )

    return order(
        args.expand_cross_product(
            ExpArgs(
                agent_args=GenericAgentArgs(
                    chat_model_args=args.CrossProd(chat_model_args_list),
                    flags=args.make_progression_study(
                        start_point=flags,
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
                env_args=args.CrossProd(env_args_list),
                enable_debug=False,
            )
        )
    )


def final_run():
    benchmark = "miniwob"
    # benchmark = "workarena.l1"
    # benchmark = "workarena.l2"
    # benchmark = "webarena"

    # agent = AGENT_3_5
    # agent = AGENT_70B
    agent = AGENT_8B
    # agent = AGENT_4o
    # agent = AGENT_4o_VISION

    agent.flags = miniwob_add_html(benchmark, agent.flags)

    env_args_list = tasks.get_benchmark_env_args(
        benchmark, meta_seed=43, max_steps=None, n_repeat=None
    )

    return args.expand_cross_product(
        ExpArgs(
            agent_args=agent,
            env_args=args.CrossProd(env_args_list),
            enable_debug=False,
            logging_level=logging.INFO,
        )
    )


def ablation_study(
    benchmark: str = "workarena.l1",
    # benchmark: str = "miniwob",
):

    flags = GenericPromptFlags(
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
            extract_clickable_tag=False,
            extract_coords="False",
            filter_visible_elements_only=False,
        ),
        action=dp.ActionFlags(
            multi_actions=True,
            action_set="bid",
            long_description=False,
            individual_examples=False,
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

    # flags = fix_flags(benchmark, flags)
    # n_seeds = get_n_seeds(benchmark, default_n_seeds=5)
    # task_list = get_task_list(benchmark)

    flags = miniwob_add_html(benchmark, flags)
    env_args_list = tasks.get_benchmark_env_args(
        benchmark,
    )

    chat_model_args_list = [CHAT_MODEL_ARGS_DICT[k] for k in model_name_list]
    if overwrite_chat_model_args_dict:
        chat_model_args_list = overwrite_chat_model_arg(
            chat_model_args_list, overwrite_chat_model_args_dict
        )

    return order(
        args.expand_cross_product(
            ExpArgs(
                agent_args=GenericAgentArgs(
                    chat_model_args=args.CrossProd(chat_model_args_list),
                    flags=args.make_ablation_study(
                        start_point=flags,
                        changes=[
                            (".action.multi_actions", False),
                            # (".obs.filter_visible_elements_only", True),
                            (".action.long_description", True),
                            (".action.individual_examples", True),
                            # [
                            #     (".action.action_set", "bid+coord"),
                            #     (".obs.extract_coords", "center"),
                            # ],
                            # [
                            #     (".action.action_set", "bid+coord"),
                            #     (".obs.extract_coords", "box"),
                            # ],
                            # obs flags
                            (".obs.use_think_history", False),
                            (".obs.use_past_error_logs", False),
                            # [
                            #     (".obs.use_screenshot", True),
                            #     (".obs.use_som", True),
                            # ],
                            # agent features
                            (".use_thinking", False),
                        ],
                    ),
                ),
                env_args=args.CrossProd(env_args_list),
                enable_debug=False,
            )
        )
    )


def ablation_study_GPT_3_5(benchmark: str = "workarena.l1"):

    flags = miniwob_add_html(benchmark, FLAGS_GPT_3_5)
    env_args_list = tasks.get_benchmark_env_args(benchmark, n_repeat=5)

    return order(
        args.expand_cross_product(
            ExpArgs(
                agent_args=GenericAgentArgs(
                    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-3.5-turbo-1106"],
                    flags=args.make_ablation_study(
                        start_point=flags,
                        changes=[
                            # (".action.multi_actions", True),
                            # (".obs.filter_visible_elements_only", True),
                            (".action.long_description", True),
                            (".action.individual_examples", False),
                            # [
                            #     (".action.action_set", "bid+coord"),
                            #     (".obs.extract_coords", "center"),
                            # ],
                            # [
                            #     (".action.action_set", "bid+coord"),
                            #     (".obs.extract_coords", "box"),
                            # ],
                            # obs flags
                            (".obs.use_think_history", True),
                            (".obs.use_past_error_logs", True),
                            (".obs.use_action_history", False),
                            (".obs.extract_visible_tag", False),
                            (".obs.extract_clickable_tag", True),
                            # [
                            #     (".obs.use_screenshot", True),
                            #     (".obs.use_som", True),
                            # ],
                            # agent features
                            # (".use_thinking", False),
                        ],
                    ),
                ),
                env_args=args.CrossProd(env_args_list),
                enable_debug=False,
            )
        )
    )


def ablation_study_OSS(
    # benchmark: str = "workarena.l1",
    benchmark: str = "miniwob",
):

    flags = FLAGS_70B

    flags = miniwob_add_html(benchmark, flags)
    env_args_list = tasks.get_benchmark_env_args(benchmark, n_repeat=5)

    return order(
        args.expand_cross_product(
            ExpArgs(
                agent_args=GenericAgentArgs(
                    chat_model_args=args.CrossProd(
                        [CHAT_MODEL_ARGS_DICT[k] for k in model_name_list]
                    ),
                    flags=args.make_ablation_study(
                        start_point=flags,
                        changes=[
                            # action flags
                            (".action.multi_actions", True),
                            (".action.long_description", True),
                            # (".action.individual_examples", False),
                            # obs flags
                            (".obs.use_think_history", False),
                            (".obs.use_error_logs", True),
                            (".obs.use_action_history", False),
                            (".obs.extract_visible_tag", False),
                            # agent features
                            (".use_thinking", False),
                        ],
                    ),
                ),
                env_args=args.CrossProd(env_args_list),
                enable_debug=False,
            )
        )
    )


def ablation_study_GPT_4(benchmark: str = "workarena.l1"):

    flags = miniwob_add_html(benchmark, FLAGS_GPT_4o)
    env_args_list = tasks.get_benchmark_env_args(benchmark, n_repeat=5)

    return order(
        args.expand_cross_product(
            ExpArgs(
                agent_args=GenericAgentArgs(
                    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-2024-05-13"],
                    flags=args.make_ablation_study(
                        start_point=flags,
                        changes=[
                            (".action.multi_actions", True),
                            # (".obs.filter_visible_elements_only", True),
                            (".action.long_description", False),
                            (".action.individual_examples", False),
                            # [
                            #     (".action.action_set", "bid+coord"),
                            #     (".obs.extract_coords", "center"),
                            # ],
                            # [
                            #     (".action.action_set", "bid+coord"),
                            #     (".obs.extract_coords", "box"),
                            # ],
                            # obs flags
                            (".obs.use_think_history", True),
                            (".obs.use_past_error_logs", True),
                            # (".obs.use_action_history", False),
                            (".obs.extract_visible_tag", False),
                            (".obs.extract_clickable_tag", False),
                            # [
                            #     (".obs.use_screenshot", True),
                            #     (".obs.use_som", True),
                            # ],
                            # agent features
                            # (".use_thinking", False),
                        ],
                    ),
                ),
                env_args=args.CrossProd(env_args_list),
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
        task_name=args.CrossProd(tasks.workarena_tasks_l1),
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


def finetuning_eval(
    # benchmark: str = "workarena.servicenow.all-menu",
    # dataset_name: str = "AllMenuTask_240603",
    # benchmark: str = "miniwob.click-menu-2",
    # dataset_name: str = "miniwob_overfit_240523",
    benchmark: str = "workarena.l1",
    dataset_name: str = "ATOMIC_TASKS_240604",
):
    """Evaluate GenericAgent with different LLMs on a selected benchmark."""

    assert len(model_name_list) == 1, "Only one model is supported for finetuning eval"
    assert model_name_list[0].startswith(
        "finetuning/"
    ), "Only finetuning models are supported for finetuning eval"

    flags = fix_flags(
        benchmark,
        mode="finetuning_eval",
    )
    env_args_list = tasks.get_benchmark_env_args(benchmark, max_steps=15, n_repeat=5)

    chat_model_args_list = [CHAT_MODEL_ARGS_DICT[k] for k in model_name_list]
    if overwrite_chat_model_args_dict:
        chat_model_args_list = overwrite_chat_model_arg(
            chat_model_args_list, overwrite_chat_model_args_dict
        )

    model_path = chat_model_args_list[0].model_path
    new_model_path = model_path + f"/{dataset_name}"
    chat_model_args_list = overwrite_chat_model_arg(
        chat_model_args_list, {"model_path": new_model_path}
    )

    return args.expand_cross_product(
        ExpArgs(
            agent_args=GenericAgentArgs(
                chat_model_args=args.CrossProd(
                    get_ckpt_list(CHAT_MODEL_ARGS_DICT[model_name_list[0]])
                ),
                flags=flags,
            ),
            env_args=args.CrossProd(env_args_list),
            # enable_debug=True,
            enable_debug=False,
            logging_level=logging.INFO,
        )
    )


DEFAULT_RS_FLAGS = GenericPromptFlags(
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
        action_set=args.Choice(["bid", "bid+coord"]),
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
        # use_html=args.Choice([True, False], p=[0.75, 0.25]),
        # use_ax_tree=args.Choice([True, False], p=[0.75, 0.25]),
        use_html=False,
        use_ax_tree=True,
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
    use_retry_and_fit=True,
)

FINETUNING_MINIWOB_FLAGS = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=True,
        use_ax_tree=False,
        use_think_history=False,
        use_error_logs=False,
        use_past_error_logs=False,
        use_history=True,
        use_action_history=True,
        use_diff=False,
        # use_diff=True,
        html_type="pruned_html",
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
    use_concrete_example=True,
    use_abstract_example=True,
    use_hints=False,
)


FINETUNING_FLAGS = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        # use_html=True,
        # use_ax_tree=False,
        use_think_history=False,
        use_error_logs=False,
        use_past_error_logs=False,
        use_history=True,
        use_action_history=True,
        # use_diff=True,
        use_diff=False,
        html_type="pruned_html",
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
    use_concrete_example=True,
    use_abstract_example=True,
    use_hints=False,
)

FLAGS_GPT_3_5 = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=True,
        use_history=True,
        use_past_error_logs=False,
        use_action_history=True,
        use_think_history=False,
        use_diff=False,
        html_type="pruned_html",
        use_screenshot=False,
        use_som=False,
        extract_visible_tag=True,
        extract_clickable_tag=False,
        extract_coords="False",
        filter_visible_elements_only=False,
    ),
    action=dp.ActionFlags(
        multi_actions=False,
        action_set="bid",
        long_description=False,
        individual_examples=True,
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

FLAGS_8B = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=False,
        use_history=True,
        use_past_error_logs=False,
        use_action_history=True,
        use_think_history=False,
        use_diff=False,
        html_type="pruned_html",
        use_screenshot=False,
        use_som=False,
        extract_visible_tag=False,
        extract_clickable_tag=False,
        extract_coords="False",
        filter_visible_elements_only=False,
    ),
    action=dp.ActionFlags(
        multi_actions=True,
        action_set="bid",
        long_description=False,
        individual_examples=True,
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
    add_missparsed_messages=True,
    use_retry_and_fit=True,
)

FLAGS_70B = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=False,
        use_history=True,
        use_past_error_logs=False,
        use_action_history=True,
        use_think_history=True,
        use_diff=False,
        html_type="pruned_html",
        use_screenshot=False,
        use_som=False,
        extract_visible_tag=True,
        extract_clickable_tag=False,
        extract_coords="False",
        filter_visible_elements_only=False,
    ),
    action=dp.ActionFlags(
        multi_actions=False,
        action_set="bid",
        long_description=False,
        individual_examples=True,
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
    add_missparsed_messages=True,
    use_retry_and_fit=True,
)


FLAGS_GPT_4o = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=True,
        use_history=True,
        use_past_error_logs=False,
        use_action_history=True,
        use_think_history=False,
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
        multi_actions=False,
        action_set="bid",
        long_description=True,
        individual_examples=True,
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

FLAGS_GPT_4o_VISION = FLAGS_GPT_4o.copy()
FLAGS_GPT_4o_VISION.obs.use_screenshot = True
FLAGS_GPT_4o_VISION.obs.use_som = True


AGENT_3_5 = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-3.5-turbo-1106"],
    flags=FLAGS_GPT_3_5,
)

AGENT_70B = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["meta-llama/Meta-Llama-3-70B-Instruct"],
    flags=FLAGS_70B,
)

AGENT_8B = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["meta-llama/Meta-Llama-3-8B-Instruct"],
    flags=FLAGS_8B,
)

AGENT_4o = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-2024-05-13"],
    flags=FLAGS_GPT_4o,
)


AGENT_4o_VISION = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-2024-05-13"],
    flags=FLAGS_GPT_4o_VISION,
)
