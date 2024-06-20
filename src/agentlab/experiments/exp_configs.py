import logging
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
from agentlab.agents.generic_agent.configs import (
    AGENT_3_5,
    AGENT_70B,
    AGENT_4o,
    AGENT_4o_VISION,
)


def get_exp_args_list(func_name: str, *a, **kw):
    """Run func_name and return exp_arg_list"""
    func = globals()[func_name]

    exp_args_list = func(*a, **kw)  # type: list[ExpArgs]

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
    benchmark = "miniwob"
    flags = AGENT_70B.flags
    flags = miniwob_add_html(benchmark, flags)
    env_args_list = tasks.get_benchmark_env_args(benchmark, max_steps=5, n_repeat=2)[:4]

    return args.expand_cross_product(
        ExpArgs(
            agent_args=GenericAgentArgs(
                # NOTE: this model ask for a 12GB GPU - sporadically, it might crash because the CUDA version is not compatible
                chat_model_args=CHAT_MODEL_ARGS_DICT["meta-llama/Meta-Llama-3-8B-Instruct"],
                flags=flags,
            ),
            env_args=args.CrossProd(env_args_list),
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
    "openai/gpt-4o-2024-05-13",
    # ------------------ OSS ------------------------
    # "finetuning/Meta-Llama-3-8B-Instruct",
    # "meta-llama/Meta-Llama-3-8B-Instruct",
    # "meta-llama/Meta-Llama-3-70B-Instruct",
    # "microsoft/Phi-3-mini-128k-instruct",
    # "codellama/CodeLlama-34b-Instruct-hf",
    # "Salesforce/xLAM-v0.1-r",
    # "deepseek-ai/deepseek-coder-6.7b-instruct",
    # "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # "microsoft/WizardLM-2-8x22B"
    # "finetuning/Meta-Llama-3-8B-Instruct",
]


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

    flags = miniwob_add_html(benchmark, flags)

    env_args_list = tasks.get_benchmark_env_args(benchmark, max_steps=20, n_repeat=20)

    return args.expand_cross_product(
        ExpArgs(
            agent_args=GenericAgentArgs(
                chat_model_args=args.CrossProd([CHAT_MODEL_ARGS_DICT[k] for k in model_name_list]),
                flags=flags,
            ),
            env_args=args.CrossProd(env_args_list),
            enable_debug=False,
            logging_level=logging.DEBUG,
        )
    )


# def random_search(benchmark: str = "miniwob"):
#     """Example of random search. Modify this at will, but don't push your
#     changes.

#     The variance will usually be relatively high and the search space is soo big
#     that the false discovery rate will be particularly high. Make sure to
#     analyze the  results with caution and don't actually draw final conclusions
#     from these experiments.
#     """
#     flags = miniwob_add_html(benchmark, DEFAULT_RS_FLAGS)
#     env_args_list = tasks.get_benchmark_env_args(benchmark)

#     return args.sample_and_expand_cross_product(
#         ExpArgs(
#             agent_args=GenericAgentArgs(
#                 chat_model_args=args.Choice([CHAT_MODEL_ARGS_DICT[k] for k in model_name_list]),
#                 flags=flags,
#             ),
#             env_args=args.CrossProd(env_args_list),
#             enable_debug=False,
#         ),
#         n_samples=20,  # number of samples
#     )


def progression_study(benchmark: str = "miniwob"):
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

    return order(
        args.expand_cross_product(
            ExpArgs(
                agent_args=GenericAgentArgs(
                    chat_model_args=args.CrossProd(
                        [CHAT_MODEL_ARGS_DICT[k] for k in model_name_list]
                    ),
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


def final_run(benchmark: str = "miniwob", model_name: str = "gpt-3.5"):

    if model_name.lower() in ["gpt-3.5"]:
        agent = AGENT_3_5
    elif model_name.lower() in ["gpt-4o"]:
        agent = AGENT_4o
    elif model_name.lower() in ["gpt-4o-vision"]:
        agent = AGENT_4o_VISION
    elif model_name.lower() in ["llama3-70b"]:
        agent = AGENT_70B

    agent.flags = miniwob_add_html(benchmark, agent.flags)

    env_args_list = tasks.get_benchmark_env_args(benchmark, max_steps=None, n_repeat=None)

    return args.expand_cross_product(
        ExpArgs(
            agent_args=agent,
            env_args=args.CrossProd(env_args_list),
            enable_debug=False,
            logging_level=logging.DEBUG,
        )
    )


def ablation_study(benchmark: str = "workarena.l1"):

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

    flags = miniwob_add_html(benchmark, flags)
    env_args_list = tasks.get_benchmark_env_args(
        benchmark,
    )

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
    chat_model_args = AGENT_3_5.chat_model_args
    flags = AGENT_3_5.flags
    flags = miniwob_add_html(benchmark, flags)
    env_args_list = tasks.get_benchmark_env_args(benchmark, n_repeat=5)

    return order(
        args.expand_cross_product(
            ExpArgs(
                agent_args=GenericAgentArgs(
                    chat_model_args=chat_model_args,
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


def ablation_study_OSS(benchmark: str = "workarena.l1"):
    chat_model_args = AGENT_70B.chat_model_args
    flags = AGENT_70B.flags

    flags = miniwob_add_html(benchmark, flags)
    env_args_list = tasks.get_benchmark_env_args(benchmark, n_seeds_default=5)

    return order(
        args.expand_cross_product(
            ExpArgs(
                agent_args=GenericAgentArgs(
                    chat_model_args=chat_model_args,
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
                            (".use_abstract_example", False),
                            (".use_concrete_example", False),
                            (".use_plan", True),
                        ],
                    ),
                ),
                env_args=args.CrossProd(env_args_list),
                enable_debug=False,
            )
        )
    )


def ablation_study_GPT_4(benchmark: str = "workarena.l1"):
    chat_model_args = AGENT_4o.chat_model_args
    flags = AGENT_4o.flags

    flags = miniwob_add_html(benchmark, flags)
    env_args_list = tasks.get_benchmark_env_args(benchmark, n_repeat=5)

    return order(
        args.expand_cross_product(
            ExpArgs(
                agent_args=GenericAgentArgs(
                    chat_model_args=chat_model_args,
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

    env_args_list = tasks.get_benchmark_env_args("workarena.l1", max_steps=15, n_repeat=3)
    for env_args in env_args_list:
        env_args.viewport = {"width": 1280, "height": 720}
        env_args.record_video = True
        env_args.wait_for_user_message = False
        env_args.slow_mo = 1000

    return args.expand_cross_product(
        ExpArgs(
            agent_args=GenericAgentArgs(
                chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-2024-05-13"],
                flags=flags,
            ),
            env_args=args.CrossProd(env_args_list),
            enable_debug=False,
        )
    )
