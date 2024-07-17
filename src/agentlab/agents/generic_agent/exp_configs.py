import logging
import random
from typing import List

from browsergym.experiments.loop import EnvArgs, ExpArgs

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent.agent_configs import (
    AGENT_3_5,
    AGENT_8B,
    AGENT_70B,
    AGENT_CUSTOM,
    AGENT_4o,
    AGENT_4o_VISION,
)
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.experiments import args
from agentlab.experiments import task_collections as tasks
from agentlab.experiments.exp_utils import get_ckpt_list, overwrite_chat_model_arg
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT


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


def generic_agent_test(agent=AGENT_3_5, benchmark="miniwob"):
    """Minimalistic experiment to test the system."""
    return args.expand_cross_product(
        ExpArgs(
            agent_args=GenericAgentArgs(
                chat_model_args=CHAT_MODEL_ARGS_DICT["azure/gpt-35-turbo/gpt-35-turbo"],
                flags=agent.flags,
            ),
            env_args=EnvArgs(
                max_steps=5,
                task_seed=args.CrossProd([None] * 2),
                task_name=args.CrossProd(tasks.miniwob_tiny_test),
            ),
            enable_debug=True,
        )
    )


DEFAULT_RS_FLAGS = GenericPromptFlags(
    flag_group="default_rs",
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


RANDOM_SEARCH_AGENT = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-2024-05-13"],
    flags=DEFAULT_RS_FLAGS,
)


def random_search(
    agent=RANDOM_SEARCH_AGENT,
    benchmark: str = "miniwob",
):
    """Example of random search. Modify this at will, but don't push your
        changes.

        The variance will usually be relatively high and the search space is soo big
        that the false discovery rate will be particularly high. Make sure to
        analyze the  results with caution and don't actually draw final conclusions
        from these experiments.
    ## you can also specify the experiment group name directly here to relaunch it
    # exp_group_name = "2024-01-22_23-46-25_random_search_prompt_OSS_LLMs"

    # WorkArena Ablation Study for ICML
    # exp_group_name = "2024-02-01_03-20-14_ablation_study_browsergym_workarena"

    # MiniWob Ablation Study for ICML
    # exp_group_name = "2024-02-01_03-24-01_ablation_study_browsergym_miniwob"


    # exp_group_name = get_most_recent_folder(RESULTS_DIR).name

    # relaunch_mode = "incomplete_only"
    # relaunch_mode = "all_errors"

    Args:
        agent: GenericAgentArgs
            The agent configuration, with some flags defined as args.Choice.
        benchmark: str
            The benchmark to use.

    Returns:
        List[ExpArgs]
            A list of experiments to run.
    """
    agent.flags = miniwob_add_html(benchmark, agent.flags)
    env_args_list = tasks.get_benchmark_env_args(benchmark)

    return args.sample_and_expand_cross_product(
        ExpArgs(
            agent_args=agent,
            env_args=args.CrossProd(env_args_list),
            enable_debug=False,
        ),
        n_samples=40,  # number of samples
    )


def progression_study(
    agent=AGENT_3_5,
    benchmark: str = "miniwob",
):
    """Example of a progression study. Modify this at will, but don't push your
    changes.

    Progression study are similar to ablation study. They start with a base
    configuration and a sequence of changes are applied to the base
    configuration progressively.

    Args:
        agent: GenericAgentArgs
            The agent configuration, with some flags defined as args.Choice.
        benchmark: str
            The benchmark to use.

    Returns:
        List[ExpArgs]
            A list of experiments to run.
    """
    flags = agent.flags

    flags = miniwob_add_html(benchmark, flags)
    env_args_list = tasks.get_benchmark_env_args(benchmark)

    return order(
        args.expand_cross_product(
            ExpArgs(
                agent_args=GenericAgentArgs(
                    chat_model_args=args.CrossProd([agent.chat_model_args]),
                    flags=args.make_progression_study(
                        start_point=flags,
                        changes=[
                            (".obs.use_error_logs", args.TOGGLE),  # use toggle for boolean flags
                            (".obs.use_past_error_logs", args.TOGGLE),
                            (".obs.use_ax_tree", args.TOGGLE),
                            (".action.multi_actions", args.TOGGLE),
                            (".obs.extract_coords", "center"),  # use string for categorical flags
                            (".action.action_set", "bid+coord"),
                            (".obs.extract_coords", "box"),
                            (".obs.extract_visible_tag", args.TOGGLE),
                        ],
                    ),
                ),
                env_args=args.CrossProd(env_args_list),
                enable_debug=False,
            )
        )
    )


def final_run(agent=AGENT_3_5, benchmark: str = "miniwob"):
    agent.flags = miniwob_add_html(benchmark, agent.flags)

    env_args_list = tasks.get_benchmark_env_args(
        benchmark, meta_seed=43, max_steps=None, n_repeat=None, is_agent_curriculum=False
    )

    return args.expand_cross_product(
        ExpArgs(
            agent_args=args.CrossProd([agent]),
            env_args=args.CrossProd(env_args_list),
            enable_debug=False,
            logging_level=logging.DEBUG,
        )
    )


def ablation_study(
    agent=AGENT_3_5,
    benchmark: str = "miniwob",
):
    flags = agent.flags

    flags = miniwob_add_html(benchmark, flags)
    env_args_list = tasks.get_benchmark_env_args(benchmark)

    return order(
        args.expand_cross_product(
            ExpArgs(
                agent_args=GenericAgentArgs(
                    chat_model_args=args.CrossProd([agent.chat_model_args]),
                    flags=args.make_ablation_study(
                        start_point=flags,
                        changes=[
                            (".action.multi_actions", args.TOGGLE),
                            (".action.long_description", args.TOGGLE),
                            (".action.individual_examples", args.TOGGLE),
                            (".obs.use_think_history", args.TOGGLE),
                            (".obs.use_past_error_logs", args.TOGGLE),
                            (".use_thinking", args.TOGGLE),
                        ],
                    ),
                ),
                env_args=args.CrossProd(env_args_list),
                enable_debug=False,
            )
        )
    )


def demo_maker(*a, **kw):
    """Runs in demo mode with video turned on"""
    flags = AGENT_4o.flags
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
