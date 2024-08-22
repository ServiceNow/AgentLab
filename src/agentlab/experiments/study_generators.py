import logging

from browsergym.experiments.loop import ExpArgs

from agentlab.agents.agent_args import AgentArgs
from agentlab.agents.generic_agent.agent_configs import RANDOM_SEARCH_AGENT, AGENT_4o_MINI
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.experiments import args
from agentlab.experiments import task_collections as tasks
from agentlab.experiments.exp_utils import order


def run_agents_on_benchmark(
    agents: list[AgentArgs] | AgentArgs = AGENT_4o_MINI, benchmark: str = "miniwob"
):
    """Run one or multiple agents on a benchmark.

    Args:
        agents: list[AgentArgs] | AgentArgs
            The agent configuration(s) to run.

        benchmark: str
            The benchmark to use.

    Returns:
        study_name: str
        List[ExpArgs]
            A list of experiments to run.
    """

    if not isinstance(agents, (list, tuple)):
        agents = [agents]

    for agent in agents:
        agent.set_benchmark(benchmark)  # the agent can adapt (lightly?) to the benchmark

    env_args_list = tasks.get_benchmark_env_args(
        benchmark, meta_seed=43, max_steps=None, n_repeat=None, is_agent_curriculum=False
    )

    if len(agents) == 1:
        study_name = f"{agents[0].agent_name}_on_{benchmark}"
    else:
        study_name = f"{len(agents)}_agents_on_{benchmark}"

    return study_name, args.expand_cross_product(
        ExpArgs(
            agent_args=args.CrossProd(agents),
            env_args=args.CrossProd(env_args_list),
            logging_level=logging.DEBUG,
        )
    )


def random_search(
    agent_random_search: AgentArgs = RANDOM_SEARCH_AGENT,
    benchmark: str = "miniwob",
    n_samples=20,
):
    """
    Random search of agent args.

    The random search mechanism will recursively search through dataclasses and
    dict to find attributes of type args.Choice. It will sample iid and replace
    with the corresponding value.

    *WARINING* The standard errror of the experiment will usually be relatively high and
    the search space is usually big so the false discovery rate will likely be
    high. Make sure to analyze the results with caution and don't actually draw
    final conclusions from these experiments.

    Args:
        agent: AgentArgs
            The agent configuration, with some sub-arguments defined as args.Choice.

        benchmark: str
            The benchmark to use.

    Returns:
        study_name: str
        List[ExpArgs]
            A list of experiments to run.
    """

    agent_random_search.set_benchmark(benchmark)  # the agent can adapt (lightly?) to the benchmark

    env_args_list = tasks.get_benchmark_env_args(benchmark)
    study_name = f"random_search_of_{agent_random_search.agent_name}_on_{benchmark}"
    return study_name, args.sample_and_expand_cross_product(
        ExpArgs(
            agent_args=agent_random_search,
            env_args=args.CrossProd(env_args_list),
        ),
        n_samples=n_samples,  # number of samples
    )


def ablation_study(agent: AgentArgs = AGENT_4o_MINI, benchmark: str = "miniwob"):
    """Example of an ablation study for GenericAgent.

    This current implementation depends on the structure of GenericAgentArgs,
    Please get some inspiration from this and adapt to your own agent.
    """

    agent.set_benchmark(benchmark)  # the agent can adapt (lightly?) to the benchmark

    env_args_list = tasks.get_benchmark_env_args(benchmark)

    study_name = f"ablation_study_{agent.agent_name}_on_{benchmark}"
    return study_name, order(
        args.expand_cross_product(
            ExpArgs(
                agent_args=GenericAgentArgs(
                    chat_model_args=args.CrossProd([agent.chat_model_args]),
                    flags=args.make_ablation_study(
                        start_point=agent.flags,
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
            )
        )
    )


def demo_maker(agent: AgentArgs = AGENT_4o_MINI, benchmark: str = "miniwob"):
    """Runs in demo mode with video turned on.

    NOTE: to get blue pointers and visual effects, you need to set the demo_mode
    in the action space. `agent.flags.action.demo_mode` works for generic agent,
    but you might need to adapt it for other agents.

    Args:
        agent: AgentArgs
            The agent configuration.

        benchmark: str
            The benchmark to use.

    Returns:
        study_name: str
        List[ExpArgs]
            A list of experiments to run.
    """

    # TODO Need a better way to set demo_mode
    try:
        agent.flags.action.demo_mode = "all_blue"
    except AttributeError:
        pass

    env_args_list = tasks.get_benchmark_env_args(benchmark)
    for env_args in env_args_list:
        env_args.viewport = {"width": 1280, "height": 720}
        env_args.record_video = True
        env_args.wait_for_user_message = False
        env_args.slow_mo = 1000

    study_name = f"demo_of_{agent.agent_name}_on_{benchmark}"
    return study_name, args.expand_cross_product(
        ExpArgs(
            agent_args=agent,
            env_args=args.CrossProd(env_args_list),
        )
    )
