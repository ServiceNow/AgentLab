from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path

from bgym import ExpArgs

from agentlab.agents.agent_args import AgentArgs
from agentlab.agents.generic_agent.agent_configs import RANDOM_SEARCH_AGENT, AGENT_4o_MINI
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.experiments import args
from agentlab.experiments import task_collections as tasks
from agentlab.experiments.exp_utils import order
from agentlab.experiments.launch_exp import run_experiments
from agentlab.experiments.exp_utils import RESULTS_DIR
from agentlab.experiments.reproducibility_util import (
    get_reproducibility_info,
    save_reproducibility_info,
    add_experiment_to_journal,
)


@dataclass
class Study:

    exp_args_list: list[ExpArgs] = None
    benchmark_name: str = None
    agent_names: list[str] = None
    dir: Path = None

    def run(self, n_jobs=1, parallel_backend="dask", strict_reproducibility=False):

        if self.exp_args_list is None:
            raise ValueError("exp_args_list is None. Please set exp_args_list before running.")

        self.make_dir()
        self.write_reproducibility_info(strict_reproducibility=strict_reproducibility)

        run_experiments(n_jobs, self.exp_args_list, self.dir, parallel_backend=parallel_backend)

    def append_to_journal(self):
        add_experiment_to_journal(self.dir)

    @property
    def name(self):
        if len(self.agent_names) == 1:
            return f"{self.agent_names[0]}_on_{self.benchmark_name}"
        else:
            return f"{len(self.agent_names)}_agents_on_{self.benchmark_name}"

    def make_dir(self, exp_root=RESULTS_DIR):
        if self.dir is None:
            dir_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{self.name}"
            self.dir = Path(exp_root) / dir_name
        self.dir.mkdir(parents=True, exist_ok=True)

    def write_reproducibility_info(self, comment=None, strict_reproducibility=False):
        info = get_reproducibility_info(
            self.agent_names,
            self.benchmark_name,
            comment,
            ignore_changes=not strict_reproducibility,
        )
        return save_reproducibility_info(self.dir, info, strict_reproducibility)


def run_agents_on_benchmark(
    agents: list[AgentArgs] | AgentArgs = AGENT_4o_MINI, benchmark: str = "miniwob"
):
    """Run one or multiple agents on a benchmark.

    Args:
        agents: list[AgentArgs] | AgentArgs
            The agent configuration(s) to run.

        benchmark: str
            The benchmark to use. One of:
                * miniwob
                * webarena
                * workarena.l1
                * workarena.l2
                * workarena.l3
                * miniwob_tiny_test

    Returns:
        study: Study
    """

    if not isinstance(agents, (list, tuple)):
        agents = [agents]

    for agent in agents:
        agent.set_benchmark(benchmark)  # the agent can adapt (lightly?) to the benchmark

    env_args_list = tasks.get_benchmark_env_args(
        benchmark, meta_seed=43, max_steps=None, n_repeat=None
    )

    exp_args_list = args.expand_cross_product(
        ExpArgs(
            agent_args=args.CrossProd(agents),
            env_args=args.CrossProd(env_args_list),
            logging_level=logging.DEBUG,
        )
    )

    return Study(
        exp_args_list=exp_args_list,
        benchmark_name=benchmark,
        agent_names=[a.agent_name for a in agents],
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
