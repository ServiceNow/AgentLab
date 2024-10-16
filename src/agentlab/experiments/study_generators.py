from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path

from bgym import ExpArgs, EnvArgs

from agentlab.agents.agent_args import AgentArgs
from agentlab.agents.generic_agent.agent_configs import RANDOM_SEARCH_AGENT, AGENT_4o_MINI
from agentlab.analyze import inspect_results
from agentlab.experiments import args
from agentlab.experiments import task_collections as tasks
from agentlab.experiments.launch_exp import run_experiments, relaunch_study
from agentlab.experiments.exp_utils import RESULTS_DIR
from agentlab.experiments import reproducibility_util as repro


@dataclass
class Study:
    """A study coresponds to one or multiple agents evaluated on a benchmark.

    This is part of the high level API to help keep experiments organized and reproducible.

    Attributes:
        exp_args_list: list[ExpArgs]
            The list of experiments to run.

        benchmark_name: str
            The name of the benchmark.

        agent_names: list[str]
            The names of the agents.

        dir: Path
            The directory where the results will be saved.

        suffix: str
            A suffix to add to the study name
    """

    exp_args_list: list[ExpArgs] = None
    benchmark_name: str = None
    agent_names: list[str] = None
    dir: Path = None
    suffix: str = ""  # used for adding a personnal comment to the study name

    def run(self, n_jobs=1, parallel_backend="joblib", strict_reproducibility=False):
        """Run all experiments in the study in parallel when possible.

        Args:
            n_jobs: int
                Number of parallel jobs.

            parallel_backend: str
                Parallel backend to use. Either "joblib", "dask" or "sequential".

            strict_reproducibility: bool
                If True, you will have to commit all your files before running the experiments.
        """

        if self.exp_args_list is None:
            raise ValueError("exp_args_list is None. Please set exp_args_list before running.")

        self.make_dir()
        self.write_reproducibility_info(strict_reproducibility=strict_reproducibility)

        run_experiments(n_jobs, self.exp_args_list, self.dir, parallel_backend=parallel_backend)
        report_df = self.get_report(ignore_cache=True)
        logging.info(f"Study {self.name} finished.")
        logging.info("\n" + str(report_df))

    def append_to_journal(self, strict_reproducibility=True):
        """Append the study to the journal.

        Args:
            strict_reproducibility: bool
                If True, incomplete experiments will raise an error.

        Raises:
            ValueError: If the reproducibility information is not compatible
                with the report.
        """
        repro.append_to_journal(
            self.load_reproducibility_info(),
            self.get_report(),
            strict_reproducibility=strict_reproducibility,
        )

    @property
    def name(self):
        if len(self.agent_names) == 1:
            study_name = f"{self.agent_names[0]}_on_{self.benchmark_name}"
        else:
            study_name = f"{len(self.agent_names)}_agents_on_{self.benchmark_name}"
        if self.suffix:
            study_name += f"_{self.suffix}"
        return study_name

    def make_dir(self, exp_root=RESULTS_DIR):
        if self.dir is None:
            dir_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{self.name}"

            self.dir = Path(exp_root) / dir_name
        self.dir.mkdir(parents=True, exist_ok=True)

    def write_reproducibility_info(self, comment=None, strict_reproducibility=False):
        info = repro.get_reproducibility_info(
            self.agent_names,
            self.benchmark_name,
            comment,
            ignore_changes=not strict_reproducibility,
        )
        return repro.save_reproducibility_info(self.dir, info, strict_reproducibility)

    def get_report(self, ignore_cache=False, ignore_stale=False):
        return inspect_results.get_study_summary(
            self.dir, ignore_cache=ignore_cache, ignore_stale=ignore_stale
        )

    def load_reproducibility_info(self):
        return repro.load_reproducibility_info(self.dir)


def make_relaunch_study(study_dir, relaunch_mode="incomplete_or_error"):
    """Create a study from an existing study directory.

    It will search for all experiments that needs to be relaunched depending on
    `relaunch_mode`.

    Args:
        study_dir: Path
            The directory where the experiments are saved.
        relaunch_mode: str
            Find all incomplete experiments and relaunch them.
            - "incomplete_only": relaunch only the incomplete experiments.
            - "incomplete_or_error": relaunch incomplete or errors.
    """
    study = Study(dir=study_dir)
    study.exp_args_list, _ = relaunch_study(study.dir, relaunch_mode=relaunch_mode)
    info = study.load_reproducibility_info()
    study.benchmark_name = info["benchmark"]
    study.agent_names = info["agent_names"]
    return study


def set_demo_mode(env_args_list: list[EnvArgs]):

    for env_args in env_args_list:
        env_args.viewport = {"width": 1280, "height": 720}
        env_args.record_video = True
        env_args.wait_for_user_message = False
        env_args.slow_mo = 1000


def run_agents_on_benchmark(
    agents: list[AgentArgs] | AgentArgs = AGENT_4o_MINI,
    benchmark: str = "miniwob",
    demo_mode=False,
    log_level=logging.INFO,
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
        agent.set_benchmark(benchmark, demo_mode)  # the agent can adapt (lightly?) to the benchmark

    env_args_list = tasks.get_benchmark_env_args(
        benchmark, meta_seed=43, max_steps=None, n_repeat=None
    )
    if demo_mode:
        set_demo_mode(env_args_list)

    exp_args_list = args.expand_cross_product(
        ExpArgs(
            agent_args=args.CrossProd(agents),
            env_args=args.CrossProd(env_args_list),
            logging_level=log_level,
        )
    )

    return Study(
        exp_args_list=exp_args_list,
        benchmark_name=benchmark,
        agent_names=[a.agent_name for a in agents],
    )


def ablation_study(start_agent: AgentArgs, changes, benchmark: str, demo_mode=False):
    """Ablation study of an agent.

    Changes is a list of tuples (path_to_attribute, value) to change in the agent
    configuration.

    Args:
        start_agent: AgentArgs
            The agent configuration to start from.

        changes: list[tuple]
            The changes to apply to the agent configuration.

        benchmark: str
            The benchmark to use.

        demo_mode: bool
            If True, the experiments will be run in demo mode.

    Returns:
        Study
    """
    agents = args.make_ablation_study(start_agent, changes)
    study = run_agents_on_benchmark(agents, benchmark, demo_mode=demo_mode)
    study.suffix = "ablation_study"
    return study


def random_search(
    random_agent: AgentArgs = RANDOM_SEARCH_AGENT,
    n_samples=10,
    benchmark: str = "miniwob",
    demo_mode=False,
):
    """
    Random search of AgentArgs (NOTE: not fully tested since refactoring)

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

        n_samples: int
            The number of samples to take.

        benchmark: str
            The benchmark to use.

        demo_mode: bool
            If True, the experiments will be run in demo mode.

    Returns:
        Study
    """
    agents = args.sample_and_expand_cross_product(random_agent, n_samples)
    study = run_agents_on_benchmark(agents, benchmark, demo_mode=demo_mode)
    study.suffix = "random_search"
    return study
