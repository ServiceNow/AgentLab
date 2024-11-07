import gzip
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import bgym
from bgym import Benchmark, EnvArgs, ExpArgs

from agentlab.agents.agent_args import AgentArgs
from agentlab.analyze import inspect_results
from agentlab.experiments import args
from agentlab.experiments import reproducibility_util as repro
from agentlab.experiments.exp_utils import RESULTS_DIR
from agentlab.experiments.launch_exp import find_incomplete, run_experiments

logger = logging.getLogger("agentlab_" + __name__)


@dataclass
class Study:
    """A study coresponds to one or multiple agents evaluated on a benchmark.

    This is part of the high level API to help keep experiments organized and reproducible.

    Attributes:
        benchmark: Benchmark | str
            The benchmark to evaluate the agents on. If a string is provided, it will be
            converted to the corresponding benchmark using bgym.DEFAULT_BENCHMARKS.

        agent_args: list[AgentArgs]
            The list of agents to evaluate.

        dir: Path
            The directory where the results will be saved.

        suffix: str
            A suffix to add to the study name

        uuid: str
            A unique identifier for the study

        reproducibility_info: dict
            The reproducibility information for the study.
    """

    agent_args: list[AgentArgs] = None
    benchmark: Benchmark | str = None
    dir: Path = None
    suffix: str = ""  # used for adding a personnal comment to the study name
    uuid: str = None
    reproducibility_info: dict = None
    logging_level: int = logging.INFO
    logging_level_stdout: int = logging.INFO

    def __post_init__(self):
        self.uuid = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if isinstance(self.benchmark, str):
            self.benchmark = bgym.DEFAULT_BENCHMARKS[self.benchmark]()
        if isinstance(self.dir, str):
            self.dir = Path(self.dir)
        self.make_exp_args_list()

    def make_exp_args_list(self):
        self.exp_args_list = _agents_on_benchmark(
            self.agent_args,
            self.benchmark,
            logging_level=self.logging_level,
            logging_level_stdout=self.logging_level_stdout,
        )

    def find_incomplete(self, relaunch_mode="incomplete_or_error"):
        """Find incomplete or errored experiments in the study directory for relaunching."""
        self.exp_args_list = find_incomplete(self.dir, relaunch_mode=relaunch_mode)

    def load_exp_args_list(self):
        logger.info(f"Loading experiments from {self.dir}")
        self.exp_args_list = list(inspect_results.yield_all_exp_results(savedir_base=self.dir))

    def set_reproducibility_info(self, strict_reproducibility=False, comment=None):
        """Gather relevant information that may affect the reproducibility of the experiment

        e.g.: versions of BrowserGym, benchmark, AgentLab..."""
        agent_names = [a.agent_name for a in self.agent_args]
        info = repro.get_reproducibility_info(
            agent_names,
            self.benchmark,
            self.uuid,
            ignore_changes=not strict_reproducibility,
            comment=comment,
        )
        if self.reproducibility_info is not None:
            repro.assert_compatible(self.reproducibility_info, info)
        self.reproducibility_info = info

    def run(self, n_jobs=1, parallel_backend="joblib", strict_reproducibility=False, comment=None):
        """Run all experiments in the study in parallel when possible.

        Args:
            n_jobs: int
                Number of parallel jobs.

            parallel_backend: str
                Parallel backend to use. Either "joblib", "dask" or "sequential".

            strict_reproducibility: bool
                If True, all modifications have to be committed before running the experiments.
                Also, if relaunching a study, it will not be possible if the code has changed.
        """

        if self.exp_args_list is None:
            raise ValueError("exp_args_list is None. Please set exp_args_list before running.")

        logger.info("Preparing backends...")
        self.benchmark.prepare_backends()
        logger.info("Backends ready.")
        self.set_reproducibility_info(
            strict_reproducibility=strict_reproducibility, comment=comment
        )
        self.save()

        run_experiments(n_jobs, self.exp_args_list, self.dir, parallel_backend=parallel_backend)
        report_df = self.get_report(ignore_cache=True)
        logger.info(f"Study {self.name} finished.")
        logger.info("\n" + str(report_df))

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
            self.reproducibility_info,
            self.get_report(),
            strict_reproducibility=strict_reproducibility,
        )

    @property
    def name(self):
        agent_names = [a.agent_name for a in self.agent_args]
        if len(agent_names) == 1:
            study_name = f"{agent_names[0]}_on_{self.benchmark.name}"
        else:
            study_name = f"{len(agent_names)}_agents_on_{self.benchmark.name}"
        if self.suffix:
            study_name += f"_{self.suffix}"
        return study_name

    def make_dir(self, exp_root=RESULTS_DIR):
        if self.dir is None:
            dir_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{self.name}"

            self.dir = Path(exp_root) / dir_name
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self):
        """Pickle the study to the directory"""

        # TODO perhaps remove exp_args_list before pickling and when loading bring them from the individual directories

        self.make_dir()

        with gzip.open(self.dir / "study.pkl.gz", "wb") as f:
            pickle.dump(self, f)

    def get_report(self, ignore_cache=False, ignore_stale=False):
        return inspect_results.get_study_summary(
            self.dir, ignore_cache=ignore_cache, ignore_stale=ignore_stale
        )

    def load(dir: Path) -> "Study":
        dir = Path(dir)
        study_path = dir / "study.pkl.gz"
        if not study_path.exists() and dir.is_dir():
            # For backward compatibility
            first_result = next(
                inspect_results.yield_all_exp_results(savedir_base=dir, progress_fn=None)
            )
            benchmark_name = first_result.exp_args.env_args.task_name.split(".")[0]
            agent_args = first_result.exp_args.agent_args
            study = Study(agent_args=agent_args, benchmark=benchmark_name, dir=dir)
        else:
            with gzip.open(dir / "study.pkl.gz", "rb") as f:
                study = pickle.load(f)  # type: Study
            study.dir = dir

            # # just a check
            # for i, exp_args in enumerate(study.exp_args_list):
            #     if exp_args.order != i:
            #         logging.warning(f"The order of the experiments is not correct. {exp_args.order} != {i}")

        return study

    @staticmethod
    def load_most_recent(root_dir: Path = None):
        return Study.load(get_most_recent_study(root_dir))


def get_most_recent_study(
    root_dir: Path = None, date_format: str = "%Y-%m-%d_%H-%M-%S", contains=None
):
    """Return the most recent directory based on the date in the folder name.

    Args:
        root_dir: The directory to search in
        date_format: The format of the date in the folder name
        contains: If not None, only consider folders that contains this string

    Returns:
        Path: The most recent folder satisfying the conditions
    """

    if root_dir is None:
        root_dir = RESULTS_DIR

    most_recent_folder = None
    most_recent_time = datetime.min

    for item in root_dir.iterdir():
        if item.is_dir() and not item.name.startswith("_"):
            if contains is not None and contains not in item.name:
                continue
            try:
                folder_date = datetime.strptime("_".join(item.name.split("_")[:2]), date_format)
                if folder_date > most_recent_time:
                    most_recent_time = folder_date
                    most_recent_folder = item
            except (ValueError, IndexError):
                continue

    return most_recent_folder


# def make_relaunch_study(study_dir, relaunch_mode="incomplete_or_error"):
#     """Create a study from an existing study directory.

#     It will search for all experiments that needs to be relaunched depending on
#     `relaunch_mode`.

#     Args:
#         study_dir: Path
#             The directory where the experiments are saved.
#         relaunch_mode: str
#             Find all incomplete experiments and relaunch them.
#             - "incomplete_only": relaunch only the incomplete experiments.
#             - "incomplete_or_error": relaunch incomplete or errors.
#     """
#     study = Study(dir=study_dir)
#     study.exp_args_list, _ = find_incomplete(study.dir, relaunch_mode=relaunch_mode)
#     info = study.load_reproducibility_info()
#     study.benchmark_name = info["benchmark"]
#     study.agent_names = info["agent_names"]
#     return study


def set_demo_mode(env_args_list: list[EnvArgs]):

    for env_args in env_args_list:
        env_args.viewport = {"width": 1280, "height": 720}
        env_args.record_video = True
        env_args.wait_for_user_message = False
        env_args.slow_mo = 1000


def _agents_on_benchmark(
    agents: list[AgentArgs] | AgentArgs,
    benchmark: bgym.Benchmark,
    demo_mode=False,
    logging_level: int = logging.INFO,
    logging_level_stdout: int = logging.INFO,
):
    """Run one or multiple agents on a benchmark.

    Args:
        agents: list[AgentArgs] | AgentArgs
            The agent configuration(s) to run.
        benchmark: bgym.Benchmark
            The benchmark to run the agents on.
        demo_mode: bool
            If True, the experiments will be run in demo mode.
        logging_level: int
            The logging level for individual jobs.

    Returns:
        study: Study
    """

    if not isinstance(agents, (list, tuple)):
        agents = [agents]

    if benchmark.name.startswith("visualwebarena") or benchmark.name.startswith("webarena"):
        if len(agents) > 1:
            raise ValueError(
                f"Only one agent can be run on {benchmark.name} since the instance requires manual reset after each evaluation."
            )

    for agent in agents:
        agent.set_benchmark(benchmark, demo_mode)  # the agent can adapt (lightly?) to the benchmark

    env_args_list = benchmark.env_args_list
    if demo_mode:
        set_demo_mode(env_args_list)

    exp_args_list = args.expand_cross_product(
        ExpArgs(
            agent_args=args.CrossProd(agents),
            env_args=args.CrossProd(env_args_list),
            logging_level=logging_level,
            logging_level_stdout=logging_level_stdout,
        )
    )  # type: list[ExpArgs]

    for i, exp_args in enumerate(exp_args_list):
        exp_args.order = i

    _flag_sequential_exp(exp_args_list, benchmark)

    return exp_args_list


def _flag_sequential_exp(exp_args_list: list[ExpArgs], benchmark: Benchmark):
    if benchmark.name.startswith("visualwebarena"):
        sequential_subset = benchmark.subset_from_glob("requires_reset", "True")
        sequential_subset = set(
            [env_args.task_name for env_args in sequential_subset.env_args_list]
        )
        for exp_args in exp_args_list:
            if exp_args.env_args.task_name in sequential_subset:
                exp_args.sequential = True


# def ablation_study(start_agent: AgentArgs, changes, benchmark: str, demo_mode=False):
#     """Ablation study of an agent.

#     Changes is a list of tuples (path_to_attribute, value) to change in the agent
#     configuration.

#     Args:
#         start_agent: AgentArgs
#             The agent configuration to start from.

#         changes: list[tuple]
#             The changes to apply to the agent configuration.

#         benchmark: str
#             The benchmark to use.

#         demo_mode: bool
#             If True, the experiments will be run in demo mode.

#     Returns:
#         Study
#     """
#     agents = args.make_ablation_study(start_agent, changes)
#     study = run_agents_on_benchmark(agents, benchmark, demo_mode=demo_mode)
#     study.suffix = "ablation_study"
#     return study


# def random_search(
#     random_agent: AgentArgs = RANDOM_SEARCH_AGENT,
#     n_samples=10,
#     benchmark: str = "miniwob",
#     demo_mode=False,
# ):
#     """
#     Random search of AgentArgs (NOTE: not fully tested since refactoring)

#     The random search mechanism will recursively search through dataclasses and
#     dict to find attributes of type args.Choice. It will sample iid and replace
#     with the corresponding value.

#     *WARINING* The standard errror of the experiment will usually be relatively high and
#     the search space is usually big so the false discovery rate will likely be
#     high. Make sure to analyze the results with caution and don't actually draw
#     final conclusions from these experiments.

#     Args:
#         agent: AgentArgs
#             The agent configuration, with some sub-arguments defined as args.Choice.

#         n_samples: int
#             The number of samples to take.

#         benchmark: str
#             The benchmark to use.

#         demo_mode: bool
#             If True, the experiments will be run in demo mode.

#     Returns:
#         Study
#     """
#     agents = args.sample_and_expand_cross_product(random_agent, n_samples)
#     study = run_agents_on_benchmark(agents, benchmark, demo_mode=demo_mode)
#     study.suffix = "random_search"
#     return study
