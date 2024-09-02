import logging
import time as t
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from browsergym.experiments import EnvArgs
from browsergym.webarena import ALL_WEBARENA_TASK_IDS

df = pd.read_csv(Path(__file__).parent / "miniwob_tasks_all.csv")
# append miniwob. to task_name column
df["task_name"] = "miniwob." + df["task_name"]
MINIWOB_ALL = df["task_name"].tolist()
tasks_eval = df[df["miniwob_category"].isin(["original", "additional", "hidden test"])][
    "task_name"
].tolist()
miniwob_debug = df[df["miniwob_category"].isin(["debug"])]["task_name"].tolist()
MINIWOB_TINY_TEST = ["miniwob.click-dialog", "miniwob.click-checkboxes"]

assert len(MINIWOB_ALL) == 125
assert len(tasks_eval) == 107
assert len(miniwob_debug) == 12
assert len(MINIWOB_TINY_TEST) == 2


webgum_tasks = [
    "miniwob.book-flight",
    "miniwob.choose-date",
    "miniwob.choose-date-easy",
    "miniwob.choose-date-medium",
    "miniwob.choose-list",
    "miniwob.click-button",
    "miniwob.click-button-sequence",
    "miniwob.click-checkboxes",
    "miniwob.click-checkboxes-large",
    "miniwob.click-checkboxes-soft",
    "miniwob.click-checkboxes-transfer",
    "miniwob.click-collapsible",
    "miniwob.click-collapsible-2",
    "miniwob.click-color",
    "miniwob.click-dialog",
    "miniwob.click-dialog-2",
    "miniwob.click-link",
    "miniwob.click-menu",
    "miniwob.click-option",
    "miniwob.click-pie",
    "miniwob.click-scroll-list",
    "miniwob.click-shades",
    "miniwob.click-shape",
    "miniwob.click-tab",
    "miniwob.click-tab-2",
    "miniwob.click-tab-2-hard",
    "miniwob.click-test",
    "miniwob.click-test-2",
    "miniwob.click-widget",
    "miniwob.count-shape",
    "miniwob.email-inbox",
    "miniwob.email-inbox-forward-nl",
    "miniwob.email-inbox-forward-nl-turk",
    "miniwob.email-inbox-nl-turk",
    "miniwob.enter-date",
    "miniwob.enter-password",
    "miniwob.enter-text",
    "miniwob.enter-text-dynamic",
    "miniwob.enter-time",
    "miniwob.focus-text",
    "miniwob.focus-text-2",
    "miniwob.grid-coordinate",
    "miniwob.guess-number",
    "miniwob.identify-shape",
    "miniwob.login-user",
    "miniwob.login-user-popup",
    "miniwob.multi-layouts",
    "miniwob.multi-orderings",
    "miniwob.navigate-tree",
    "miniwob.search-engine",
    "miniwob.social-media",
    "miniwob.social-media-all",
    "miniwob.social-media-some",
    "miniwob.tic-tac-toe",
    "miniwob.use-autocomplete",
    "miniwob.use-spinner",
]


# TODO add miniwob_tiny_test as benchmarks
def get_benchmark_env_args(
    benchmark_name: str, meta_seed=42, max_steps=None, n_repeat=None
) -> list[EnvArgs]:
    """
    Returns a list of EnvArgs for the given benchmark_name.

    Args:
        benchmark_name: A string representing the benchmark name.
        meta_seed: The seed for the random number generator.
        max_steps: None or int. The maximum number of steps for each task.
            if None, it will use the default value for the benchmark.
        n_repeat: None or int. The number of seeds for each task.
            if None, it will use the default value for the benchmark.
        is_agent_curriculum: wether to use the agent curriculum or the human curriculum.

    Returns:
        A list of EnvArgs.

    Raises:
        ValueError: If the benchmark_name is not recognized, or if the benchmark_name is not
            followed by a subcategory for workarena.
    """
    env_args_list = []
    rng = np.random.RandomState(meta_seed)

    filters = benchmark_name.split(".")
    benchmark_id = filters[0]
    if filters[0] == "workarena":
        benchmark_id = "workarena." + filters[1]

    max_steps_default = {
        "workarena.l1": 15,
        "workarena.l2": 50,
        "workarena.l3": 50,
        "webarena": 15,
        "miniwob": 10,
        "miniwob_tiny_test": 5,
    }

    n_repeat_default = {
        "workarena.l1": 10,
        "workarena.l2": 1,
        "workarena.l3": 1,
        "webarena": 1,
        "miniwob": 5,
        "miniwob_tiny_test": 2,
    }

    if max_steps is None:
        max_steps = max_steps_default[benchmark_id]
    if n_repeat is None:
        n_repeat = n_repeat_default[benchmark_id]
    else:
        if benchmark_id == "webarena" and n_repeat != 1:
            logger.warning(
                f"webarena is expected to have only one seed per task. Ignoring n_seeds_default = {n_repeat}"
            )
            n_repeat = 1

    if benchmark_name.startswith("workarena"):
        t0 = t.time()
        from browsergym.workarena import ALL_WORKARENA_TASKS, ATOMIC_TASKS, get_all_tasks_agents

        dt = t.time() - t0
        print(f"done importing workarena, took {dt:.2f} seconds")

        if len(filters) < 2:
            raise ValueError(f"You must specify the sub set of workarena, e.g.: workarena.l2.")

        if benchmark_name == "workarena.l1.sort":
            task_names = [task.get_task_id() for task in ATOMIC_TASKS]
            task_names = [task for task in task_names if "sort" in task]
            env_args_list = _make_env_args(task_names, max_steps, n_repeat, rng)

        else:
            for task, seed in get_all_tasks_agents(
                filter=".".join(filters[1:]),
                meta_seed=meta_seed,
                n_seed_l1=n_repeat,
            ):
                task_name = task.get_task_id()
                env_args_list.append(
                    EnvArgs(task_name=task_name, task_seed=seed, max_steps=max_steps)
                )

    elif benchmark_name == "webarena":
        from browsergym.webarena import ALL_WEBARENA_TASK_IDS

        env_args_list = _make_env_args(ALL_WEBARENA_TASK_IDS, max_steps, n_repeat, rng)
    elif benchmark_name.startswith("miniwob"):
        miniwob_benchmarks_map = {
            "miniwob": MINIWOB_ALL,
            "miniwob_tiny_test": MINIWOB_TINY_TEST,
        }
        env_args_list = _make_env_args(
            miniwob_benchmarks_map[benchmark_name], max_steps, n_repeat, rng
        )
    else:
        raise ValueError(f"Unknown benchmark name: {benchmark_name}")

    return env_args_list


def _make_env_args(task_list, max_steps, n_seeds_default, rng):
    env_args_list = []
    for task in task_list:
        for seed in rng.randint(0, 100, n_seeds_default):
            env_args_list.append(EnvArgs(task_name=task, task_seed=int(seed), max_steps=max_steps))
    return env_args_list


if __name__ == "__main__":
    env_args_list = get_benchmark_env_args("workarena.l2")
    print(f"Number of tasks: {len(env_args_list)}")
    for env_args in env_args_list:
        if "infeasible" in env_args.task_name:
            print(env_args.task_seed, env_args.task_name)
