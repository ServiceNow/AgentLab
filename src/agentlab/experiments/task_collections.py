from logging import warning
from pathlib import Path
import numpy as np
import pandas as pd
import time as t
import logging

t0 = t.time()
from browsergym.workarena import ALL_WORKARENA_TASKS, ATOMIC_TASKS, get_all_tasks_agents

logger = logging.getLogger(__name__)

dt = t.time() - t0
print(f"done importing workarena, took {dt:.2f} seconds")

from browsergym.webarena import ALL_WEBARENA_TASK_IDS
from browsergym.experiments import EnvArgs

# workarena_dashboard_tasks = [task_class.get_task_id() for task_class in DASHBOARD_TASKS]
# workarena_order_tasks = [task for task in workarena_tasks if "order" in task]
# workarena_sort_tasks = [task for task in workarena_tasks if "sort" in task]
# workarena_filter_tasks = [task for task in workarena_tasks if "filter" in task]


df = pd.read_csv(Path(__file__).parent / "miniwob_tasks_all.csv")
# append miniwob. to task_name column
df["task_name"] = "miniwob." + df["task_name"]
MINIWOB_ALL = df["task_name"].tolist()
tasks_eval = df[df["miniwob_category"].isin(["original", "additional", "hidden test"])][
    "task_name"
].tolist()
miniwob_debug = df[df["miniwob_category"].isin(["debug"])]["task_name"].tolist()
miniwob_tiny_test = ["miniwob.click-dialog", "miniwob.click-dialog-2"]

assert len(MINIWOB_ALL) == 125
assert len(tasks_eval) == 107
assert len(miniwob_debug) == 12
assert len(miniwob_tiny_test) == 2

# small set of task that should be a good indicator of the agent's performance
miniwob_allac_test = [
    "miniwob.use-slider-2",
    "miniwob.book-flight",  # long html
    "miniwob.hot-cold",  # many iterations + memory
    "miniwob.login-user-popup",  # challenge: it sometimes has a random popup that prevents the agent from logging in. Seed 43 has a popup.
    "miniwob.guess-number",
    "miniwob.copy-paste-2",  # requires ctrl+A befor ctrl+C and cmd on mac
    "miniwob.bisect-angle",  # requires good 2d understanding
]

suspisous_tasks = [
    "miniwob.choose-date",
    "miniwob.copy-paste",
    "miniwob.copy-paste-2",
    "miniwob.find-word",
    "miniwob.resize-textarea",
    "miniwob.text-transform",
    "miniwob.use-autocomplete",
    "miniwob.use-colorwheel",
    "miniwob.use-colorwheel-2",
    "miniwob.click-button-sequence",
    "miniwob.click-checkboxes-large",
]

# the best agent is able to solve these tasks some of the time but often fails
edge_tasks = [
    "miniwob.choose-date",
    "miniwob.click-scroll-list",
    "miniwob.count-shape",
    "miniwob.daily-calendar",
    "miniwob.drag-cube",
    "miniwob.drag-shapes",
    "miniwob.draw-line",
    "miniwob.email-inbox-forward",
    "miniwob.email-inbox-forward-nl",
    "miniwob.email-inbox-forward-nl-turk",
    "miniwob.form-sequence",
    "miniwob.form-sequence-2",
    "miniwob.hot-cold",
    "miniwob.resize-textarea",
    "miniwob.right-angle",
    "miniwob.sign-agreement",
    "miniwob.text-editor",
    "miniwob.use-slider-2",
    "miniwob.bisect-angle",
    "miniwob.choose-date-medium",
    "miniwob.choose-date-nodelay",
]

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
    """
    env_args_list = []
    rng = np.random.RandomState(meta_seed)

    filters = benchmark_name.split(".")
    benchmark_id = filters[0]
    if filters[0] == "workarena":
        # benchmark_id = "workarena." + filters[1]
        benchmark_id = benchmark_name
    # elif filters[0] == "miniwob":
    #     benchmark_id = "miniwob." + filters[0]

    max_steps_default = {
        "workarena.l1": 15,
        "workarena.l2": 30,
        "workarena.l3": 30,
        "webarena": 15,
        "miniwob": 10,
        "miniwob.click-menu-2": 10,
        "workarena.servicenow.all-menu": 10,
    }

    n_repeat_default = {
        "workarena.l1": 10,
        "workarena.l2": 1,
        "workarena.l3": 1,
        "webarena": 1,
        "miniwob": 5,
        "miniwob.click-menu-2": 10,
        "workarena.servicenow.all-menu": 10,
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

        if len(filters) < 2:
            raise ValueError(f"You must specify the sub set of workarena, e.g.: workarena.l2.")

        if benchmark_name == "workarena.servicenow.all-menu":
            task_names = [benchmark_name]
            env_args_list = _make_env_args(task_names, max_steps, n_repeat, rng)

        elif benchmark_name == "workarena.l1.sort":
            task_names = [task.get_task_id() for task in ATOMIC_TASKS]
            task_names = [task for task in task_names if "sort" in task]
            env_args_list = _make_env_args(task_names, max_steps, n_repeat, rng)

        else:
            for task, seed in get_all_tasks_agents(
                filter=".".join(filters[1:]), meta_seed=meta_seed, n_seed_l1=n_repeat
            ):
                task_name = task.get_task_id()
                env_args_list.append(
                    EnvArgs(task_name=task_name, task_seed=seed, max_steps=max_steps)
                )

    elif benchmark_name == "webarena":
        env_args_list = _make_env_args(ALL_WEBARENA_TASK_IDS, max_steps, n_repeat, rng)
    elif benchmark_name.startswith("miniwob"):
        if benchmark_name == "miniwob":
            env_args_list = _make_env_args(MINIWOB_ALL, max_steps, n_repeat, rng)
        else:
            env_args_list = _make_env_args([benchmark_name], max_steps, n_repeat, rng)

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
    env_args_list = get_benchmark_env_args("workarena.l1.sort")
    print(f"Number of tasks: {len(env_args_list)}")
    for env_args in env_args_list:
        print(env_args.task_seed, env_args.task_name)
