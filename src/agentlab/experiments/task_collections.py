from logging import warning
from pathlib import Path
import numpy as np
import pandas as pd
import time as t

t0 = t.time()
from browsergym.workarena import ALL_WORKARENA_TASKS, ATOMIC_TASKS, get_all_tasks_agents

dt = t.time() - t0
print(f"done importing workarena, took {dt:.2f} seconds")

from browsergym.webarena import ALL_WEBARENA_TASK_IDS
from browsergym.experiments import EnvArgs

workarena_tasks_all = [task_class.get_task_id() for task_class in ALL_WORKARENA_TASKS]
workarena_tasks_atomic = [task_class.get_task_id() for task_class in ATOMIC_TASKS]
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


TASK_CATEGORY_MAP = {
    "workarena.servicenow.all-menu": "menu",
    "workarena.servicenow.create-change-request": "form",
    "workarena.servicenow.create-hardware-asset": "form",
    "workarena.servicenow.create-incident": "form",
    "workarena.servicenow.create-problem": "form",
    "workarena.servicenow.create-user": "form",
    "workarena.servicenow.filter-asset-list": "list-filter",
    "workarena.servicenow.filter-change-request-list": "list-filter",
    "workarena.servicenow.filter-hardware-list": "list-filter",
    "workarena.servicenow.filter-incident-list": "list-filter",
    "workarena.servicenow.filter-service-catalog-item-list": "list-filter",
    "workarena.servicenow.filter-user-list": "list-filter",
    "workarena.servicenow.impersonation": "menu",
    "workarena.servicenow.knowledge-base-search": "knowledge",
    "workarena.servicenow.order-apple-mac-book-pro15": "service catalog",
    "workarena.servicenow.order-apple-watch": "service catalog",
    "workarena.servicenow.order-developer-laptop": "service catalog",
    "workarena.servicenow.order-development-laptop-p-c": "service catalog",
    "workarena.servicenow.order-ipad-mini": "service catalog",
    "workarena.servicenow.order-ipad-pro": "service catalog",
    "workarena.servicenow.order-loaner-laptop": "service catalog",
    "workarena.servicenow.order-sales-laptop": "service catalog",
    "workarena.servicenow.order-standard-laptop": "service catalog",
    "workarena.servicenow.sort-asset-list": "list-sort",
    "workarena.servicenow.sort-change-request-list": "list-sort",
    "workarena.servicenow.sort-hardware-list": "list-sort",
    "workarena.servicenow.sort-incident-list": "list-sort",
    "workarena.servicenow.sort-service-catalog-item-list": "list-sort",
    "workarena.servicenow.sort-user-list": "list-sort",
    "workarena.servicenow.dashboard-min-max-retrieval": "dashboard",
    "workarena.servicenow.dashboard-value-retrieval": "dashboard",
    "workarena.servicenow.report-value-retrieval": "dashboard",
    "workarena.servicenow.report-min-max-retrieval": "dashboard",
}


workarena_tasks_l1 = list(TASK_CATEGORY_MAP.keys())
workarena_task_categories = {}
for task in workarena_tasks_atomic:
    if task not in TASK_CATEGORY_MAP:
        warning(f"Atomic task {task} not found in TASK_CATEGORY_MAP")
        continue
    cat = TASK_CATEGORY_MAP[task]
    if cat in workarena_task_categories:
        workarena_task_categories[cat].append(task)
    else:
        workarena_task_categories[cat] = [task]


def get_task_category(task_name):
    benchmark = task_name.split(".")[0]
    return benchmark, TASK_CATEGORY_MAP.get(task_name, None)


def get_benchmark_env_args(benchmark_name: str, meta_seed=42, max_steps=None) -> list[EnvArgs]:

    env_args_list = []
    rng = np.random.RandomState(meta_seed)

    if benchmark_name.startswith("workarena"):

        filters = benchmark_name.split(".")
        if len(filters) < 2:
            raise ValueError(f"You must specify the sub set of workarena, e.g.: workarena.l2.")

        if max_steps is None:
            max_steps = {"l1": 15, "l2": 20, "l3": 20}[filters[1]]

        for task, seed in get_all_tasks_agents(filter=".".join(filters[1:]), meta_seed=meta_seed):
            task_name = task.get_task_id()
            env_args_list.append(EnvArgs(task_name=task_name, task_seed=seed, max_steps=max_steps))

    elif benchmark_name == "webarena":
        if max_steps is None:
            max_steps = 15

        for task_name in ALL_WEBARENA_TASK_IDS:
            seed = rng.randint(0, 100)
            env_args_list.append(EnvArgs(task_name=task_name, task_seed=seed, max_steps=max_steps))

    elif benchmark_name == "miniwob":
        if max_steps is None:
            max_steps = 10

        for task_name in MINIWOB_ALL:
            seed = rng.randint(0, 100)
            env_args_list.append(EnvArgs(task_name=task_name, task_seed=seed, max_steps=max_steps))
    else:
        raise ValueError(f"Unknown benchmark name: {benchmark_name}")

    return env_args_list


if __name__ == "__main__":
    for env_args in get_benchmark_env_args("workarena.l2"):
        print(env_args.task_name)
