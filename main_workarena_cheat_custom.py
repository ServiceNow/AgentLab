"""
Run CheatingAgent with cheat_custom adapters on a small WorkArena subset.
"""

import os

# This is very import for runtime. DO NOT remove!
os.environ.pop("SNOW_INSTANCE_PWD", None)
os.environ.pop("SNOW_INSTANCE_URL", None)
os.environ.pop("SNOW_INSTANCE_UNAME", None)
# Do not remove or override: keep experiment outputs local to this repo.
os.environ.setdefault(
    "AGENTLAB_EXP_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "agentlab_results"),
)

import logging

import bgym

from agentlab.agents import CHEATING_AGENT
from agentlab.cheat_custom.workarena_adapters import register_workarena_cheat_customs
from agentlab.experiments.study import make_study

# TASK_IDS = [
#     # L1 tasks (kept for later):
#     "workarena.servicenow.all-menu",
#     "workarena.servicenow.filter-incident-list",
#     "workarena.servicenow.create-incident",
#     "workarena.servicenow.order-apple-watch",
#     # L3 tasks (navigate + do):
#     #"workarena.servicenow.navigate-and-create-incident-l3",
#     #"workarena.servicenow.navigate-and-filter-incident-list-l3",
#     #"workarena.servicenow.navigate-and-order-apple-watch-l3",
# ]

# Number of parallel jobs
n_jobs = 80
parallel_backend = "ray"
avg_step_timeout = 1200  # seconds per step used for Ray cancel timeout
max_steps = 50  # override WorkArena default episode length (was 15 in your env)

# Increase WorkArena Playwright default timeout (ms)
CHEATING_AGENT.snow_browser_timeout_ms = 120_000

if __name__ == "__main__":
    register_workarena_cheat_customs()

    benchmark = bgym.DEFAULT_BENCHMARKS["workarena_l2_agent_curriculum_eval"]()

    missing_l2_task_ids = set()
    missing_l2_base_ids = set()
    batch1_base_ids = {
        # Infeasible navigate-and-do
        "workarena.servicenow.infeasible-navigate-and-create-change-request-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-create-hardware-asset-l2",
        "workarena.servicenow.infeasible-navigate-and-create-hardware-asset-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-create-incident-l2",
        "workarena.servicenow.infeasible-navigate-and-create-problem-l2",
        "workarena.servicenow.infeasible-navigate-and-create-user-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-filter-asset-list-l2",
        "workarena.servicenow.infeasible-navigate-and-filter-asset-list-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-filter-change-request-list-l2",
        "workarena.servicenow.infeasible-navigate-and-filter-change-request-list-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-filter-hardware-list-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-filter-incident-list-l2",
        "workarena.servicenow.infeasible-navigate-and-filter-user-list-l2",
        "workarena.servicenow.infeasible-navigate-and-order-apple-watch-l2",
        "workarena.servicenow.infeasible-navigate-and-order-developer-laptop-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-order-ipad-mini-l2",
        "workarena.servicenow.infeasible-navigate-and-order-ipad-mini-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-order-ipad-pro-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-order-loaner-laptop-l2",
        "workarena.servicenow.infeasible-navigate-and-order-standard-laptop-l2",
        "workarena.servicenow.infeasible-navigate-and-sort-asset-list-l2",
        "workarena.servicenow.infeasible-navigate-and-sort-asset-list-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-sort-hardware-list-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-sort-incident-list-l2",
        "workarena.servicenow.infeasible-navigate-and-sort-incident-list-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-sort-service-catalog-item-list-with-reason-l2",
        "workarena.servicenow.infeasible-navigate-and-sort-user-list-l2",
        # Expense management
        "workarena.servicenow.amount-based-expense-management-large-l2",
        "workarena.servicenow.amount-based-expense-management-medium-l2",
        "workarena.servicenow.basic-expense-management-large-l2",
        "workarena.servicenow.basic-expense-management-medium-l2",
        "workarena.servicenow.basic-expense-management-small-l2",
        "workarena.servicenow.date-based-expense-management-large-l2",
        "workarena.servicenow.date-based-expense-management-medium-l2",
        "workarena.servicenow.date-based-expense-management-small-l2",
        "workarena.servicenow.easy-expense-management-large-l2",
        "workarena.servicenow.easy-expense-management-medium-l2",
        "workarena.servicenow.easy-expense-management-small-l2",
        # Maximize investment return
        "workarena.servicenow.filter-random-expenses-and-delete-wrong-investments-medium-l2",
        "workarena.servicenow.filter-random-expenses-and-find-total-return-large-l2",
        "workarena.servicenow.filter-random-expenses-and-find-total-return-medium-l2",
        "workarena.servicenow.filter-random-expenses-and-find-total-return-small-l2",
        "workarena.servicenow.filter-random-expenses-and-select-investments-large-l2",
        "workarena.servicenow.filter-random-expenses-and-select-investments-medium-l2",
        "workarena.servicenow.filter-random-expenses-and-select-investments-small-l2",
        "workarena.servicenow.filter-random-expenses-find-total-return-and-select-investments-medium-l2",
        "workarena.servicenow.filter-single-item-expenses-and-delete-wrong-investments-large-l2",
        "workarena.servicenow.filter-single-item-expenses-and-delete-wrong-investments-medium-l2",
        "workarena.servicenow.filter-single-item-expenses-and-find-total-return-large-l2",
        "workarena.servicenow.filter-single-item-expenses-and-find-total-return-medium-l2",
        "workarena.servicenow.filter-single-item-expenses-and-find-total-return-small-l2",
        "workarena.servicenow.filter-single-item-expenses-and-select-investments-medium-l2",
        "workarena.servicenow.filter-single-item-expenses-find-total-return-and-select-investments-medium-l2",
        "workarena.servicenow.filter-single-item-uniform-expenses-and-delete-wrong-investments-small-l2",
        "workarena.servicenow.filter-single-item-uniform-expenses-and-select-investments-large-l2",
        "workarena.servicenow.filter-single-item-uniform-expenses-and-select-investments-medium-l2",
        "workarena.servicenow.filter-single-item-uniform-expenses-find-total-return-and-select-investments-medium-l2",
        "workarena.servicenow.filter-three-items-uniform-expenses-and-select-investments-medium-l2",
        "workarena.servicenow.filter-three-items-uniform-expenses-find-total-return-and-select-investments-large-l2",
        "workarena.servicenow.filter-three-items-uniform-expenses-find-total-return-and-select-investments-medium-l2",
        "workarena.servicenow.filter-trivial-expenses-and-find-total-return-large-l2",
        "workarena.servicenow.filter-trivial-expenses-and-select-investments-large-l2",
        "workarena.servicenow.filter-trivial-expenses-find-total-return-and-select-investments-large-l2",
        "workarena.servicenow.filter-trivial-expenses-find-total-return-and-select-investments-small-l2",
        "workarena.servicenow.filter-two-items-uniform-expenses-and-select-investments-small-l2",
    }
    journal_path = os.path.join(
        os.path.dirname(__file__),
        "longmemevalv2_trajectory_collection_journal.json",
    )
    # Use the journal to filter out tasks that already have a successful trajectory.
    if os.path.exists(journal_path):
        import json

        with open(journal_path, "r", encoding="utf-8") as f:
            journal = json.load(f)
        for entry in journal:
            task_id = entry.get("task_id")
            if not task_id or "-l2_" not in task_id:
                continue
            records = entry.get("trajectory_records", [])
            if not any(r.get("reward") == 1 for r in records):
                missing_l2_task_ids.add(task_id)
                if "_" in task_id:
                    base, suffix = task_id.rsplit("_", 1)
                    if suffix.isdigit():
                        missing_l2_base_ids.add(base)

    # benchmark.env_args_list = [
    #     env_args for env_args in benchmark.env_args_list if env_args.task_name in TASK_IDS
    # ]
    if missing_l2_base_ids:
        missing_l2_base_ids = missing_l2_base_ids.intersection(batch1_base_ids)
        benchmark.env_args_list = [
            env_args
            for env_args in benchmark.env_args_list
            if env_args.task_name in missing_l2_base_ids
        ]
    print(
        "Running "
        f"{len(benchmark.env_args_list)} L2 tasks after filter "
        f"(batch1+missing-success)"
    )
    if not benchmark.env_args_list:
        raise SystemExit("No experiments to run after filtering.")

    for env_args in benchmark.env_args_list:
        env_args.headless = True
        env_args.max_steps = max_steps

    study = make_study(
        benchmark=benchmark,
        agent_args=[CHEATING_AGENT],
        comment="cheat_custom L2 missing-success subset",
    )
    study.avg_step_timeout = avg_step_timeout
    study.run(
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        strict_reproducibility=False,
        n_relaunch=3,
    )
