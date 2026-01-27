#!/usr/bin/env python3
"""
Scan successful WorkArena rerun trajectories and update the journal.

This script scans:
  /local/diwu/longmemeval-v2-data/workarena/successful_rerun_trajectories
and upserts successful trajectory paths into:
  enterprise/AgentLab/longmemevalv2_trajectory_collection_journal.json

Use --dry-run to preview changes without writing to the journal.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

BASE_DIR = Path("/local/diwu/longmemeval-v2-data/workarena/successful_rerun_trajectories")
JOURNAL_PATH = Path(__file__).resolve().parent / "longmemevalv2_trajectory_collection_journal.json"

_TIMESTAMP_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_")
_TASK_ID_RE = re.compile(r"on_(workarena\..+)$")


def find_trajectory_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted({p.parent for p in root.rglob("summary_info.json")})


def load_summary_info(traj_dir: Path) -> Dict:
    summary_path = traj_dir / "summary_info.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_task_id(dir_name: str) -> Optional[str]:
    if "_on_" in dir_name:
        return dir_name.split("_on_", 1)[1]
    if "-on-" in dir_name:
        return dir_name.split("-on-", 1)[1]
    match = _TASK_ID_RE.search(dir_name)
    if match:
        return match.group(1)
    return None


def parse_agent_model(dir_name: str) -> Tuple[Optional[str], Optional[str]]:
    segment = dir_name
    if "_on_" in segment:
        segment = segment.split("_on_", 1)[0]
    elif "-on-" in segment:
        segment = segment.split("-on-", 1)[0]

    if _TIMESTAMP_PREFIX_RE.match(segment):
        parts = segment.split("_", 2)
        if len(parts) == 3:
            segment = parts[2]

    if "-re-" in segment:
        segment = segment.split("-re-", 1)[0]

    if "-" not in segment:
        return None, None

    agent_part, model_part = segment.split("-", 1)
    agent = agent_part.strip().lower() if agent_part else None
    model = model_part.strip() if model_part else None
    return agent, model


def extract_reward(summary: Dict) -> Optional[int]:
    reward = summary.get("cum_reward")
    if reward is None:
        reward = summary.get("reward")
    if reward is None:
        reward = summary.get("final_reward")
    if reward is None:
        reward = summary.get("success")
    if reward is None:
        return None
    if isinstance(reward, bool):
        return int(reward)
    try:
        return int(reward)
    except (TypeError, ValueError):
        return None


def build_success_records(traj_dirs: Iterable[Path]) -> List[Dict]:
    records: List[Dict] = []
    for traj_dir in traj_dirs:
        summary = load_summary_info(traj_dir)
        reward = extract_reward(summary)
        if reward != 1:
            continue
        task_id = parse_task_id(traj_dir.name)
        if not task_id:
            print(f"[warn] Could not parse task_id from {traj_dir.name}")
            continue
        agent, model = parse_agent_model(traj_dir.name)
        if not agent or not model:
            print(f"[warn] Could not parse agent/model from {traj_dir.name}")
            continue
        records.append(
            {
                "task_id": task_id,
                "agent": agent,
                "model": model,
                "reward": 1,
                "path": str(traj_dir.resolve()),
            }
        )
    return records


def upsert_records(journal: List[Dict], records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    changes: List[Dict] = []
    entry_by_task = {entry["task_id"]: entry for entry in journal}

    for record in records:
        task_id = record["task_id"]
        entry = entry_by_task.get(task_id)
        action = "add"
        if entry is None:
            entry = {"task_id": task_id, "trajectory_records": []}
            journal.append(entry)
            entry_by_task[task_id] = entry
        else:
            action = "update"

        existing_idx = None
        for idx, existing in enumerate(entry.get("trajectory_records", [])):
            if existing.get("agent") == record["agent"] and existing.get("model") == record["model"]:
                existing_idx = idx
                break

        if existing_idx is None:
            entry.setdefault("trajectory_records", []).append(
                {
                    "agent": record["agent"],
                    "model": record["model"],
                    "reward": record["reward"],
                    "path": record["path"],
                }
            )
            changes.append({"action": "add", **record})
        else:
            existing = entry["trajectory_records"][existing_idx]
            if existing.get("reward") != record["reward"] or existing.get("path") != record["path"]:
                entry["trajectory_records"][existing_idx] = {
                    "agent": record["agent"],
                    "model": record["model"],
                    "reward": record["reward"],
                    "path": record["path"],
                }
                changes.append({"action": action, **record})

    return journal, changes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update the WorkArena trajectory collection journal with successful rerun paths.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the changes that would be applied without writing the journal.",
    )
    args = parser.parse_args()

    traj_dirs = find_trajectory_dirs(BASE_DIR)
    if not traj_dirs:
        print(f"No trajectories found under {BASE_DIR}")
        return

    success_records = build_success_records(traj_dirs)
    if not success_records:
        print("No successful trajectories found.")
        return

    if not JOURNAL_PATH.exists():
        raise SystemExit(f"Journal not found: {JOURNAL_PATH}")

    with JOURNAL_PATH.open("r", encoding="utf-8") as f:
        journal = json.load(f)

    updated_journal, changes = upsert_records(journal, success_records)

    if args.dry_run:
        print("Dry run: no changes written.")
        if changes:
            print(f"Would apply {len(changes)} change(s):")
            for change in changes:
                print(
                    "- "
                    f"{change['action']} {change['task_id']} "
                    f"agent={change['agent']} model={change['model']} "
                    f"reward={change['reward']} path={change['path']}"
                )
        else:
            print("No changes needed.")
        return

    if not changes:
        print("No changes needed.")
        return

    with JOURNAL_PATH.open("w", encoding="utf-8") as f:
        json.dump(updated_journal, f, indent=2)
        f.write("\n")

    print(f"Updated journal with {len(changes)} change(s): {JOURNAL_PATH}")


if __name__ == "__main__":
    main()
