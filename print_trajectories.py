#!/usr/bin/env python3
"""
Print goals, action sequences, and final rewards for AgentLab trajectories.

Usage:
  python print_trajectories.py /path/to/experiment_dir
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


STEP_RE = re.compile(r"step_(\d+)\.pkl\.gz$")
DEFAULT_WIDTH = 100
STUB_MODULE_PREFIXES = ("agentlab.", "browsergym.")


_STUB_CACHE: Dict[str, type] = {}


def _make_stub_class(module: str, name: str) -> type:
    key = f"{module}.{name}"
    if key not in _STUB_CACHE:
        _STUB_CACHE[key] = type(name, (), {"__module__": module})
    return _STUB_CACHE[key]


class SafeUnpickler(pickle.Unpickler):
    """Unpickler that stubs missing AgentLab/BrowserGym classes."""

    def find_class(self, module: str, name: str):  # type: ignore[override]
        if module.startswith(STUB_MODULE_PREFIXES):
            return _make_stub_class(module, name)
        return super().find_class(module, name)


def load_pickle(path: Path):
    try:
        with gzip.open(path, "rb") as f:
            return SafeUnpickler(f).load()
    except ModuleNotFoundError as exc:
        if exc.name == "numpy":
            raise RuntimeError(
                "numpy is required to read step_*.pkl.gz files. Install numpy and retry."
            ) from exc
        raise


def is_trajectory_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "summary_info.json").exists():
        return True
    return any(path.glob("step_*.pkl.gz"))


def find_trajectory_dirs(exp_dir: Path) -> List[Path]:
    if is_trajectory_dir(exp_dir):
        return [exp_dir]
    candidates = [p for p in exp_dir.iterdir() if is_trajectory_dir(p)]
    if candidates:
        return sorted(candidates)
    # Fallback: search recursively for step files and take parent dirs.
    dirs = {p.parent for p in exp_dir.rglob("step_*.pkl.gz")}
    return sorted(dirs)


def iter_step_files(traj_dir: Path) -> List[Path]:
    step_files: List[Tuple[int, Path]] = []
    for path in traj_dir.glob("step_*.pkl.gz"):
        match = STEP_RE.match(path.name)
        if match:
            step_files.append((int(match.group(1)), path))
    return [p for _, p in sorted(step_files)]


def load_summary_info(traj_dir: Path) -> Dict:
    summary_path = traj_dir / "summary_info.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_goal_from_object(goal_object) -> Optional[str]:
    if goal_object is None:
        return None
    if isinstance(goal_object, str):
        return goal_object
    if isinstance(goal_object, dict):
        for key in (
            "goal",
            "instruction",
            "task",
            "text",
            "objective",
            "description",
            "goal_text",
        ):
            val = goal_object.get(key)
            if isinstance(val, str) and val.strip():
                return val
        # Sometimes the goal is nested.
        for key in ("task", "goal", "instruction"):
            nested = goal_object.get(key)
            if isinstance(nested, dict):
                for nkey in ("goal", "instruction", "text", "description"):
                    val = nested.get(nkey)
                    if isinstance(val, str) and val.strip():
                        return val
    for attr in ("goal", "instruction", "task", "text", "objective", "description"):
        if hasattr(goal_object, attr):
            val = getattr(goal_object, attr)
            if isinstance(val, str) and val.strip():
                return val
    return str(goal_object)


def extract_goal(traj_dir: Path, first_step) -> Optional[str]:
    if getattr(first_step, "obs", None) is None or not isinstance(first_step.obs, dict):
        return None
    goal = first_step.obs.get("goal")
    if isinstance(goal, str) and goal.strip():
        return goal

    goal_object = first_step.obs.get("goal_object")
    if goal_object is None:
        goal_object_path = traj_dir / "goal_object.pkl.gz"
        if goal_object_path.exists():
            try:
                goal_object = load_pickle(goal_object_path)
            except Exception:
                goal_object = None
    return _extract_goal_from_object(goal_object)


def normalize_actions(action) -> List[str]:
    if action is None:
        return []
    if isinstance(action, (list, tuple)):
        flat: List[str] = []
        for item in action:
            if item is None:
                continue
            flat.append(str(item).strip())
        return [a for a in flat if a]
    action_str = str(action).strip()
    if not action_str:
        return []
    # If the action is a multi-line string, split into separate lines.
    if "\n" in action_str:
        lines = [line.strip() for line in action_str.splitlines() if line.strip()]
        return lines or [action_str]
    return [action_str]


def format_block(text: Optional[str], width: int, indent: str) -> str:
    if not text:
        return f"{indent}<goal not found>"
    paragraphs = str(text).splitlines()
    formatted: List[str] = []
    for para in paragraphs:
        if not para.strip():
            formatted.append(indent.rstrip())
            continue
        formatted.append(
            textwrap.fill(
                para,
                width=width,
                initial_indent=indent,
                subsequent_indent=indent,
            )
        )
    return "\n".join(formatted)


def format_actions(
    steps: List, width: int, indent: str
) -> List[str]:
    lines: List[str] = []
    for step in steps:
        step_idx = getattr(step, "step", None)
        actions = normalize_actions(getattr(step, "action", None))
        if not actions:
            continue
        for sub_idx, action in enumerate(actions, start=1):
            if step_idx is None:
                label = "??"
            elif len(actions) == 1:
                label = f"{step_idx:02d}"
            else:
                label = f"{step_idx:02d}.{sub_idx}"
            prefix = f"{indent}[{label}] "
            wrap_width = max(20, width - len(prefix))
            wrapped = textwrap.wrap(action, width=wrap_width) or [""]
            lines.append(prefix + wrapped[0])
            for extra in wrapped[1:]:
                lines.append(" " * len(prefix) + extra)
    if not lines:
        lines.append(f"{indent}<no actions found>")
    return lines


def render_trajectory(traj_dir: Path, width: int) -> str:
    step_files = iter_step_files(traj_dir)
    steps = [load_pickle(path) for path in step_files]
    summary = load_summary_info(traj_dir)

    goal = extract_goal(traj_dir, steps[0]) if steps else None
    actions_lines = format_actions(steps, width=width, indent="  ")
    action_count = sum(
        len(normalize_actions(getattr(step, "action", None))) for step in steps
    )

    header = f"Trajectory: {traj_dir.name}"
    lines: List[str] = []
    lines.append("=" * width)
    lines.append(header)
    lines.append(f"Steps: {len(steps)}  Actions: {action_count}")
    lines.append("-" * width)
    lines.append("Goal:")
    lines.append(format_block(goal, width=width, indent="  "))
    lines.append("-" * width)
    lines.append("Actions:")
    lines.extend(actions_lines)
    lines.append("-" * width)

    cum_reward = summary.get("cum_reward")
    cum_raw_reward = summary.get("cum_raw_reward")
    terminated = summary.get("terminated")
    truncated = summary.get("truncated")

    if summary:
        lines.append(
            f"Final rewards: cum_reward={cum_reward}  "
            f"cum_raw_reward={cum_raw_reward}  "
            f"terminated={terminated}  truncated={truncated}"
        )
    else:
        last_reward = getattr(steps[-1], "reward", None) if steps else None
        lines.append(f"Final rewards: <summary_info.json missing>  last_reward={last_reward}")

    lines.append("=" * width)
    return "\n".join(lines)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print goal, action sequence, and final rewards for trajectories."
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Experiment directory (either a run folder or a single trajectory folder).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Output width for pretty formatting (default: {DEFAULT_WIDTH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file path. Defaults to <experiment_dir>/trajectory_printout.log.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print to stdout instead of writing a log file.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    exp_dir = args.experiment_dir
    if not exp_dir.exists():
        print(f"Error: {exp_dir} does not exist.", file=sys.stderr)
        return 2
    if not exp_dir.is_dir():
        print(f"Error: {exp_dir} is not a directory.", file=sys.stderr)
        return 2

    traj_dirs = find_trajectory_dirs(exp_dir)
    if not traj_dirs:
        print(f"No trajectories found under {exp_dir}.", file=sys.stderr)
        return 1

    output_chunks: List[str] = []
    for idx, traj_dir in enumerate(traj_dirs):
        try:
            output_chunks.append(render_trajectory(traj_dir, width=args.width))
        except RuntimeError as exc:
            print(f"Error while reading {traj_dir}: {exc}", file=sys.stderr)
            return 1

    output_text = "\n\n".join(output_chunks) + "\n"

    if args.stdout:
        print(output_text, end="")
        return 0

    output_path = args.output or (exp_dir / "trajectory_printout.log")
    output_path.write_text(output_text, encoding="utf-8")
    print(f"Wrote trajectory log to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
