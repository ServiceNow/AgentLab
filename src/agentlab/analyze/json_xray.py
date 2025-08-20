#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio dashboard to browse conversation-style JSON datasets produced in your experiments.
- Scan a base folder for JSON files (e.g., */epoch_*/seed_*/dataset/*.json)
- Load a file and browse by trajectory and step
- See prompt/think/action/output/expected and rewards
- Search trajectories by keyword

Requirements: pip install gradio pandas
Run:
- python src/agentlab/analyze/json_xray.py --base /path/to/dir
- XRAY_BASE_DIR=/path/to/dir python src/agentlab/analyze/json_xray.py
"""
from __future__ import annotations
import json
import re
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse

import gradio as gr
import pandas as pd

# -------- Helpers ---------


def find_json_files(base_dir: str) -> List[str]:
    p = Path(base_dir)
    if not p.exists():
        return []
    # Look for likely dataset files
    patterns = [
        "**/dataset/*.json",
        "**/evals/**/*.json",
        "**/*.json",
    ]
    found = []
    for pat in patterns:
        for fp in p.glob(pat):
            # Prefer small-medium JSONs, skip huge checkpoints, shards, etc.
            name = fp.name.lower()
            if any(k in name for k in ["train", "valid", "val", "test", "dataset"]):
                found.append(str(fp.resolve()))
    # Dedup while preserving order
    dedup = []
    seen = set()
    for f in found:
        if f not in seen:
            seen.add(f)
            dedup.append(f)
    return dedup


def load_dataset(
    json_path: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a list of items in the JSON file")

    # Normalize step_key to int where possible and group by trajectory/trace id
    by_traj: Dict[str, List[Dict[str, Any]]] = {}
    for item in data:
        # Prefer explicit IDs if present
        traj_group_key = (
            item.get("trace_id")
            or item.get("trajectory_key")
            or item.get("trajectory_goal")
            or "<unknown>"
        )
        step_raw = item.get("step_id")
        if step_raw is None:
            step_raw = item.get("step_key")
        try:
            item["_step_idx"] = int(step_raw) if step_raw is not None else 0
        except Exception:
            item["_step_idx"] = 0
        by_traj.setdefault(traj_group_key, []).append(item)

    for k in by_traj:
        by_traj[k] = sorted(by_traj[k], key=lambda x: x.get("_step_idx", 0))

    return data, by_traj


def summarize(by_traj: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for traj, items in by_traj.items():
        goal = None
        matches = 0
        total = len(items)
        og_rewards = []
        match_rewards = []
        for it in items:
            goal = goal or it.get("trajectory_goal")
            if it.get("action") == it.get("expected_action"):
                matches += 1
            if it.get("og_reward") is not None:
                og_rewards.append(it.get("og_reward"))
            if it.get("match_reward") is not None:
                match_rewards.append(it.get("match_reward"))
        rows.append(
            {
                "trajectory_key": traj,
                "goal": goal,
                "steps": total,
                "match_rate": round(matches / total, 3) if total else None,
                "avg_og_reward": (
                    round(sum(og_rewards) / len(og_rewards), 3) if og_rewards else None
                ),
                "avg_match_reward": (
                    round(sum(match_rewards) / len(match_rewards), 3) if match_rewards else None
                ),
            }
        )
    df = (
        pd.DataFrame(rows)
        .sort_values(["match_rate", "steps"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return df


def extract_section(text: str, start_marker: str, end_marker: str) -> str | None:
    if not isinstance(text, str):
        return None
    try:
        start = text.find(start_marker)
        if start == -1:
            return None
        start += len(start_marker)
        end = text.find(end_marker, start)
        if end == -1:
            return text[start:].strip()
        return text[start:end].strip()
    except Exception:
        return None


def render_item(item: Dict[str, Any]) -> str:
    traj = item.get("trajectory_key", "")
    goal = item.get("trajectory_goal", "")
    step = item.get("_step_idx", 0)
    action = item.get("action")
    expected = item.get("expected_action")
    match = "✅" if action == expected else "❌"
    og_reward = item.get("og_reward")
    match_reward = item.get("match_reward")
    # Rename Prompt->Input; keep Output; drop Think/HTML/Privileged
    prompt = item.get("prompt", "")
    output = item.get("output", "")

    def fence(title: str, body: str | None):
        if not body:
            return f"\n**{title}:**\n_None_\n"
        return f"\n**{title}:**\n```\n{body}\n```\n"

    md = []
    md.append(
        f"### Trajectory\n- Key: `{traj}`\n- Goal: `{goal}`\n- Step: `{step}`\n- Action: `{action}`\n- Expected: `{expected}` {match}\n- og_reward: `{og_reward}`\n- match_reward: `{match_reward}`\n"
    )
    md.append(fence("Input", prompt))
    if output:
        md.append(fence("Output", output))
    return "\n".join(md)


# -------- Gradio App ---------

DEFAULT_BASE_DIR = str(Path(__file__).resolve().parent)

# If this repo layout matches yours, set a friendlier default:
CANDIDATE_DEFAULT = "/mnt/adea/data_rw/finetuning/emiliano_home/experiments/20250812_043826_on_policy_miniwob_random_tasks_v5"
if Path(CANDIDATE_DEFAULT).exists():
    DEFAULT_BASE_DIR = CANDIDATE_DEFAULT


def build_demo(default_base_dir: str) -> gr.Blocks:
    with gr.Blocks(title="Conversation Dataset Viewer", theme=gr.themes.Base()) as demo:
        gr.Markdown(
            """# Conversation Dataset Viewer
Use this to browse JSON conversation logs produced during finetuning/evaluation.
1) Enter a base folder and Scan for JSON files.
2) Pick a file to load.
3) Select a trajectory and step to view details.
"""
        )

        ds_state = gr.State(
            {}
        )  # { 'data': list, 'by_traj': dict, 'traj_list': list, 'summary': df }

        with gr.Row():
            base_dir = gr.Textbox(label="Base folder", value=default_base_dir, scale=5)
            scan_btn = gr.Button("Scan", variant="secondary", scale=1)
        found_files = gr.Dropdown(label="Found JSON files", choices=[], interactive=True)

        with gr.Row():
            load_btn = gr.Button("Load selected file", variant="primary")
            file_label = gr.Markdown("_No file loaded_")

        with gr.Row():
            search = gr.Textbox(
                label="Filter trajectories (substring match)",
                placeholder="e.g., miniwob.click-scroll-list or Trinidad",
            )

        with gr.Row():
            traj_dd = gr.Dropdown(label="Trajectory", choices=[], interactive=False)
            step_dd = gr.Dropdown(label="Step", choices=[], interactive=True)

        summary_df = gr.Dataframe(
            label="Summary by trajectory", interactive=True, row_count=(0, "dynamic")
        )

        with gr.Row():
            prev_btn = gr.Button("◀ Prev")
            next_btn = gr.Button("Next ▶")

        details_md = gr.Markdown("_Load a file to see details_")

        # ---- Callbacks ----

        def on_scan(base: str):
            files = find_json_files(base)
            return gr.update(choices=files, value=(files[0] if files else None))

        scan_btn.click(on_scan, inputs=[base_dir], outputs=[found_files])

        def on_load(file_path: str):
            if not file_path:
                return (
                    "_No file selected_",
                    gr.update(choices=[], value=None),
                    gr.update(choices=[], value=None),
                    pd.DataFrame(),
                    ds_state,
                    "_No file loaded_",
                )
            data, by_traj = load_dataset(file_path)
            traj_list = sorted(by_traj.keys())
            df = summarize(by_traj)
            # Initialize state
            new_state = {
                "file": file_path,
                "data": data,
                "by_traj": by_traj,
                "traj_list": traj_list,
                "summary": df,
            }
            ds_state.value = new_state
            file_md = (
                f"Loaded: `{file_path}`  \\\nTrajectories: {len(traj_list)} | Steps: {len(data)}"
            )
            # Preselect first traj and step
            first_traj = traj_list[0] if traj_list else None
            step_choices = [
                str(it.get("_step_idx", i)) for i, it in enumerate(by_traj.get(first_traj, []))
            ]
            first_step = step_choices[0] if step_choices else None

            # Render details for the first trajectory's first step
            if first_traj and by_traj.get(first_traj):
                first_step_data = by_traj[first_traj][0]
                details = render_item(first_step_data)
            else:
                details = "_No data available_"

            return (
                file_md,
                gr.update(choices=traj_list, value=first_traj),
                gr.update(choices=step_choices, value=first_step),
                df,
                new_state,
                details,
            )

        load_btn.click(
            on_load,
            inputs=[found_files],
            outputs=[file_label, traj_dd, step_dd, summary_df, ds_state, details_md],
        )

        def on_filter(query: str, state: Dict[str, Any]):
            if not state or not state.get("traj_list"):
                return gr.update(choices=[], value=None)
            trajs = state["traj_list"]
            if not query:
                return gr.update(choices=trajs, value=(trajs[0] if trajs else None))
            q = query.lower()
            filtered = []
            for t in trajs:
                # match on key or goal text from first item
                first = state["by_traj"][t][0]
                goal = (first.get("trajectory_goal") or "").lower()
                key_lower = t.lower()
                if q in key_lower or q in goal:
                    filtered.append(t)
            val = filtered[0] if filtered else None
            return gr.update(choices=filtered, value=val)

        search.change(on_filter, inputs=[search, ds_state], outputs=[traj_dd])

        def on_table_select(state: Dict[str, Any], evt: gr.SelectData):
            """Handle table row selection to update trajectory dropdown from table clicks"""
            # Validate state payload without triggering pandas truthiness
            if not isinstance(state, dict) or "summary" not in state:
                return gr.update(), gr.update(), "_No data loaded_"
            df = state["summary"]
            try:
                is_empty = df is None or (hasattr(df, "empty") and df.empty)
            except Exception:
                is_empty = True
            if is_empty:
                return gr.update(), gr.update(), "_No data_"

            # Resolve selected row index from event
            if isinstance(evt.index, (list, tuple)) and len(evt.index) > 0:
                row_idx = evt.index[0]
            else:
                row_idx = evt.index
            if not isinstance(row_idx, int) or row_idx < 0 or row_idx >= len(df):
                return gr.update(), gr.update(), f"_Invalid selection: row {row_idx}_"

            # Extract trajectory key and build step choices
            try:
                traj_key = df.iloc[row_idx]["trajectory_key"]
            except Exception as e:
                return gr.update(), gr.update(), f"_Error getting trajectory: {e}_"

            steps = state["by_traj"].get(traj_key, [])
            step_choices = [str(it.get("_step_idx", i)) for i, it in enumerate(steps)]
            first_step = step_choices[0] if step_choices else None
            details = render_item(steps[0]) if steps else "_No steps_"

            return (
                gr.update(value=traj_key),
                gr.update(choices=step_choices, value=first_step),
                details,
            )

        summary_df.select(
            on_table_select, inputs=[ds_state], outputs=[traj_dd, step_dd, details_md]
        )

        def on_select_traj(traj_key: str, state: Dict[str, Any]):
            if not traj_key or not state:
                return gr.update(choices=[], value=None), "_No trajectory selected_"
            steps = state["by_traj"].get(traj_key, [])
            step_choices = [str(it.get("_step_idx", i)) for i, it in enumerate(steps)]
            # Render first step details
            details = render_item(steps[0]) if steps else "_No steps_"
            return (
                gr.update(choices=step_choices, value=(step_choices[0] if step_choices else None)),
                details,
            )

        traj_dd.change(on_select_traj, inputs=[traj_dd, ds_state], outputs=[step_dd, details_md])

        def on_select_step(traj_key: str, step_key: str, state: Dict[str, Any]):
            if not traj_key or not step_key or not state:
                return "_Select a trajectory and step_"
            steps = state["by_traj"].get(traj_key, [])
            # Find by numeric step
            try:
                idx = int(step_key)
            except Exception:
                idx = 0
            # Map step idx to position if non-contiguous
            pos = 0
            for i, it in enumerate(steps):
                if it.get("_step_idx", i) == idx:
                    pos = i
                    break
            details = render_item(steps[pos]) if steps else "_No steps_"
            return details

        step_dd.change(on_select_step, inputs=[traj_dd, step_dd, ds_state], outputs=[details_md])

        def on_prev(traj_key: str, step_key: str, state: Dict[str, Any]):
            if not traj_key or not state:
                return gr.update(), "_No selection_"
            steps = state["by_traj"].get(traj_key, [])
            if not steps:
                return gr.update(), "_No steps_"
            # Find current position
            try:
                cur = int(step_key) if step_key is not None else steps[0].get("_step_idx", 0)
            except Exception:
                cur = steps[0].get("_step_idx", 0)
            positions = [it.get("_step_idx", i) for i, it in enumerate(steps)]
            if cur in positions:
                idx = positions.index(cur)
            else:
                idx = 0
            new_idx = max(0, idx - 1)
            new_step_key = str(positions[new_idx])
            return gr.update(value=new_step_key), render_item(steps[new_idx])

        def on_next(traj_key: str, step_key: str, state: Dict[str, Any]):
            if not traj_key or not state:
                return gr.update(), "_No selection_"
            steps = state["by_traj"].get(traj_key, [])
            if not steps:
                return gr.update(), "_No steps_"
            try:
                cur = int(step_key) if step_key is not None else steps[0].get("_step_idx", 0)
            except Exception:
                cur = steps[0].get("_step_idx", 0)
            positions = [it.get("_step_idx", i) for i, it in enumerate(steps)]
            if cur in positions:
                idx = positions.index(cur)
            else:
                idx = 0
            new_idx = min(len(steps) - 1, idx + 1)
            new_step_key = str(positions[new_idx])
            return gr.update(value=new_step_key), render_item(steps[new_idx])

        prev_btn.click(on_prev, inputs=[traj_dd, step_dd, ds_state], outputs=[step_dd, details_md])
        next_btn.click(on_next, inputs=[traj_dd, step_dd, ds_state], outputs=[step_dd, details_md])

    return demo


if __name__ == "__main__":
    # Parse optional base directory from CLI or env.
    parser = argparse.ArgumentParser(description="Launch Conversation Dataset Viewer (Gradio)")
    parser.add_argument(
        "--base",
        "--base-dir",
        dest="base_dir",
        help="Base folder to scan for JSON files",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        help="Positional base folder to scan (alternative to --base)",
    )
    args = parser.parse_args()

    env_base = os.getenv("XRAY_BASE_DIR")
    chosen_base = args.base_dir or args.directory or env_base or DEFAULT_BASE_DIR

    # Launch the app. Env overrides: GRADIO_SERVER_NAME, GRADIO_SERVER_PORT, GRADIO_SHARE
    demo = build_demo(chosen_base)
    demo.queue(max_size=64)
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    port_env = os.getenv("GRADIO_SERVER_PORT", "")
    server_port = int(port_env) if port_env.isdigit() else None  # None => pick a free port
    share = os.getenv("GRADIO_SHARE", "true").lower() in ("1", "true", "yes")
    demo.launch(server_name=server_name, server_port=server_port, share=share)
