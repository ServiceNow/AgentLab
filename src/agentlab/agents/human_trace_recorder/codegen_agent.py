"""Simple Codegen Agent

Captures human interactions using playwright inspector.
Playwright trace logs are stored in "think" messages and can be viewed in Agentlab Xray.
"""

from __future__ import annotations

import json
import logging
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import bgym
from agentlab.agents.agent_args import AgentArgs
from browsergym.core.observation import (
    extract_dom_extra_properties,
    extract_dom_snapshot,
    extract_focused_element_bid,
    extract_merged_axtree,
    extract_screenshot,
)
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
from dotenv import load_dotenv
from playwright.sync_api import Page

load_dotenv()

def extract_log_message_from_pw_trace(pw_trace_file_path):
    zip_file = zipfile.ZipFile(pw_trace_file_path, "r")
    trace_lines = zip_file.read("trace.trace").decode("utf-8").splitlines()

    actions = []
    for line in trace_lines:
        if line.strip():
            event = json.loads(line)
            if event.get("type") == "log":
                actions.append(event)
    # Extract log messages from the trace
    return [log["message"].strip() for log in sorted(actions, key=lambda x: x.get("time", 0))]


def clean_pw_logs(logs, exclude_blacklist=True, use_substitutions=True):
    clean_logs = list(logs)
    blacklist = {
        "attempting click action",
        "waiting for element to be visible, enabled and stable",
        "element is visible, enabled and stable",
        "scrolling into view if needed",
        "done scrolling",
        "performing click action",
        "click action done",
        "waiting for scheduled navigations to finish",
        "navigations have finished",
    }

    substitutions = [("waiting for ", "")]

    def apply_substitutions(log):
        for old, new in substitutions:
            log = log.replace(old, new)
        return log

    if exclude_blacklist:
        clean_logs = [log for log in clean_logs if log not in blacklist]
    if use_substitutions:
        clean_logs = [apply_substitutions(log) for log in clean_logs]

    return clean_logs


@dataclass
class PlayWrightCodeGenAgentArgs(AgentArgs):
    agent_name: str = "PlayWrightCodeGenAgent"
    trace_dir: str = "playwright_codegen_traces"
    use_raw_page_output: bool = True
    store_raw_trace: bool = False

    def make_agent(self) -> bgym.Agent:  # type: ignore[override]
        return PlayWrightCodeGenAgent(self.trace_dir, self.store_raw_trace)

    def set_reproducibility_mode(self):
        pass


class PlayWrightCodeGenAgent(bgym.Agent):
    def __init__(self, trace_dir: str, store_raw_trace: bool):
        self.action_set = bgym.HighLevelActionSet(["bid"], multiaction=False)
        self._root = Path(trace_dir)
        self._page: Page | None = None
        self._step = 0
        self.store_raw_trace = store_raw_trace
        self._episode_trace_dir = None  # Cache for single episode

    def _get_trace_dir(self):
        """Return the trace directory based on store_raw_trace setting."""
        if self._episode_trace_dir is None:
            if self.store_raw_trace:
                import datetime

                dt_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self._episode_trace_dir = self._root / f"codegen_traces_{dt_str}"
                self._episode_trace_dir.mkdir(parents=True, exist_ok=True)
            else:
                self._episode_trace_dir = Path(tempfile.mkdtemp())
        return self._episode_trace_dir

    def obs_preprocessor(self, obs: dict):  # type: ignore[override]
        if isinstance(obs, dict):
            self._page = obs.get("page")
            obs["screenshot"] = extract_screenshot(self._page)
            obs["dom_object"] = extract_dom_snapshot(self._page)
            obs["axtree_object"] = extract_merged_axtree(self._page)
            scale_factor = getattr(self._page, "_bgym_scale_factor", 1.0)
            extra_properties = extract_dom_extra_properties(
                obs["dom_object"], scale_factor=scale_factor
            )
            obs["extra_element_properties"] = extra_properties
            obs["focused_element_bid"] = extract_focused_element_bid(self._page)

            if obs["axtree_object"]:
                obs["axtree_txt"] = flatten_axtree_to_str(obs["axtree_object"])

            if obs["dom_object"]:
                obs["dom_txt"] = flatten_dom_to_str(obs["dom_object"])
                obs["pruned_html"] = prune_html(obs["dom_txt"])

        if "page" in obs:  # unpickable
            del obs["page"]

        return obs

    def get_action(self, obs: dict):  # type: ignore[override]

        if self._page is None:
            raise RuntimeError("Playwright Page missing; ensure use_raw_page_output=True")

        page = self._page
        trace_dir = self._get_trace_dir()
        trace_path = trace_dir / f"step_{self._step}.zip"
        page.context.tracing.start(screenshots=True, snapshots=True, sources=True)
        page.context.tracing.start_chunk(name=f"step_{self._step}")

        print(
            f"{'─'*60}\n" f"Step {self._step}\n",
            f"{'─'*60}\n",
            "1. 🔴 Start Recording (Press 'Record' in the Playwright Inspector.)\n",
            "2. ✨ Perform actions for a single step.\n",
            "3. ⚫ Stop Recording (Press 'Record' again to stop recording.)\n",
            "4. ▶️  Press 'Resume' in the Playwright Inspector.",
        )

        page.pause()  # Launch Inspector and record actions
        page.context.tracing.stop_chunk(path=trace_path)
        page.context.tracing.stop()

        pw_logs = extract_log_message_from_pw_trace(trace_path)
        pw_logs = clean_pw_logs(pw_logs, exclude_blacklist=True)
        pw_logs_str = "\n".join([f"{i}. {log}" for i, log in enumerate(pw_logs, 1)])

        print(f"\n Playwright logs for step {self._step}:\n{pw_logs_str}")

        self._step += 1

        agent_info = bgym.AgentInfo(
            think=pw_logs_str,
            chat_messages=[],
            stats={},
        )

        return "noop()", agent_info


PW_CODEGEN_AGENT = PlayWrightCodeGenAgentArgs(store_raw_trace=True)


if __name__ == "__main__":
    from agentlab.agents.human_trace_recorder.codegen_agent import PW_CODEGEN_AGENT
    from agentlab.experiments.study import Study

    agent_configs = [PW_CODEGEN_AGENT]
    benchmark = bgym.DEFAULT_BENCHMARKS["workarena_l1"]()  # type: bgym.Benchmark
    benchmark = benchmark.subset_from_glob("task_name", "*create*")
    benchmark.env_args_list = benchmark.env_args_list[:1]
    for env_args in benchmark.env_args_list:
        print(env_args.task_name)
        env_args.max_steps = 15
        env_args.headless = False

    study = Study(agent_configs, benchmark, logging_level_stdout=logging.INFO)
    study.run(n_jobs=1, parallel_backend="sequential", n_relaunch=1)
