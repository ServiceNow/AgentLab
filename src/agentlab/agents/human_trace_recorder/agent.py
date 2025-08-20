"""Minimal Human Trace Agent (<200 lines)

Per step we capture ONLY:
  - axtree_txt, pruned_html, actions.json, after.html
  - Auto-resume after detecting user action
  - Visible recording indicator
"""

from __future__ import annotations

import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import bgym
from playwright.sync_api import Page

from agentlab.agents.agent_args import AgentArgs
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html


@dataclass
class HumanTraceAgentArgs(AgentArgs):
    agent_name: str = "HumanTraceAgent"
    trace_dir: str = "human_traces"
    use_raw_page_output: bool = True

    def make_agent(self) -> bgym.Agent:  # type: ignore[override]
        return HumanTraceAgent(self.trace_dir)

    def set_reproducibility_mode(self):
        pass


class HumanTraceAgent(bgym.Agent):
    def __init__(self, trace_dir: str):
        self.action_set = bgym.HighLevelActionSet(["bid"], multiaction=False)
        self._root = Path(trace_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._page: Page | None = None
        self._step = 0
        self._task_name = None
        self._seed = None

    def obs_preprocessor(self, obs: dict):  # type: ignore[override]
        if isinstance(obs, dict):
            if self._page is None and "page" in obs:
                self._page = obs["page"]

            # Extract task name and seed from obs if available
            if self._task_name is None:
                self._task_name = obs.get("task_name") or obs.get("task", {}).get(
                    "task_name", "unknown_task"
                )
            if self._seed is None:
                self._seed = obs.get("seed") or obs.get("task", {}).get("seed", "unknown_seed")

            dom = obs.get("dom_object")
            axt = obs.get("axtree_object")
            if axt is not None:
                try:
                    obs["axtree_txt"] = flatten_axtree_to_str(axt)
                except Exception:
                    pass
            if dom is not None:
                try:
                    obs["pruned_html"] = prune_html(flatten_dom_to_str(dom))
                except Exception:
                    pass
            for k in ("dom_object", "axtree_object", "page"):
                obs.pop(k, None)
        return obs

    def get_action(self, obs: dict):  # type: ignore[override]
        if self._page is None:
            raise RuntimeError("Playwright Page missing; ensure use_raw_page_output=True")

        page = self._page

        # Create directory structure: trace_dir/task_name/seed/step_XXXX
        task_dir = self._root / str(self._task_name or "unknown_task")
        seed_dir = task_dir / str(self._seed or "unknown_seed")
        step_dir = seed_dir / f"step_{self._step:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        trace_path = step_dir / "temp_trace.zip"
        actions_path = step_dir / "actions.json"

        print(
            f"[HumanTrace] Task: {self._task_name}, Seed: {self._seed}, Step {self._step}: Perform ONE action"
        )

        # Small recording indicator
        page.evaluate(
            """
            const div = document.createElement('div');
            div.id = '__rec';
            div.innerHTML = '🔴 REC';
            div.style.cssText = 'position:fixed;top:5px;right:5px;background:#f44;color:#fff;padding:5px 8px;border-radius:4px;font:bold 12px monospace;z-index:99999';
            document.body.appendChild(div);
        """
        )

        # Start tracing
        try:
            page.context.tracing.start(screenshots=True, snapshots=True)
            page.context.tracing.start_chunk()
        except Exception:
            pass

        # Wait for action
        self._wait_for_action(page)

        # Stop tracing and save
        try:
            page.context.tracing.stop_chunk(path=str(trace_path))
            actions = self._extract_trace(str(trace_path))
            actions_path.write_text(json.dumps(actions, indent=2))
            trace_path.unlink(missing_ok=True)
        except Exception:
            pass

        # Remove indicator
        page.evaluate("document.getElementById('__rec')?.remove()")

        # Save screenshot
        try:
            page.screenshot(path=str(step_dir / "screenshot.png"))
        except Exception:
            pass

        # Save HTML
        try:
            (step_dir / "after.html").write_text(page.content())
        except Exception:
            pass

        self._step += 1
        return "noop()", {
            "extra_info": {
                "step": self._step - 1,
                "task_name": self._task_name,
                "seed": self._seed,
                "trace_dir": str(step_dir),
            }
        }

    def _wait_for_action(self, page):
        """Wait for user action with auto-resume."""
        page.evaluate(
            """
            window.__acted = false;
            ['click','keydown','input','change'].forEach(e => 
                document.addEventListener(e, () => window.__acted = true, true)
            );
        """
        )

        start = time.time()
        while time.time() - start < 300:  # 5 min max
            try:
                if page.evaluate("window.__acted"):
                    page.evaluate("document.getElementById('__rec').innerHTML = '💾 SAVING'")
                    time.sleep(0.3)
                    return
            except Exception:
                pass
            time.sleep(0.1)

    def _extract_trace(self, trace_file: str):
        """Extract ALL events from trace zip."""
        all_events = []
        try:
            with zipfile.ZipFile(trace_file, "r") as zf:
                for name in zf.namelist():
                    if name.endswith(".trace"):
                        with zf.open(name) as f:
                            for line in f:
                                try:
                                    event = json.loads(line.decode())
                                    # Save everything - don't filter
                                    all_events.append(event)
                                except Exception:
                                    continue
        except Exception:
            pass
        return all_events


HUMAN_TRACE_AGENT = HumanTraceAgentArgs()
