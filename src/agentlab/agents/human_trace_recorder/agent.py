from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass

import bgym
from playwright.sync_api import Page

from agentlab.agents.agent_args import AgentArgs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simplified variant: capture human step (trace + screenshot + html) only
# ---------------------------------------------------------------------------


@dataclass
class SimpleHumanTraceCaptureAgentArgs(AgentArgs):
    """Args for SimpleHumanTraceCaptureAgent.

    This version ONLY captures what the human does in the paused browser per step.
    It does NOT attempt to map or translate actions. Always returns noop().
    Set use_raw_page_output=True in loop/env so that obs contains a Playwright Page.
    """

    agent_name: str = "SimpleHumanTraceCapture"
    trace_dir: str = "human_traces"
    screenshots: bool = True
    snapshots: bool = True  # playwright tracing snapshots (DOM/Sources)
    sources: bool = False  # include source files (bigger trace)
    # Ensure the raw Playwright Page object is present in observations so we can pause.
    use_raw_page_output: bool = True

    def make_agent(self) -> bgym.Agent:
        return SimpleHumanTraceCaptureAgent(
            trace_dir=self.trace_dir,
            screenshots=self.screenshots,
            snapshots=self.snapshots,
            sources=self.sources,
        )

    def set_reproducibility_mode(self):
        pass


class SimpleHumanTraceCaptureAgent(bgym.Agent):
    """Minimal human-in-the-loop recorder.

    On each get_action:
      1. Start a Playwright tracing capture (if not already running for this step).
      2. Call page.pause() to open Inspector; user performs EXACTLY one logical action.
      3. Stop tracing, save trace zip, screenshot (after action), and HTML snapshot.
      4. Return noop() so the environment advances.

    Artifacts are stored under trace_dir/step_<n>/
    """

    def __init__(self, trace_dir: str, screenshots: bool, snapshots: bool, sources: bool):
        self.action_set = bgym.HighLevelActionSet(["bid"], multiaction=False)
        self._step_idx = 0
        from pathlib import Path

        self._root = Path(trace_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        # Store trace config booleans; Playwright tracing.start expects them as named params.
        self._trace_conf = dict(screenshots=screenshots, snapshots=snapshots, sources=sources)
        self._tracing_started = False  # track if global tracing has been started
        self._page: Page | None = None  # optional persistent page ref (when not in obs)

    def set_page(self, page: Page):
        """Manually inject a Playwright Page so the agent can function without it in obs.

        Call this once after you create / reset the environment if you prefer not to
        expose the page through observations (e.g., for safety or serialization reasons).
        """
        self._page = page

    def obs_preprocessor(self, obs):  # keep original obs so page is available
        return obs

    def get_action(self, obs: dict):  # type: ignore[override]
        import json
        import time

        # Resolve page priority: observation > stored page
        page: Page | None = obs.get("page") or self._page
        if page is None:
            raise RuntimeError(
                "No Playwright Page available. Provide use_raw_page_output=True OR call set_page(page)."
            )
        # Cache page if first time we see it via obs so later steps can omit it
        if self._page is None:
            self._page = page

        step_dir = self._root / f"step_{self._step_idx:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        trace_path = step_dir / "trace.zip"
        screenshot_path = step_dir / "after.png"
        html_path = step_dir / "after.html"

        # Lazy start of tracing (once per context) then per-step chunk
        if not self._tracing_started:
            try:
                page.context.tracing.start(
                    screenshots=self._trace_conf["screenshots"],
                    snapshots=self._trace_conf["snapshots"],
                    sources=self._trace_conf["sources"],
                )
                self._tracing_started = True
            except Exception as e:  # pragma: no cover
                print(f"[SimpleHumanTraceCapture][WARN] initial tracing.start failed: {e}")

        try:
            page.context.tracing.start_chunk()
        except Exception as e:  # pragma: no cover
            print(f"[SimpleHumanTraceCapture][WARN] tracing.start_chunk failed: {e}")

        print("\n[SimpleHumanTraceCapture] Perform ONE action then resume Inspector.")
        print("[SimpleHumanTraceCapture] A trace will be saved to:", trace_path)
        try:
            page.pause()
        except Exception as e:  # pragma: no cover
            print(f"[SimpleHumanTraceCapture][WARN] page.pause failed: {e}")

        # Stop current chunk & save
        try:
            page.context.tracing.stop_chunk(path=str(trace_path))
        except Exception as e:  # pragma: no cover
            print(f"[SimpleHumanTraceCapture][WARN] tracing.stop_chunk failed: {e}")

        # Post-action artifacts
        try:
            page.screenshot(path=str(screenshot_path))
        except Exception as e:  # pragma: no cover
            print(f"[SimpleHumanTraceCapture][WARN] screenshot failed: {e}")
        try:
            html = page.content()
            html_path.write_text(html)
        except Exception as e:  # pragma: no cover
            print(f"[SimpleHumanTraceCapture][WARN] html capture failed: {e}")

        meta = {
            "url": page.url,
            "timestamp": time.time(),
            "step": self._step_idx,
            "trace_path": str(trace_path),
            "screenshot_path": str(screenshot_path),
            "html_path": str(html_path),
        }
        (step_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        # --- Derive a lightweight human-readable script summary from the trace ---
        script_summary_lines: list[str] = []
        try:
            import json as _json
            import zipfile

            with zipfile.ZipFile(trace_path, "r") as zf:
                # Playwright trace usually contains one or more *.trace files (jsonl)
                trace_files = [n for n in zf.namelist() if n.endswith(".trace")]
                for tf in trace_files:
                    with zf.open(tf, "r") as fh:
                        for raw_line in fh:
                            try:
                                evt = _json.loads(raw_line.decode("utf-8"))
                            except Exception:
                                continue
                            if evt.get("type") != "action":
                                continue
                            a = evt.get("action", {})
                            api_name = a.get("apiName") or a.get("name") or "action"
                            selector = a.get("selector") or a.get("locator") or ""
                            value = a.get("value") or a.get("text") or ""
                            line = f"{api_name}"
                            if selector:
                                line += f" selector={selector!r}"
                            if value and isinstance(value, str) and len(value) < 200:
                                line += f" value={value!r}"
                            script_summary_lines.append(line)
            if not script_summary_lines:
                script_summary_lines.append("(no action events parsed from trace chunk)")
        except Exception as e:  # pragma: no cover
            script_summary_lines.append(f"(failed to parse trace for script summary: {e})")

        # Prepare chat messages (simple list of strings for easy viewing)
        chat_messages = [
            "PLAYWRIGHT TRACE STEP SUMMARY:",
            f"Step {self._step_idx} URL: {page.url}",
            "Actions:",
            *script_summary_lines,
            f"Trace file: {trace_path}",
            "Open with: npx playwright show-trace " + str(trace_path),
        ]

        self._step_idx += 1

        agent_info = bgym.AgentInfo(
            think="human-recorded",
            chat_messages=chat_messages,
            stats={"step": self._step_idx},
            markdown_page=textwrap.dedent(
                f"""### Simple Human Trace Capture\nSaved artifacts for step {meta['step']}:\n- URL: {meta['url']}\n- Trace: {meta['trace_path']}\n- Screenshot: {meta['screenshot_path']}\n- HTML: {meta['html_path']}\n"""
            ),
            extra_info=meta,
        )
        return "noop()", agent_info


SIMPLE_TRACE_CAPTURE_AGENT = SimpleHumanTraceCaptureAgentArgs()

##1. Simple debug agent
# 2. Instead of using the page object Launch codegen directly in a subprocess using the playwright codegen --url or somethiing
