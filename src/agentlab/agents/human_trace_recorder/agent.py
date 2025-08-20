"""Human Trace Agent for Browser Automation Training Data

Captures human interactions at each step including:
  - Comprehensive action tracking (clicks, input, navigation, etc.)
  - Saves only human_action.json files in simple numbered folders
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import bgym
from playwright.sync_api import Page

from agentlab.agents.agent_args import AgentArgs
from agentlab.agents.human_trace_recorder.event_listeners import (
    get_interaction_tracking_script,
    get_recording_indicators_script,
)
from browsergym.core.observation import (
    extract_dom_extra_properties,
    extract_dom_snapshot,
    extract_focused_element_bid,
    extract_merged_axtree,
    extract_screenshot,
)
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

    def obs_preprocessor(self, obs: dict):  # type: ignore[override]
        if isinstance(obs, dict):
            self._page = obs.get("page")
            # Remove the page object from obs to avoid pickle issues
            if "page" in obs:
                del obs["page"]

            obs["screenshot"] = extract_screenshot(self._page)
            obs["dom_object"] = extract_dom_snapshot(self._page)
            obs["axtree_object"] = extract_merged_axtree(self._page)
            scale_factor = getattr(self._page, "_bgym_scale_factor", 1.0)
            extra_properties = extract_dom_extra_properties(
                obs["dom_object"], scale_factor=scale_factor
            )
            obs["extra_element_properties"] = extra_properties
            obs["focused_element_bid"] = extract_focused_element_bid(self._page)

            # Add text representations for easier analysis
            if obs["axtree_object"]:
                axt = obs["axtree_object"]
                if extra_properties:
                    obs["axtree_txt"] = flatten_axtree_to_str(axt)

            if obs["dom_object"]:
                obs["dom_txt"] = flatten_dom_to_str(obs["dom_object"])
                obs["pruned_html"] = prune_html(obs["dom_txt"])
        return obs

    def get_action(self, obs: dict):  # type: ignore[override]
        if self._page is None:
            raise RuntimeError("Playwright Page missing; ensure use_raw_page_output=True")

        page = self._page
        step_dir = self._create_step_directory()
        
        self._display_recording_prompt()
        self._show_recording_indicators(page)
        
        # Capture human interactions
        captured_action, human_interactions = self._capture_interactions_with_js(page, step_dir)
        
        # Save and cleanup
        self._save_human_action(captured_action, step_dir)
        self._cleanup_indicators(page)
        
        self._step += 1
        return "noop()", {
            "extra_info": {
                "step": self._step - 1,
                "human_interactions": human_interactions,
            }
        }

    def _create_step_directory(self) -> Path:
        """Create directory for current step."""
        step_dir = self._root / str(self._step)
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir

    def _display_recording_prompt(self):
        """Display prompt messages to user."""
        print(f"[HumanTrace] Step {self._step}: Perform ONE action")
        print("[HumanTrace] ⚠️  WAIT FOR THE RED BORDER TO APPEAR BEFORE PERFORMING ANY ACTION ⚠️")
        print("[HumanTrace] The system will automatically save after detecting your action")

    def _show_recording_indicators(self, page: Page):
        """Show visual recording indicators on the page."""
        page.evaluate(get_recording_indicators_script())

    def _save_human_action(self, captured_action: dict, step_dir: Path):
        """Save the captured human action to JSON file."""
        try:
            human_action_path = step_dir / "human_action.json"
            if captured_action and isinstance(captured_action, dict):
                human_action_path.write_text(json.dumps(captured_action, indent=2))
                action_type = captured_action.get("type", "unknown")
            else:
                # Create empty action record for consistency
                empty_action = {
                    "type": "no_action",
                    "timestamp": time.time() * 1000,
                    "reason": "No meaningful human action captured in this step",
                }
                human_action_path.write_text(json.dumps(empty_action, indent=2))
                action_type = "no_action"

            print(f"[HumanTrace] Step {self._step} complete - Action: {action_type}")

        except Exception as e:
            print(f"[HumanTrace] Warning: Failed to save human action: {e}")

    def _cleanup_indicators(self, page: Page):
        """Remove recording indicators from the page."""
        page.evaluate("document.getElementById('__rec')?.remove(); document.getElementById('__rec_border')?.remove()")

    def _capture_interactions_with_js(self, page: Page, step_dir: Path) -> tuple[dict, str]:
        """Capture human interactions using JavaScript injection."""
        try:
            print("[HumanTrace] JavaScript interaction tracking enabled")
            initial_url, initial_title = page.url, page.title()

            # Inject interaction tracking
            self._inject_interaction_tracking(page)
            
            # Wait for user action
            self._wait_for_user_action(page)
            
            # Collect and process interaction data
            return self._collect_interaction_data(page, initial_url, initial_title)

        except Exception as e:
            print(f"[HumanTrace] Error: {e}")
            return {
                "type": "error",
                "timestamp": time.time() * 1000,
                "error": str(e),
            }, f"Error: {e}"

    def _inject_interaction_tracking(self, page: Page):
        """Inject JavaScript code for comprehensive interaction tracking."""
        tracking_script = get_interaction_tracking_script()
        page.evaluate(tracking_script)

    def _wait_for_user_action(self, page: Page):
        """Wait for user to perform an action."""
        start_time = time.time()
        while time.time() - start_time < 300:
            try:
                action_detected = page.evaluate("window.__acted || false")
                if action_detected:
                    print(f"[HumanTrace] Action detected! Exiting immediately...")
                    break
            except Exception as e:
                print(f"[HumanTrace] Debug: Error checking actions: {e}")
                pass
            time.sleep(0.1)

    def _collect_interaction_data(self, page: Page, initial_url: str, initial_title: str) -> tuple[dict, str]:
        """Collect and format interaction data."""
        try:
            action_detected = page.evaluate("window.__acted || false")
            interactions = page.evaluate("window.__interactions || []")
            
            action_data = {
                "type": "human_interactions" if action_detected else "no_action",
                "timestamp": time.time() * 1000,
                "detected": action_detected,
                "interactions": interactions,
                "interaction_count": len(interactions)
            }
            
            summary = self._create_interaction_summary(interactions)
            self._add_page_change_info(action_data, initial_url, initial_title, page)
            
            print(f"[HumanTrace] {summary}")
            return action_data, summary
            
        except Exception as e:
            return {
                "type": "error",
                "timestamp": time.time() * 1000,
                "detected": False,
                "error": str(e),
                "interactions": [],
                "interaction_count": 0
            }, f"Error collecting interactions: {e}"

    def _create_interaction_summary(self, interactions: list) -> str:
        """Create a summary string of captured interactions."""
        if interactions:
            interaction_types = {}
            for interaction in interactions:
                itype = interaction.get('type', 'unknown')
                interaction_types[itype] = interaction_types.get(itype, 0) + 1
            
            summary_parts = []
            for itype, count in interaction_types.items():
                summary_parts.append(f"{itype}:{count}")
            return f"Captured {len(interactions)} interactions: {', '.join(summary_parts)}"
        else:
            return "No interactions detected"

    def _add_page_change_info(self, action_data: dict, initial_url: str, initial_title: str, page: Page):
        """Add page change information to action data."""
        final_url, final_title = page.url, page.title()
        if initial_url != final_url or initial_title != final_title:
            action_data["page_changed"] = True
            action_data["url_change"] = {"from": initial_url, "to": final_url}
            action_data["title_change"] = {"from": initial_title, "to": final_title}

    def _format_js_interaction_summary(self, action_data, interaction_log):
        """Format JavaScript-captured interactions into readable summary."""
        lines = ["Human Interactions (JavaScript Tracking):"]

        if action_data["interactions"]:
            lines.append(f"Total Actions: {len(action_data['interactions'])}")
            lines.append("")

            # Group interactions by type
            by_type = {}
            for interaction in action_data["interactions"]:
                interaction_type = interaction["type"]
                if interaction_type not in by_type:
                    by_type[interaction_type] = []
                by_type[interaction_type].append(interaction)

            # Show summary by type
            for interaction_type, interactions in by_type.items():
                lines.append(f"{interaction_type.title()}: {len(interactions)} actions")

            lines.append("")
            lines.append("Detailed Actions:")

            # Add each interaction from the log
            for log_entry in interaction_log:
                lines.append(f"  {log_entry}")
        else:
            lines.append("No interactions detected - user may have just observed the page")

        # Add page state changes if URL changed
        if action_data.get("page_changed"):
            url_info = action_data.get("url")
            if url_info:
                lines.append("")
                lines.append("� Page Navigation:")
                lines.append(f"  From: {url_info['from']}")
                lines.append(f"  To: {url_info['to']}")

        return "\n".join(lines)


HUMAN_TRACE_AGENT = HumanTraceAgentArgs()


if __name__ == "__main__":
    from agentlab.agents.human_trace_recorder.agent import HUMAN_TRACE_AGENT
    from agentlab.experiments.study import Study

    agent_configs = [HUMAN_TRACE_AGENT]
    benchmark = bgym.DEFAULT_BENCHMARKS["workarena_l1"](n_repeats=1)  # type: bgym.Benchmark
    benchmark = benchmark.subset_from_glob("task_name", "*filter*")
    benchmark.env_args_list = benchmark.env_args_list[:1]
    for env_args in benchmark.env_args_list:
        print(env_args.task_name)
        env_args.max_steps = 15
        env_args.headless = False

    study = Study(agent_configs, benchmark)
    study.run(n_jobs=1, parallel_backend="sequential")
