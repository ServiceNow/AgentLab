import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from tapeagents.core import Step, StepMetadata
from tapeagents.observe import retrieve_all_llm_calls
from tapeagents.renderers.camera_ready_renderer import CameraReadyRenderer
from tapeagents.tape_browser import TapeBrowser

from agentlab.agents.tapeagent.agent import ExtendedMetadata, Tape
from agentlab.benchmarks.gaia import step_error

logger = logging.getLogger(__name__)
fmt = "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(funcName)s() - %(message)s"
logging.basicConfig(level=logging.INFO, force=True, format=fmt, handlers=[logging.StreamHandler()])


class WrapperStep(Step):
    content: dict


def pretty_yaml(data: dict | None) -> str:
    return yaml.dump(data, sort_keys=False, indent=2) if data else ""


class TapesRender(CameraReadyRenderer):

    @property
    def style(self):
        style = "<style>.thought {{ background-color: #ffffba !important; }};</style>"
        return super().style + style

    def render_step(self, step: WrapperStep, index: int, **kwargs):
        step_dict = step.content.copy()
        step_dict.pop("metadata", None)
        kind = step_dict.pop("kind", "Step")
        if kind == "set_next_node":
            return ""
        # remove empty keys
        step_dict = {k: v for k, v in step_dict.items() if v is not None and v != ""}
        if len(step_dict) == 1:
            content = list(step_dict.values())[0]
        elif kind == "page_observation":
            content = step_dict.get("text", pretty_yaml(step_dict))
            if len(content) > 100:
                summary = content[:100]
                content = f"<details><summary>{summary}</summary>---<br>{content}</details>"
        elif kind == "python_code_action":
            content = step_dict.get("code", pretty_yaml(step_dict))
        elif kind == "code_execution_result":
            content = pretty_yaml(step_dict.get("result"))
        elif len(step_dict) == 1 and "content" in step_dict:
            content = step_dict["content"]
        elif len(step_dict) == 1 and "reasoning" in step_dict:
            content = step_dict["reasoning"]
        else:
            content = pretty_yaml(step_dict)

        if step_dict.get("error") or step_dict.get("result", {}).get("exit_code"):
            class_ = "error"
        elif kind.endswith("thought"):
            class_ = "thought"
            kind = kind[:-8]
        elif kind.endswith("action"):
            class_ = "action"
            kind = kind[:-7]
        else:
            class_ = "observation"
        return f"<div class='basic-renderer-box {class_}'><h4 class='step-header'>{kind}</h4><pre class='step-text'>{content}</pre></div>"


class TapesBrowser(TapeBrowser):
    def __init__(self, tapes_folder):
        super().__init__(Tape, tapes_folder, TapesRender(), ".json")

    def get_tape_files(self) -> list[str]:
        logger.info(f"Searching for tapes in {self.tapes_folder}")
        fpath = Path(self.tapes_folder)
        exps = [
            str(exp_dir.relative_to(fpath))
            for exp_dir in fpath.iterdir()
            if exp_dir.is_dir() and len(list(exp_dir.rglob("tape.json"))) > 0
        ]
        assert exps, f"No experiments found in {self.tapes_folder}"
        logger.info(f"Found {len(exps)} experiments in {self.tapes_folder}")
        return sorted(exps)

    def get_steps(self, tape: dict) -> list:
        return tape["steps"]

    def load_llm_calls(self):
        sqlite_path = self.exp_path / "tapedata.sqlite"
        if sqlite_path.exists():
            try:
                self.llm_calls = {
                    call.prompt.id: call for call in retrieve_all_llm_calls(str(sqlite_path))
                }
                logger.info(f"Loaded {len(self.llm_calls)} LLM calls from {sqlite_path}")
            except Exception as e:
                logger.warning(f"Failed to load LLM calls from {sqlite_path}: {e}")
        else:
            logger.warning(f"{sqlite_path} not found")

    def get_context(self, tape: Tape) -> list:
        return []

    def get_tape_name(self, i: int, tape: Tape) -> str:
        errors = [
            bool(s.content.get("error", False) or s.content.get("result", {}).get("exit_code"))
            for s in tape.steps
        ]
        mark = "✅ " if tape.metadata.reward > 0 else ""
        if any(errors):
            mark = "⚠ "
        if tape.metadata.task.get("file_name"):
            mark += "📁 "
        number = tape.metadata.task.get("number", "")
        n = f"{tape.metadata.task.get('Level', '')}.{number} " if number else ""
        name = tape.steps[0].content["content"][:32] + "..."
        return f"{n}({len(tape.steps)}){mark}{name}"

    def get_exp_label(self, filename: str, tapes: list[Tape]) -> str:
        acc, n_solved = self.calculate_accuracy(tapes)
        errors = defaultdict(int)
        prompt_tokens_num = 0
        output_tokens_num = 0
        total_cost = 0.0
        visible_prompt_tokens_num = 0
        visible_output_tokens_num = 0
        visible_cost = 0.0
        no_result = 0
        actions = defaultdict(int)
        for llm_call in self.llm_calls.values():
            prompt_tokens_num += llm_call.prompt_length_tokens
            output_tokens_num += llm_call.output_length_tokens
            total_cost += llm_call.cost
        avg_steps = np.mean([len(tape) for tape in tapes])
        std_steps = np.std([len(tape) for tape in tapes])
        for tape in tapes:
            if tape.metadata.truncated:
                no_result += 1
            if tape.metadata.error:
                errors["fatal"] += 1
            last_action = None
            counted = set([])
            for step in tape:
                step_dict = step.content.copy()
                kind = step_dict.get("kind", "unknown")
                llm_call = self.llm_calls.get(step.metadata.prompt_id)
                if llm_call and step.metadata.prompt_id not in counted:
                    counted.add(step.metadata.prompt_id)
                    visible_prompt_tokens_num += llm_call.prompt_length_tokens
                    visible_output_tokens_num += llm_call.output_length_tokens
                    visible_cost += llm_call.cost
                if kind.endswith("action"):
                    actions[kind] += 1
                    last_action = kind
                if error := self.get_step_error(step_dict, last_action):
                    errors[error] += 1
        timers, timer_counts = self.aggregate_timer_times(tapes)
        html = f"<h2>Solved {acc:.2f}%, {n_solved} out of {len(tapes)}</h2>"
        if "all" in filename:
            html += f"Prompt tokens: {prompt_tokens_num}<br>Output tokens: {output_tokens_num}<br>Cost: {total_cost:.2f} USD<h3>Visible</h3>"
        html += f"Prompt tokens: {visible_prompt_tokens_num}<br>Output tokens: {visible_output_tokens_num}<br>Cost: {visible_cost:.2f} USD"
        html += f"<h2>Steps per tape: {avg_steps:.1f} ± {std_steps:.1f}</h2>"
        if errors:
            errors_str = "<br>".join(f"{k}: {v}" for k, v in errors.items())
            html += f"<h2>No result: {no_result}</h2>"
            html += f"<h2>Errors: {sum(errors.values())}</h2>{errors_str}"
        if actions:
            actions_str = "<br>".join(f"{k}: {v}" for k, v in actions.items())
            html += f"<h2>Actions: {sum(actions.values())}</h2>{actions_str}"
        if timers:
            timers_str = "<br>".join(
                f"{'execute ' if k.endswith('action') else ''}{k}: {v:.1f} sec, avg. {v/timer_counts[k]:.1f} sec"
                for k, v in timers.items()
            )
            html += f"<h2>Timings</h2>{timers_str}"
        return html

    def get_step_error(self, step_dict: dict, last_action: str | None) -> str:
        return step_error(step_dict, last_action)

    def calculate_accuracy(self, tapes: list[Tape]) -> tuple[float, int]:
        solved = [tape.metadata.reward for tape in tapes]
        accuracy = 100 * (sum(solved) / len(solved) if solved else 0.0)
        return accuracy, int(sum(solved))

    def aggregate_timer_times(self, tapes: list[Tape]):
        timer_sums = defaultdict(float)
        timer_counts = defaultdict(int)
        for tape in tapes:
            timers = tape.metadata.other.get("timers", {})
            for timer_name, exec_time in timers.items():
                timer_sums[timer_name] += exec_time
                timer_counts[timer_name] += 1
            for step in tape.steps:
                action_kind = step.metadata.other.get("action_kind")
                action_execution_time = step.metadata.other.get("action_execution_time")
                if action_kind and action_execution_time:
                    timer_sums[action_kind] += action_execution_time
                    timer_counts[action_kind] += 1
        return dict(timer_sums), dict(timer_counts)

    def load_tapes(self, exp_dir: str) -> list[Tape]:
        tapes: list[Tape] = []
        fpath = Path(self.tapes_folder) / exp_dir
        for json_file in fpath.rglob("tape.json"):
            if json_file.stat().st_size == 0:
                logger.warning(f"Empty tape file: {json_file}")
                continue
            try:
                with open(json_file) as f:
                    tape_dict = json.load(f)
                    tape = Tape(steps=[], metadata=ExtendedMetadata(**tape_dict["metadata"]))
                    tape.steps = [
                        WrapperStep(content=s, metadata=StepMetadata(**s["metadata"]))
                        for s in tape_dict["steps"]
                    ]
                    tapes.append(tape)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
        logger.info(f"Loaded {len(tapes)} tapes from {fpath}")
        self.exp_path = fpath
        return sorted(
            tapes,
            key=lambda x: f"{x.metadata.task.get('Level', '')}{x.metadata.task.get('number', 0):03d}",
        )

    def save_annotation(self, step: int, annotation: str, tape_id: int):
        pass


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "~/agentlab_results/"
    tapes_browser = TapesBrowser(Path(results_dir).expanduser())
    tapes_browser.launch()
