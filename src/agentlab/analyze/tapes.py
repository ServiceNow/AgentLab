import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from tapeagents.core import Step, StepMetadata, Tape
from tapeagents.renderers.camera_ready_renderer import CameraReadyRenderer
from tapeagents.tape_browser import TapeBrowser

from agentlab.agents.tapeagent.agent import ExtendedMetadata

logger = logging.getLogger(__name__)
fmt = "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(funcName)s() - %(message)s"
logging.basicConfig(level=logging.INFO, force=True, format=fmt, handlers=[logging.StreamHandler()])


class WrapperStep(Step):
    content: dict


class TapesRender(CameraReadyRenderer):

    @property
    def style(self):
        style = "<style>.thought {{ background-color: #ffffba !important; }};</style>"
        return super().style + style

    def render_step(self, step: WrapperStep, index: int, **kwargs):
        step_dict = step.content.copy()
        step_dict.pop("metadata", None)
        kind = step_dict.pop("kind", "Step")
        # remove empty keys
        step_dict = {k: v for k, v in step_dict.items() if v is not None and v != ""}
        if len(step_dict) == 1:
            content = list(step_dict.values())[0]
        elif kind == "page_observation":
            content = step_dict["text"]
            if len(content) > 100:
                summary = content[:100]
                content = f"<details><summary>{summary}</summary>---<br>{content}</details>"
        elif kind == "python_code_action":
            content = step_dict["code"]
        elif kind == "code_execution_result":
            content = yaml.dump(step_dict["result"], sort_keys=False, indent=2)
        else:
            content = yaml.dump(step_dict, sort_keys=False, indent=2) if step_dict else ""

        if kind.endswith("thought"):
            class_ = "thought"
            kind = kind[:-8]
        elif kind.endswith("action"):
            class_ = "action"
            kind = kind[:-7]
        else:
            class_ = "observation"
        return (
            f"<div class='basic-renderer-box {class_}'>"
            f"<h4 class='step-header'>{kind}</h4>"
            f"<pre class='step-text'>{content}</pre>"
            f"</div>"
        )


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

    def get_steps(self, tape) -> list:
        return tape["steps"]

    def load_llm_calls(self):
        pass

    def get_context(self, tape: Tape) -> list:
        return []

    def get_tape_name(self, i: int, tape: Tape) -> str:
        return tape[0].content["content"][:32] + "..."

    def get_exp_label(self, filename: str, tapes: list[Tape]) -> str:
        acc, n_solved = 0, 0  # calculate_accuracy(tapes)
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
        for tape in tapes:
            if tape.metadata.result in ["", None, "None"]:
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
                if kind == "search_results_observation" and not len(step_dict["serp"]):
                    errors["search_empty"] += 1
                if kind == "page_observation" and step_dict["error"]:
                    errors["browser"] += 1
                elif kind == "llm_output_parsing_failure_action":
                    errors["parsing"] += 1
                elif kind == "action_execution_failure":
                    if last_action:
                        errors[f"{last_action}"] += 1
                    else:
                        errors["unknown_action_execution_failure"] += 1
                elif kind == "code_execution_result" and step_dict["result"]["exit_code"]:
                    errors["code_execution"] += 1
        timers, timer_counts = self.aggregate_timer_times(tapes)
        html = f"<h2>Solved {acc:.2f}%, {n_solved} out of {len(tapes)}</h2>"
        if "all" in filename:
            html += f"Prompt tokens: {prompt_tokens_num}<br>Output tokens: {output_tokens_num}<br>Cost: {total_cost:.2f} USD<h3>Visible</h3>"
        html += f"Prompt tokens: {visible_prompt_tokens_num}<br>Output tokens: {visible_output_tokens_num}<br>Cost: {visible_cost:.2f} USD"
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

    def load_tapes(self, exp_dir: str) -> list[dict]:
        tape_dicts = []
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
                    tape_dicts.append(tape)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
        logger.info(f"Loaded {len(tape_dicts)} tapes from {exp_dir}")
        return tape_dicts

    def save_annotation(self, step: int, annotation: str, tape_id: int):
        pass


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "~/agentlab_results/"
    tapes_browser = TapesBrowser(Path(results_dir).expanduser())
    tapes_browser.launch()
