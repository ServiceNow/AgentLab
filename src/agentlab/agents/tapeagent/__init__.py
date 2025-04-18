import json
from dataclasses import asdict, is_dataclass

import numpy as np
from tapeagents.core import Step, StepMetadata
from tapeagents.dialog_tape import AssistantStep, AssistantThought
from tapeagents.io import save_json_tape, save_tape_images

from agentlab.agents.tapeagent.agent import DictObservation, Tape, TapeAgent

__all__ = ["as_tape", "save_tape", "TapeAgent", "Tape"]


def as_tape(steps_info: list) -> Tape:
    """
    Create a Tape object from the steps info.

    Args:
        steps_info: list of StepInfo objects.

    Returns:
        Tape: a Tape object containing the steps and metadata.
    """

    class JsonEncoder(json.JSONEncoder):
        def default(self, obj):
            if is_dataclass(obj):
                return asdict(obj)  # type: ignore
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    steps: list[Step] = []
    for step_info in steps_info:
        if step_info.obs is not None:
            json_obs = json.dumps(step_info.obs, cls=JsonEncoder)
            steps.append(DictObservation(content=json_obs))
        if thought := step_info.agent_info.get("think"):
            steps.append(AssistantThought(content=thought))
        if step_info.action is not None:
            step_metadata = StepMetadata(
                other=dict(
                    reward=step_info.reward,
                    raw_reward=step_info.raw_reward,
                    terminated=step_info.terminated,
                    truncated=step_info.truncated,
                    agent_info=step_info.agent_info,
                    stats=step_info.stats,
                )
            )
            steps.append(AssistantStep(content=step_info.action, metadata=step_metadata))
    return Tape(steps=steps)


def save_tape(exp_dir: str, episode_info: list, task: dict, tape: Tape):
    tape.metadata.reward = sum([step.reward for step in episode_info])
    tape.metadata.truncated = episode_info[-1].truncated
    tape.metadata.terminated = episode_info[-1].terminated
    tape.metadata.task = task
    save_json_tape(tape, exp_dir, "tape.json")
    save_tape_images(tape, f"{exp_dir}/tape_attachments")
