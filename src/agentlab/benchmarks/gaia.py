import logging
import os
import shutil
from pathlib import Path
from typing import Any, Literal

import datasets
from pydantic import Field
from tapeagents.core import Observation, StopStep, Thought
from tapeagents.environment import ContainerExecutor, StatefulTool, Tool
from tapeagents.steps import ImageObservation
from tapeagents.tools.browser import Browser
from tapeagents.tools.code_executor import CodeExecutor
from tapeagents.tools.media_reader import VideoReader
from tapeagents.tools.web_search import WebSearch

from agentlab.benchmarks.abstract_env import AbstractBenchmark, AbstractEnvArgs
from agentlab.benchmarks.multitool_gym import MultiToolGym

logger = logging.getLogger(__name__)


class GaiaGym(MultiToolGym):
    task: dict
    exp_dir: str

    def __init__(self, tools: list[Tool | StatefulTool], task: dict, exp_dir: str):
        super().__init__(tools=tools)
        self.task = task
        self.exp_dir = exp_dir

    def reset(self, seed=None) -> tuple[list[Observation], dict]:
        super().reset()
        question = GaiaQuestion.from_task(self.task)
        steps = [question]
        if image_obs := with_image(question):
            steps.append(image_obs)
        return steps, {}

    def step(self, action: str) -> tuple[Observation, float, bool, bool, dict]:
        logger.info(f"step called with action: {type(action)}")
        super().step(action)


class GaiaGymArgs(AbstractEnvArgs):
    task: dict[str, Any]
    viewport_chars: int = 64000

    def make_env(self, exp_dir: str | Path, action_mapping=None) -> GaiaGym:
        exp_dir = str(exp_dir)
        self.init_code_sandbox(exp_dir)
        tools = [
            WebSearch(),
            VideoReader(exp_path=exp_dir),
            Browser(exp_path=exp_dir, viewport_chars=self.viewport_chars),
            CodeExecutor(exp_path=exp_dir),
        ]
        env = GaiaGym(tools=tools, task=self.task, exp_dir=exp_dir)
        return env

    def init_code_sandbox(self, exp_dir: str) -> None:
        code_path = os.path.join(exp_dir, "code")
        os.makedirs(code_path, exist_ok=True)
        ContainerExecutor(
            work_dir=code_path,
            container_name="gaia_code_sandbox",
            restart_if_exists=False,
            stop_container=False,
            no_deps=True,
        )


class GaiaBenchmark(AbstractBenchmark):
    name: str = "gaia"
    split: Literal["test", "validation"]
    env_args_list: list[GaiaGymArgs] = None

    def model_post_init(self, __context: Any) -> None:
        self.env_args_list = []
        dataset = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all")[self.split]
        for task in dataset:
            env_args = GaiaGymArgs(task=task)
            self.env_args_list.append(env_args)


class ExtractedFacts(Thought):
    """
    Thought that contains the list of facts extracted from the document
    """

    kind: Literal["extracted_facts_thought"] = "extracted_facts_thought"
    extracted_facts: list[str] | dict[str, Any] | str = Field(
        description="facts extracted from the observation"
    )


class GaiaQuestion(Observation):
    kind: Literal["question"] = "question"
    content: str
    filename: str | None = None

    @classmethod
    def from_task(cls, question: dict):
        question_prompt = question["Question"]
        filename = None
        if question["file_path"]:
            basename = os.path.basename(question["file_path"])
            tmp_fname = f"/tmp/{basename}"
            shutil.copyfile(question["file_path"], tmp_fname)
            assert os.path.exists(tmp_fname)
            filename = tmp_fname
        return cls(content=question_prompt, filename=filename)


def with_image(question: GaiaQuestion) -> ImageObservation | None:
    if question.filename and question.filename.endswith((".png", ".jpg", ".jpeg")):
        return ImageObservation(
            image_path=question.filename,
            image_caption="Attached image",
        )


class GaiaAnswer(StopStep):
    """
    Action that indicates the agent has finished the plan and contains the answer or description of failure.
    The answer should use already determined facts without additional conversion!
    Your final answer should be a number OR as few words as possible OR a comma-separated list of numbers and/or strings.
    ADDITIONALLY, your final answer MUST follow any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
    If asked for a number, express it numerically, don't use commas, do not add anything after the number, don't include units such as $ or percent signs unless specified otherwise in the question.
    If asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
    If asked for a comma-separated list, apply the above rules depending on whether the elements are numbers or strings.
    If unable to determine the final answer, output an empty string.
    """

    kind: Literal["gaia_answer_action"] = "gaia_answer_action"
    success: bool = Field(description="True if the task was successful, False otherwise")
    overview: str = Field(
        description="List of steps performed to answer the question. If the task was not successful, includes the reason for failure"
    )
    answer_unit: str = Field(
        description="Unit of measurement for the answer, if applicable; otherwise an empty string"
    )
    answer: Any = Field(description="Short final answer")
    long_answer: str = Field(description="Detailed final answer not restricted by format rules")
