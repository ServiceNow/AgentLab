import logging
import os
import re
import shutil
import string
from pathlib import Path
from typing import Any, Literal

import datasets
from pydantic import Field
from tapeagents.core import Action, Observation, StopStep, Thought
from tapeagents.environment import ContainerExecutor, StatefulTool, Tool
from tapeagents.steps import ImageObservation
from tapeagents.tools.browser import Browser
from tapeagents.tools.code_executor import CodeExecutor
from tapeagents.tools.media_reader import VideoReader
from tapeagents.tools.web_search import WebSearch

from agentlab.benchmarks.abstract_env import AbstractBenchmark, SerializableEnvArgs
from agentlab.benchmarks.multitool_gym import MultiToolGym

logger = logging.getLogger(__name__)


class GaiaGym(MultiToolGym):
    task: dict
    exp_dir: str

    def __init__(self, tools: list[Tool | StatefulTool], task: dict, exp_dir: str):
        super().__init__(tools=tools)
        self.task = task
        self.exp_dir = exp_dir
        os.makedirs(".cache", exist_ok=True)

    def reset(self, seed=None) -> tuple[list[Observation], dict]:
        super().reset()
        question = GaiaQuestion.from_task(self.task)
        steps = [question]
        if image_obs := with_image(question):
            steps.append(image_obs)
        return steps, {}

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        logger.info(f"Gym step called with action {type(action)}")
        observation, reward, terminated, truncated, env_info = super().step(action)
        logger.info(f"Gym observation: {observation.short_view()}")
        return observation, reward, terminated, truncated, env_info

    def calculate_reward(self, action: Action) -> float:
        if isinstance(action, GaiaAnswer):
            model_answer = action.answer
            ground_truth = self.task["Final answer"]
            reward = 1.0 if question_scorer(model_answer, ground_truth) else 0.0
        else:
            reward = 0.0

        if reward == 1.0:
            logger.info(f"Task {self.task['task_id']} solved.")
        else:
            logger.info(f"Task {self.task['task_id']} failed.")

        return reward


class GaiaGymArgs(SerializableEnvArgs):
    task: dict[str, Any]
    viewport_chars: int
    task_seed: int
    task_name: str

    def __init__(
        self, task_name: str, task: dict[str, Any], viewport_chars: int = 64000, task_seed: int = 0
    ):
        self.task_name = task_name
        self.task = task
        self.viewport_chars = viewport_chars
        self.task_seed = task_seed

    def make_env(self, exp_dir: str | Path, action_mapping=None) -> GaiaGym:
        exp_dir = str(exp_dir)
        self.init_code_sandbox(exp_dir)
        tools = [
            WebSearch(),
            VideoReader(exp_path=exp_dir),
            Browser(exp_path=exp_dir, viewport_chars=self.viewport_chars),
            CodeExecutor(exp_path=exp_dir, reuse_computer_container=True),
        ]
        env = GaiaGym(tools=tools, task=self.task, exp_dir=exp_dir)
        return env

    def init_code_sandbox(self, exp_dir: str) -> None:
        code_path = os.path.join(exp_dir, "code")
        os.makedirs(code_path, exist_ok=True)
        container_name = "gaia_code_sandbox"
        os.environ["COMPUTER_CONTAINER_NAME"] = container_name
        ContainerExecutor(
            work_dir=code_path,
            container_name=container_name,
            restart_if_exists=False,
            stop_container=False,
            no_deps=True,
        )


class GaiaBenchmark(AbstractBenchmark):
    name: str = "gaia"
    split: Literal["test", "validation"]
    level: Literal["1", "2", "3", "all"] = "all"
    env_args_list: list[GaiaGymArgs] = None

    def model_post_init(self, __context: Any) -> None:
        self.env_args_list = []
        dataset = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all")[self.split]
        for task in dataset:
            if self.level != "all" and task["Level"] != self.level:
                continue
            env_args = GaiaGymArgs(task_name="gaia." + task["task_id"], task=task)
            self.env_args_list.append(env_args)
        logger.info(
            f"Loaded {len(self.env_args_list)} tasks from {self.split} split of GAIA benchmark."
        )


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


def normalize_number_str(number_str: str) -> float:
    # we replace these common units and commas to allow
    # conversion to float
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        logger.info(f"String {number_str} cannot be normalized to number str.")
        return float("inf")


def split_string(
    s: str,
    char_list: list[str] = [",", ";"],
) -> list[str]:
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def question_scorer(
    model_answer: str,
    ground_truth: str,
) -> bool:
    def is_float(element: any) -> bool:
        try:
            float(element)
            return True
        except ValueError:
            return False

    # if gt is a number
    if is_float(ground_truth):
        logger.info(f"Evaluating {model_answer} as a number.")
        normalized_answer = normalize_number_str(model_answer)
        return normalized_answer == float(ground_truth)

    # if gt is a list
    elif any(char in ground_truth for char in [",", ";"]):
        logger.info(f"Evaluating {model_answer} as a comma separated list.")
        # question with the fish: normalization removes punct

        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)

        # check length is the same
        if len(gt_elems) != len(ma_elems):
            logger.warning("Answer lists have different lengths, returning False.")
            return False

        # compare each element as float or str
        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem)
                comparisons.append(normalized_ma_elem == float(gt_elem))
            else:
                # we do not remove punct since comparisons can include punct
                comparisons.append(
                    normalize_str(ma_elem, remove_punct=False)
                    == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    # if gt is a str
    else:
        logger.info(f"Evaluating {model_answer} as a string.")
        return normalize_str(model_answer) == normalize_str(ground_truth)


def normalize_str(input_str, remove_punct=True) -> str:
    """
    Normalize a string by:
    - Removing all white spaces
    - Optionally removing punctuation (if remove_punct is True)
    - Converting to lowercase

    Args:
        input_str: str, the string to normalize
        remove_punct: bool, whether to remove punctuation (default: True)

    Returns:
        str, the normalized string
    """
    # Remove all white spaces. Required e.g for seagull vs. sea gull
    no_spaces = re.sub(r"\s", "", input_str)

    # Remove punctuation, if specified.
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    else:
        return no_spaces.lower()
