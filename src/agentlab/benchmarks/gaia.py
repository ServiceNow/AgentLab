import logging
import os
import re
import shutil
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import datasets
import hydra
import podman
import tapeagents.config
from omegaconf import DictConfig
from pdf2image import convert_from_path
from pydantic import ConfigDict, Field
from tapeagents.core import Action, Observation, StopStep, Thought
from tapeagents.environment import ContainerExecutor, StatefulTool, Tool
from tapeagents.steps import ImageObservation
from tapeagents.tools.simple_browser import SimpleTextBrowser

from agentlab.benchmarks.abstract_env import AbstractBenchmark, AbstractEnvArgs
from agentlab.benchmarks.multitool_gym import MultiToolGym

logger = logging.getLogger(__name__)

CONTAINER_NAME = "gaia_code_shared"


class GaiaGym(MultiToolGym):
    task: dict
    exp_dir: str

    def __init__(self, tools: list[Tool | StatefulTool], task: dict, exp_dir: str):
        super().__init__(tools=tools)
        self.task = task
        self.exp_dir = exp_dir
        os.makedirs(".cache", exist_ok=True)

    def reset(self, seed=None) -> tuple[list[Observation], dict]:
        """
        Reset the state of all the tools and prepare initial observations from the task again
        """
        super().reset()
        return task_to_observations(self.task), {}

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


@dataclass
class GaiaGymArgs(AbstractEnvArgs):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    task: dict[str, Any]
    task_seed: int
    task_name: str
    env_config: DictConfig

    def __init__(
        self,
        task_name: str,
        task: dict[str, Any],
        env_config: DictConfig,
        task_seed: int = 0,
    ):
        self.task_name = task_name
        self.task = task
        self.task_seed = task_seed
        self.env_config = env_config

    def make_env(self, exp_dir: Path, action_mapping=None) -> GaiaGym:
        tapeagents.config.DB_DEFAULT_FILENAME = str(exp_dir.parent / "tapedata.sqlite")
        exp_dir_str = str(exp_dir)
        logger.info(f"Init gaia env with directory {exp_dir_str}")
        init_code_sandbox(exp_dir_str)
        for i in range(len(self.env_config.tools)):
            if hasattr(self.env_config.tools[i], "exp_path"):
                self.env_config.tools[i].exp_path = exp_dir_str
        tools = hydra.utils.instantiate(self.env_config.tools)
        env = GaiaGym(tools=tools, task=self.task, exp_dir=exp_dir_str)
        return env


def init_code_sandbox(exp_dir: str) -> None:
    # Use a common code directory for all tasks in the experiment, which is mounted in the container
    root_exp_dir = Path(exp_dir).parent
    code_path = os.path.join(root_exp_dir, "shared_code")
    os.makedirs(code_path, exist_ok=True)
    os.environ["COMPUTER_CONTAINER_NAME"] = CONTAINER_NAME

    # symlink task code to the shared code directory
    task_code_path = os.path.join(exp_dir, "code")
    if not os.path.exists(task_code_path):
        os.symlink(code_path, task_code_path)

    try:
        ContainerExecutor(container_name=CONTAINER_NAME, work_dir=code_path, no_deps=True)
    except Exception as e:
        logger.warning(f"Failed to initialize container executor: {e}")


def stop_old_sandbox():
    try:
        podman.from_env().containers.get(CONTAINER_NAME).stop()
    except Exception as e:
        logger.warning(f"Failed to stop old container {CONTAINER_NAME}: {e}")


class GaiaBenchmark(AbstractBenchmark):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "gaia"
    split: Literal["test", "validation"]
    level: Literal["1", "2", "3", "all"] = "all"
    env_args_list: list[GaiaGymArgs] = None  # type: ignore
    dataset: dict | None = None  # type: ignore
    env_config: DictConfig = None  # type: ignore

    @classmethod
    def from_config(cls, config: DictConfig, dataset: dict | None = None) -> Self:
        return cls(
            split=config.split,
            level=config.level,
            env_config=config.environment,
            dataset=dataset,
        )

    def model_post_init(self, __context: Any) -> None:
        self.env_args_list = []
        number = 0
        if self.dataset is None:
            self.dataset = datasets.load_dataset(
                path="gaia-benchmark/GAIA",
                name="2023_all",
                trust_remote_code=True,
            )  # type: ignore
        for task in self.dataset[self.split]:  # type: ignore
            if self.level != "all" and task["Level"] != self.level:
                continue
            number += 1
            task["number"] = number
            name = f"gaia.{task['task_id']}"
            env_args = GaiaGymArgs(task_name=name, task=task, env_config=self.env_config)
            self.env_args_list.append(env_args)
        logger.info(f"Loaded {len(self.env_args_list)} tasks from {self.split} split")


class ExtractedFacts(Thought):
    """
    Thought that contains the list of facts extracted from the document
    """

    kind: Literal["extracted_facts_thought"] = "extracted_facts_thought"  # type: ignore
    extracted_facts: list[str] | dict[str, Any] | str = Field(
        description="facts extracted from the observation"
    )


class GaiaQuestion(Observation):
    kind: Literal["question"] = "question"  # type: ignore
    content: str
    filename: str | None = None

    @classmethod
    def from_task(cls, question: dict, files_dir: str = "/tmp/gaia_files"):
        os.makedirs(files_dir, exist_ok=True)
        question_prompt = question["Question"]
        filename = None
        if question["file_path"]:
            basename = os.path.basename(question["file_path"])
            tmp_fname = os.path.join(files_dir, basename)
            shutil.copyfile(question["file_path"], tmp_fname)
            assert os.path.exists(tmp_fname)
            filename = tmp_fname
        return cls(content=question_prompt, filename=filename)


def task_to_observations(task: dict, max_doc_length: int = 8000) -> list[Observation]:
    browser = SimpleTextBrowser()
    question = GaiaQuestion.from_task(task)
    if not question.filename:
        return [question]

    filename: str | None = question.filename
    question.filename = None
    steps: list[Observation] = []
    name, ext = filename.rsplit(".", maxsplit=1)
    ext = ext.lower()
    if ext == "zip":
        folder_name = name
        os.makedirs(folder_name, exist_ok=True)
        shutil.unpack_archive(filename, folder_name)
        document_text = "\n\nArchive contains the following files:\n"
        for i, file in enumerate(os.listdir(folder_name)):
            file_path = os.path.join(folder_name, file)
            content = browser.get_whole_document(file_path)
            file_text = f"{i+1}. {file}. Content:\n{content}\n\n"
            if len(file_text) > max_doc_length:
                file_text = ""
            file_text += f"{i+1}. Path to the '{file}': {file_path}"
            document_text += file_text
    elif ext in ("png", "jpg", "jpeg"):
        steps.append(ImageObservation(image_path=filename, image_caption="Attached image"))
        document_text = ""
    else:
        attach_doc_text = True
        if ext == "pdf":
            images, total_pages = pdf_to_images(filename)
            if total_pages <= 3:
                attach_doc_text = False
            for i, img_path in enumerate(images):
                steps.append(ImageObservation(image_path=img_path, image_caption=f"PDF page {i+1}"))
        if attach_doc_text:
            try:
                content = browser.get_whole_document(filename)
            except Exception as e:
                logger.exception(f"Failed to read document: {e}")
                content = ""
            document_text = f"\n\nAttached {ext.upper()} file content:\n{content}\n"
            if not len(content) or len(document_text) > max_doc_length:
                document_text = ""
        else:
            document_text = "\nDocument pages attached as images below"
        question.filename = filename
    question.content += document_text
    return [question] + steps


def pdf_to_images(filename: str, n_pages: int = 3):
    images = []
    for i, image in enumerate(convert_from_path(filename)):
        page_index = i + 1
        page_fname = filename[:-4] + f"_{page_index}.png"
        if os.path.exists(page_fname):
            images.append(page_fname)
            continue
        image.save(page_fname)
        images.append(page_fname)
    return images[:n_pages], len(images)


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

    kind: Literal["gaia_answer_action"] = "gaia_answer_action"  # type: ignore
    success: bool = Field(description="True if the task was successful, False otherwise")
    overview: str = Field(
        description="List of steps performed to answer the question. If the task was not successful, includes the reason for failure"
    )
    answer_unit: str = Field(
        description="Unit of measurement for the answer, if applicable; otherwise an empty string"
    )
    answer: Any = Field(description="Short final answer")
    long_answer: str = Field(description="Detailed final answer not restricted by format rules")


def step_error(step_dict: dict, last_action: str | None) -> str:
    kind = step_dict.get("kind", "unknown")
    error = ""
    if kind == "search_results_observation" and not len(step_dict.get("serp", [])):
        error = "search_empty"
    elif kind == "page_observation" and step_dict.get("error"):
        error = "browser"
    elif kind == "llm_output_parsing_failure_action":
        error = "parsing"
    elif kind == "action_execution_failure":
        error = last_action if last_action else "action_failure"
    elif kind == "code_execution_result" and step_dict.get("result", {}).get("exit_code"):
        error = "code"
    return error


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
    def is_float(element: Any) -> bool:
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
