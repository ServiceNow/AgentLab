from agentlab.llm.llm_utils import (
    Discussion,
    extract_code_blocks,
    HumanMessage,
    ParseError,
    parse_html_tags_raise,
)
from browsergym.core.action.highlevel import HighLevelActionSet
from dataclasses import dataclass
from PIL import Image
from typing import Callable, Optional, Union
from .base import VLPrompt, VLPromptArgs, VLPromptPart
from ..utils import image_to_image_url
import numpy as np


class IntroductionPromptPart(VLPromptPart):
    def __init__(self):
        self.text = """\
You are an agent working to address a web-based task through step-by-step interactions with the browser. \
To achieve the goal of the task, at each step, you need to submit an action based on the current state of the browser. \
This action will be executed to update the state of the browser, and you will proceed to the next step.
"""

    def get_message_content(self) -> list[dict]:
        return [{"type": "text", "text": self.text}]


class GoalPromptPart(VLPromptPart):
    def __init__(self, goal_object: list[dict]):
        text = """
# The goal of the task
"""
        for item in goal_object:
            if item["type"] == "text":
                text += f"""
{item['text']}
"""
        self.text = text

    def get_message_content(self) -> list[dict]:
        return [{"type": "text", "text": self.text}]


class ScreenshotPromptPart(VLPromptPart):
    def __init__(self, screenshot: Union[Image.Image, np.ndarray]):
        self.text = """
# The screenshot of the current web page
"""
        self.image_url = image_to_image_url(screenshot)

    def get_message_content(self) -> list[dict]:
        return [
            {"type": "text", "text": self.text},
            {"type": "image_url", "image_url": {"url": self.image_url}},
        ]


class TabsPromptPart(VLPromptPart):
    def __init__(
        self, open_pages_titles: list[str], open_pages_urls: list[str], active_page_index: int
    ):
        text = """
# The titles and URLs of the open tabs
"""
        for index, (title, url) in enumerate(zip(open_pages_titles, open_pages_urls)):
            text += f"""
## Tab {index}{' (active tab)' if index == active_page_index else ''}

### Title

{title}

### URL

{url}
"""
        self.text = text

    def get_message_content(self) -> list[dict]:
        return [{"type": "text", "text": self.text}]


class HistoryPromptPart(VLPromptPart):
    def __init__(self, thoughts: list[str], actions: list[str]):
        text = """
# The thoughts and actions of the previous steps
"""
        for index, (thought, action) in enumerate(zip(thoughts, actions)):
            text += f"""
## Step {index}

### Though

{thought}

### Action

{action}
"""
        self.text = text

    def get_message_content(self) -> list[dict]:
        return [{"type": "text", "text": self.text}]


class ErrorPromptPart(VLPromptPart):
    def __init__(
        self,
        last_action_error: str,
        logs_separator: str = "Call log:",
        logs_limit: int = 5,
    ):
        text = """
# The error caused by the last action
"""
        if logs_separator in last_action_error:
            error, logs = last_action_error.split(logs_separator)
            logs = logs.split("\n")[:logs_limit]
            text += f"""
{error}

{logs_separator}
"""
            for log in logs:
                text += f"""
{log}
"""
        else:
            text += f"""
{last_action_error}
"""
        self.text = text

    def get_message_content(self) -> list[dict]:
        return [{"type": "text", "text": self.text}]


class PreliminaryAnswerPromptPart(VLPromptPart):
    def __init__(
        self, action_set_description: str, use_abstract_example: bool, use_concrete_example: bool
    ):
        text = f"""
# The action space

Here are all the actions you can take to interact with the browser.

{action_set_description}

# The answer requirements

Think about the action, and describe the location where the action is to be taken. \
Your answer should contain one thought and one location.
"""
        if use_abstract_example:
            text += """
# An abstract example of the answer

<thought>
The thought about the action.
</thought>
<location>
The description of the location where the action is to be taken.
</location>
"""
        if use_concrete_example:
            text += """
# A concrete example of the answer

<thought>
The goal is to click on the numbers in ascending order. \
The smallest number visible on the screen is '1'. \
I will use the 'mouse_click' action to directly click on the number '1'.
</thought>
<location>
The number '1' in the top-left quadrant of the white area.
</location>
"""
        self.text = text

    def get_message_content(self) -> list[dict]:
        return [{"type": "text", "text": self.text}]


class FinalAnswerPromptPart(VLPromptPart):
    def __init__(
        self,
        action_set_description: str,
        preliminary_answer: dict,
        use_abstract_example: bool,
        use_concrete_example: bool,
    ):
        text = f"""
# The action space

Here are all the actions you can take to interact with the browser.

{action_set_description}

# The thought about the action

{preliminary_answer['thought']}

# The coordinates where the action is to be taken

{preliminary_answer['coordinates']}

# The answer requirements

Formulate the action to be taken. \
Your answer should contain only one action.
"""
        if use_abstract_example:
            text += """
# An abstract example of the answer

<action>
The action to be taken.
</action>
"""
        if use_concrete_example:
            text += """
# A concrete example of the answer

<action>
mouse_click(50, 50)
</action>
"""
        self.text = text

    def get_message_content(self) -> list[dict]:
        return [{"type": "text", "text": self.text}]


@dataclass
class MainUIPrompt(VLPrompt):
    introduction_prompt_part: IntroductionPromptPart
    goal_prompt_part: GoalPromptPart
    screenshot_prompt_part: ScreenshotPromptPart
    tabs_prompt_part: Optional[TabsPromptPart]
    history_prompt_part: Optional[HistoryPromptPart]
    error_prompt_part: Optional[ErrorPromptPart]
    answer_prompt_part: Union[PreliminaryAnswerPromptPart, FinalAnswerPromptPart]
    action_validator: Callable

    def get_messages(self) -> Discussion:
        message_content = self.introduction_prompt_part.get_message_content()
        message_content.extend(self.goal_prompt_part.get_message_content())
        message_content.extend(self.screenshot_prompt_part.get_message_content())
        if self.tabs_prompt_part is not None:
            message_content.extend(self.tabs_prompt_part.get_message_content())
        if self.history_prompt_part is not None:
            message_content.extend(self.history_prompt_part.get_message_content())
        if self.error_prompt_part is not None:
            message_content.extend(self.error_prompt_part.get_message_content())
        message_content.extend(self.answer_prompt_part.get_message_content())
        messages = Discussion([HumanMessage(message_content)])
        messages.merge()
        return messages

    def parse_answer(self, answer_content: list[dict]) -> dict:
        answer_text = answer_content[0]["text"]
        answer_dict = {}
        if isinstance(self.answer_prompt_part, PreliminaryAnswerPromptPart):
            try:
                answer_dict.update(parse_html_tags_raise(answer_text, keys=["thought", "location"]))
            except ParseError as error:
                raise error
        elif isinstance(self.answer_prompt_part, FinalAnswerPromptPart):
            try:
                answer_dict.update(parse_html_tags_raise(answer_text, keys=["action"]))
            except ParseError as error:
                code_blocks = extract_code_blocks(answer_text)
                if len(code_blocks) == 0:
                    raise error
                else:
                    answer_dict["action"] = "\n".join([block for _, block in code_blocks])
            if answer_dict["action"] == "None":
                answer_dict["action"] = None
            else:
                try:
                    self.action_validator(answer_dict["action"])
                except:
                    raise ParseError(f"Invalid action: {answer_dict['action']}")
        else:
            raise ValueError(
                f"Unsupported answer prompt part type: {type(self.answer_prompt_part)}"
            )
        return answer_dict


@dataclass
class AuxiliaryUIPrompt(VLPrompt):
    screenshot: Union[Image.Image, np.ndarray]
    location: str

    def get_messages(self) -> Discussion:
        image_url = image_to_image_url(self.screenshot)
        text = f"""\
Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on this screen based on a description. \
Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible. \
If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose. \
Your answer should be a single string (x, y) corresponding to the point of interest.

Description: {self.location}
"""
        message_content = [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": text},
        ]
        messages = Discussion([HumanMessage(message_content)])
        return messages

    def parse_answer(self, answer_content: list[dict]) -> dict:
        answer_text = answer_content[0]["text"]
        answer_dict = {"coordinates": answer_text}
        return answer_dict


@dataclass
class UIPromptArgs(VLPromptArgs):
    use_tabs: bool
    use_history: bool
    use_error: bool
    use_abstract_example: bool
    use_concrete_example: bool

    def make_prompt(
        self,
        obs: dict,
        thoughts: list[str],
        actions: list[str],
        action_set: HighLevelActionSet,
        preliminary_answer: Optional[dict] = None,
    ) -> VLPrompt:
        introduction_prompt_part = IntroductionPromptPart()
        goal_prompt_part = GoalPromptPart(obs["goal_object"])
        screenshot_prompt_part = ScreenshotPromptPart(obs["screenshot"])
        if self.use_tabs:
            tabs_prompt_part = TabsPromptPart(
                open_pages_titles=obs["open_pages_titles"],
                open_pages_urls=obs["open_pages_urls"],
                active_page_index=obs["active_page_index"],
            )
        else:
            tabs_prompt_part = None
        if self.use_history and len(thoughts) == len(actions) > 0:
            history_prompt_part = HistoryPromptPart(thoughts=thoughts, actions=actions)
        else:
            history_prompt_part = None
        if self.use_error and obs["last_action_error"]:
            error_prompt_part = ErrorPromptPart(obs["last_action_error"])
        else:
            error_prompt_part = None
        if preliminary_answer is None:
            preliminary_answer_prompt_part = PreliminaryAnswerPromptPart(
                action_set_description=action_set.describe(
                    with_long_description=True, with_examples=False
                ),
                use_abstract_example=self.use_abstract_example,
                use_concrete_example=self.use_concrete_example,
            )
            self.preliminary_main_ui_prompt = MainUIPrompt(
                introduction_prompt_part=introduction_prompt_part,
                goal_prompt_part=goal_prompt_part,
                screenshot_prompt_part=screenshot_prompt_part,
                tabs_prompt_part=tabs_prompt_part,
                history_prompt_part=history_prompt_part,
                error_prompt_part=error_prompt_part,
                answer_prompt_part=preliminary_answer_prompt_part,
                action_validator=action_set.to_python_code,
            )
            return self.preliminary_main_ui_prompt
        else:
            if "coordinates" in preliminary_answer:
                final_answer_prompt_part = FinalAnswerPromptPart(
                    action_set_description=action_set.describe(
                        with_long_description=True, with_examples=False
                    ),
                    preliminary_answer=preliminary_answer,
                    use_abstract_example=self.use_abstract_example,
                    use_concrete_example=self.use_concrete_example,
                )
                self.final_main_ui_prompt = MainUIPrompt(
                    introduction_prompt_part=introduction_prompt_part,
                    goal_prompt_part=goal_prompt_part,
                    screenshot_prompt_part=screenshot_prompt_part,
                    tabs_prompt_part=tabs_prompt_part,
                    history_prompt_part=history_prompt_part,
                    error_prompt_part=error_prompt_part,
                    answer_prompt_part=final_answer_prompt_part,
                    action_validator=action_set.to_python_code,
                )
                return self.final_main_ui_prompt
            else:
                self.auxiliary_ui_prompt = AuxiliaryUIPrompt(
                    screenshot=obs["screenshot"], location=preliminary_answer["location"]
                )
                return self.auxiliary_ui_prompt
