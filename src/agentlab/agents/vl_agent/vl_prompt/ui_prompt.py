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
from typing import Optional, Union
from .base import VLPrompt, VLPromptArgs, VLPromptPart
from ..utils import image_to_image_url
import numpy as np


class IntroductionPromptPart(VLPromptPart):
    def __init__(self):
        self.text = """\
You are an agent working to address a web-based task through step-by-step interactions with the browser. \
To achieve the goal of the task, at each step, you need to submit an action according to the current state of the browser. \
This action will be executed to update the state of the browser, and you will proceed to the next step.
"""

    def get_message_content(self) -> list[dict]:
        return [{"type": "text", "text": self.text}]


class GoalPromptPart(VLPromptPart):
    def __init__(self, goal_object: list[dict]):
        text = """\
# The goal of the task
"""
        for item in goal_object:
            if item["type"] == "text":
                text += f"""\
{item['text']}
"""
        self.text = text

    def get_message_content(self) -> list[dict]:
        return [{"type": "text", "text": self.text}]


class ScreenshotPromptPart(VLPromptPart):
    def __init__(self, screenshot: Union[Image.Image, np.ndarray]):
        self.text = """\
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
        text = """\
# The open tabs of the browser
"""
        for index, (title, url) in enumerate(zip(open_pages_titles, open_pages_urls)):
            text += f"""\
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
        text = """\
# The thoughts and actions of the previous steps
"""
        for index, (thought, action) in enumerate(zip(thoughts, actions)):
            text += f"""\
## Step {index}
### Thought
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
        text = """\
# The error caused by the last action
"""
        if logs_separator in last_action_error:
            error, logs = last_action_error.split(logs_separator)
            logs = logs.split("\n")[:logs_limit]
            text += f"""\
{error}
{logs_separator}
"""
            for log in logs:
                text += f"""\
{log}
"""
        else:
            text += f"""\
{last_action_error}
"""
        self.text = text

    def get_message_content(self) -> list[dict]:
        return [{"type": "text", "text": self.text}]

class PreliminaryAnswerPromptPart(VLPromptPart):
    def __init__(
        self, action_set_description: str, use_abstract_example: bool, use_concrete_example: bool
    ):
        text = f"""\
# The action space
Here are all the actions you can take to interact with the browser. \
They are Python functions based on the Playwright library.
{action_set_description}
# The format of the answer
Think about the action to take, and describe the location to take the action. \
Your answer should include one thought and one location.
"""
        if use_abstract_example:
            text += """\
# An abstract example of the answer
<thought>
The thought about the action.
</thought>
<location>
The description of the location.
</location>
"""
        if use_concrete_example:
            text += """\
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

class AnswerPromptPart(VLPromptPart):
    def __init__(
        self, action_set_description: str, use_abstract_example: bool, use_concrete_example: bool
    ):
        text = f"""\
# The action space
Here are all the actions you can take to interact with the browser. \
They are Python functions based on the Playwright library.
{action_set_description}
# The format of the answer
Think about the action to take, and choose it from the action space. \
Your answer should include one thought and one action.
"""
        if use_abstract_example:
            text += """\
# An abstract example of the answer
<thought>
The thought about the action.
</thought>
<action>
The action to take.
</action>
"""
        if use_concrete_example:
            text += """\
# A concrete example of the answer
<thought>
The goal is to click on the numbers in ascending order. \
The smallest number visible on the screen is '1'. \
I will use the 'mouse_click' action to directly click on the number '1'.
</thought>
<action>
mouse_click(50, 50)
</action>
"""
        self.text = text

    def get_message_content(self) -> list[dict]:
        return [{"type": "text", "text": self.text}]


@dataclass
class UIPrompt(VLPrompt):
    introduction_prompt_part: IntroductionPromptPart
    goal_prompt_part: GoalPromptPart
    screenshot_prompt_part: Optional[ScreenshotPromptPart]
    tabs_prompt_part: Optional[TabsPromptPart]
    history_prompt_part: Optional[HistoryPromptPart]
    error_prompt_part: Optional[ErrorPromptPart]
    answer_prompt_part: AnswerPromptPart
    action_validator: callable

    def get_messages(self) -> Discussion:
        message_content = self.introduction_prompt_part.get_message_content()
        message_content.extend(self.goal_prompt_part.get_message_content())
        if self.screenshot_prompt_part is not None:
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
        try:
            answer_dict.update(parse_html_tags_raise(answer_text, keys=["thought", "action"]))
        except ParseError as error:
            answer_dict["parse_error"] = str(error)
            answer_dict["thought"] = answer_text
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
            except Exception as error:
                raise ParseError(str(error))
        return answer_dict


@dataclass
class UIPromptArgs(VLPromptArgs):
    use_screenshot: bool
    use_screenshot_som: bool
    use_tabs: bool
    use_history: bool
    use_error: bool
    use_abstract_example: bool
    use_concrete_example: bool
    extra_instruction: Optional[str]

    def make_prompt(
        self, obs: dict, thoughts: list[str], actions: list[str], action_set: HighLevelActionSet
    ) -> UIPrompt:
        introduction_prompt_part = IntroductionPromptPart()
        goal_prompt_part = GoalPromptPart(obs["goal_object"])
        if self.use_screenshot:
            screenshot_prompt_part = ScreenshotPromptPart(obs["screenshot"])
        else:
            screenshot_prompt_part = None
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
        answer_prompt_part = AnswerPromptPart(
            action_set_description=action_set.describe(
                with_long_description=True, with_examples=False
            ),
            use_abstract_example=self.use_abstract_example,
            use_concrete_example=self.use_concrete_example,
        )
        self.ui_prompt = UIPrompt(
            introduction_prompt_part=introduction_prompt_part,
            goal_prompt_part=goal_prompt_part,
            screenshot_prompt_part=screenshot_prompt_part,
            tabs_prompt_part=tabs_prompt_part,
            history_prompt_part=history_prompt_part,
            error_prompt_part=error_prompt_part,
            answer_prompt_part=answer_prompt_part,
            action_validator=action_set.to_python_code,
        )
        return self.ui_prompt
