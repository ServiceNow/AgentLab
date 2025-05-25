from agentlab.llm.llm_utils import (
    Discussion,
    extract_code_blocks,
    HumanMessage,
    ParseError,
    parse_html_tags_raise,
    SystemMessage,
)
from browsergym.core.action.highlevel import HighLevelActionSet
from dataclasses import dataclass
from PIL import Image
from typing import Optional, Union
from .base import VLPrompt, VLPromptArgs, VLPromptPart
from ..utils import image_to_image_url
import numpy as np


class SystemPromptPart(VLPromptPart):
    def __init__(self):
        self.text = """\
You are an agent working to address a web-based task through step-by-step interactions with the browser. \
At each step, you need to submit an action according to the current state of the browser. \
This action will be executed and the state of the browser will be updated.
"""

    def get_message_content(self) -> list[dict]:
        return [{"type": "text", "text": self.text}]


class InstructionPromptPart(VLPromptPart):
    def __init__(
        self,
        goal_object: list[dict],
        extra_instruction: Optional[str] = None,
    ):
        text = """\
# Instruction
Review the current state of the browser and all other information to find the best next action to achieve the goal.
## Goal
"""
        for item in goal_object:
            if item["type"] == "text":
                text += f"""\
{item['text']}
"""
        if extra_instruction is not None:
            text += f"""\
## Extra Instruction
{extra_instruction}
"""
        self.text = text

    def get_message_content(self) -> list[dict]:
        return [{"type": "text", "text": self.text}]


class ScreenshotPromptPart(VLPromptPart):
    def __init__(self, screenshot: Union[Image.Image, np.ndarray]):
        self.text = """\
# The Screenshot of the Current Web Page
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
# The Open Tabs of the Browser
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
# The Previous Steps
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
# The Error from Last Action
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


class AnswerPromptPart(VLPromptPart):
    def __init__(
        self,
        action_set_description: str,
        use_abstract_example: bool,
        use_concrete_example: bool,
        preliminary_answer: Optional[dict] = None,
    ):
        text = f"""\
# Answer Requirements
## Action Space
Here are all the actions you can take to interact with the browser. \
They are Python functions based on the Playwright library.
{action_set_description}
## Answer Format
Think about the next action to take, and choose it from the action space. \
Your answer should include both the thought and the action.
"""
        if use_abstract_example:
            text += """\
### An Abstract Example of the Answer
<thought>
The thought about the next action.
</thought>
<action>
The next action to take.
</action>
"""
        if use_concrete_example:
            text += """\
### A Concrete Example of the Answer
<thought>
From previous action I tried to set the value of year to "2022", using select_option, but it doesn't appear to be in the form. \
It may be a dynamic dropdown, I will try using click with the bid "a324" and look at the response from the page.
</thought>
<action>
click('a324')
</action>
"""
        if preliminary_answer is not None:
            text += f"""\
## A Preliminary Answer
Here is a preliminary answer, which might be incorrect or inaccurate. \
You can refine it to get your answer.
<thought>
{preliminary_answer['thought']}
</thought>
<action>
{preliminary_answer['action']}
</action>
"""
        self.text = text

    def get_message_content(self) -> list[dict]:
        return [{"type": "text", "text": self.text}]


@dataclass
class UIPrompt(VLPrompt):
    system_prompt_part: SystemPromptPart
    instruction_prompt_part: InstructionPromptPart
    screenshot_prompt_part: Optional[ScreenshotPromptPart]
    tabs_prompt_part: Optional[TabsPromptPart]
    history_prompt_part: Optional[HistoryPromptPart]
    error_prompt_part: Optional[ErrorPromptPart]
    answer_prompt_part: AnswerPromptPart
    action_validator: callable

    def get_messages(self) -> Discussion:
        system_message_content = self.system_prompt_part.get_message_content()
        human_message_content = self.instruction_prompt_part.get_message_content()
        if self.screenshot_prompt_part is not None:
            human_message_content.extend(self.screenshot_prompt_part.get_message_content())
        if self.tabs_prompt_part is not None:
            human_message_content.extend(self.tabs_prompt_part.get_message_content())
        if self.history_prompt_part is not None:
            human_message_content.extend(self.history_prompt_part.get_message_content())
        if self.error_prompt_part is not None:
            human_message_content.extend(self.error_prompt_part.get_message_content())
        human_message_content.extend(self.answer_prompt_part.get_message_content())
        messages = Discussion(
            [SystemMessage(system_message_content), HumanMessage(human_message_content)]
        )
        messages.merge()
        return messages

    def parse_answer(self, answer_text: str) -> dict:
        answer_dict = {}
        try:
            answer_dict.update(
                parse_html_tags_raise(answer_text, keys=["thought"], merge_multiple=True)
            )
        except ParseError as error:
            answer_dict["thought"] = answer_text
            answer_dict["thought_parse_error"] = str(error)
        try:
            answer_dict.update(
                parse_html_tags_raise(answer_text, keys=["action"], merge_multiple=True)
            )
        except ParseError as error:
            code_blocks = extract_code_blocks(answer_text)
            if len(code_blocks) == 0:
                raise error
            else:
                answer_dict["action"] = "\n".join([block for _, block in code_blocks])
                answer_dict["action_parse_error"] = str(error)
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
        self,
        obs: dict,
        thoughts: list[str],
        actions: list[str],
        action_set: HighLevelActionSet,
        preliminary_answer: Optional[dict] = None,
    ) -> UIPrompt:
        system_prompt_part = SystemPromptPart()
        instruction_prompt_part = InstructionPromptPart(
            goal_object=obs["goal_object"], extra_instruction=self.extra_instruction
        )
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
        if self.use_history:
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
            preliminary_answer=preliminary_answer,
        )
        return UIPrompt(
            system_prompt_part=system_prompt_part,
            instruction_prompt_part=instruction_prompt_part,
            screenshot_prompt_part=screenshot_prompt_part,
            tabs_prompt_part=tabs_prompt_part,
            history_prompt_part=history_prompt_part,
            error_prompt_part=error_prompt_part,
            answer_prompt_part=answer_prompt_part,
            action_validator=action_set.to_python_code,
        )
