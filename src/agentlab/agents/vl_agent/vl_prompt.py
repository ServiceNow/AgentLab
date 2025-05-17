from abc import ABC, abstractmethod
from agentlab.agents import dynamic_prompting as dp
from agentlab.llm.llm_utils import HumanMessage, SystemMessage
from browsergym.experiments.benchmark.base import HighLevelActionSetArgs
from browsergym.core.action.highlevel import HighLevelActionSet
from dataclasses import dataclass
from PIL import Image
from typing import Optional, Union
from .utils import image_to_image_url
import numpy as np
import time


class VLPromptPart(ABC):
    @abstractmethod
    def get_message_items(self) -> list[dict]:
        raise NotImplementedError


class VLPrompt(ABC):
    @abstractmethod
    def get_messages(self) -> list[Union[SystemMessage, HumanMessage]]:
        raise NotImplementedError

    @abstractmethod
    def parse_answer(self, answer_text: str) -> dict:
        raise NotImplementedError


class VLPromptArgs(ABC):
    @abstractmethod
    def make_prompt(self, obs: dict, actions: list[str], thoughts: list[str]) -> VLPrompt:
        raise NotImplementedError


class SystemPromptPart(VLPromptPart):
    def __init__(self, text: Optional[str]):
        if text is None:
            text = """\
You are an agent trying to solve a web task based on the content of the page and user instructions. \
You can interact with the page and explore, and send messages to the user. \
Each time you submit an action it will be sent to the browser and you will receive a new page.
"""
        self.text = text

    def get_message_items(self) -> list[dict]:
        return [{"type": "text", "text": self.text}]


class ChatInstructionPromptPart(VLPromptPart):
    def __init__(
        self,
        chat_messages: list[dict],
        extra_instruction: Optional[str],
    ):
        text = """\
# Instruction
Your goal is to help the user perform tasks using a web browser. \
You can communicate with the user via a chat, in which the user gives you instructions and in which you can send back messages. \
Review the current state of the page and all other information to find the best possible next action to accomplish your goal. \
Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.
## Chat Messages
"""
        for chat_message in chat_messages:
            text += f"""\
[{time.asctime(time.localtime(chat_message['timestamp']))}] {chat_message['role']}: {chat_message['message']}
"""
        if extra_instruction is not None:
            text += f"""\
## Extra Instruction
{extra_instruction}
"""
        self.text = text

    def get_message_items(self) -> list[dict]:
        return [{"type": "text", "text": self.text}]


class GoalInstructionPromptPart(VLPromptPart):
    def __init__(
        self,
        goal_object: list[dict],
        extra_instruction: Optional[str],
    ):
        text = """\
# Instruction
Review the current state of the page and all other information to find the best possible next action to accomplish your goal. \
Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.
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

    def get_message_items(self) -> list[dict]:
        return [{"type": "text", "text": self.text}]


class ScreenshotPromptPart(VLPromptPart):
    def __init__(self, screenshot: Union[Image.Image, np.ndarray]):
        self.text = """\
# Screenshot
"""
        self.image_url = image_to_image_url(screenshot)

    def get_message_items(self) -> list[dict]:
        return [
            {"type": "text", "text": self.text},
            {"type": "image_url", "image_url": {"url": self.image_url}},
        ]


class TabsPromptPart(VLPromptPart):
    def __init__(
        self, open_pages_titles: list[str], open_pages_urls: list[str], active_page_index: int
    ):
        text = """\
# Open Tabs
"""
        for index, (title, url) in enumerate(zip(open_pages_titles, open_pages_urls)):
            text += f"""\
Tab {index}{' (active tab)' if index == active_page_index else ''}: {title} ({url})
"""
        self.text = text

    def get_message_items(self) -> list[dict]:
        return [{"type": "text", "text": self.text}]


class ErrorPromptPart(VLPromptPart):
    def __init__(self, last_action_error: str):
        text = """\
# Error from Last Action
"""
        separator = "Call log:"
        if separator in last_action_error:
            error, logs = last_action_error.split(separator)
            error = error.strip()
            logs = logs.split("\n")
            text += f"""\
{error}
{separator}
"""
            for log in logs[:10]:
                text += f"""\
{log}
"""
        else:
            text += f"""\
{last_action_error}
"""
        self.text = text

    def get_message_items(self) -> list[dict]:
        return [{"type": "text", "text": self.text}]


class HistoryPromptPart(VLPromptPart):
    def __init__(self, thoughts: list[str], actions: list[str]):
        text = """\
# Thoughts and Actions of Previous Steps
"""
        for index, (thought, action) in enumerate(zip(thoughts, actions)):
            text += f"""
## Step {index}
### Thought
{thought}
### Action
{action}
"""
        self.text = text

    def get_message_items(self) -> list[dict]:
        return [{"type": "text", "text": self.text}]


class AnswerPromptPart(VLPromptPart):
    def __init__(
        self,
        action_set: HighLevelActionSet,
        use_abstract_example: bool,
        use_concrete_example: bool,
        preliminary_answer: Optional[dict],
    ):
        text = f"""\
# Answer Format Requirements
## Action Space
These actions allow you to interact with your environment. \
Most of them are python functions executing playwright code.
{action_set.describe(with_long_description=True, with_examples=False)}
"""
        if use_abstract_example:
            text += """
## Abstract Example
<thought>
The thought about which action to take at the current step.
</thought>
<action>
One single action to be executed. You can only use one action at a time.
</action>
"""
        if use_concrete_example:
            text += """
## Concrete Example
<thought>
From previous action I tried to set the value of year to "2022", using select_option, but it doesn't appear to be in the form. \
It may be a dynamic dropdown, I will try using click with the bid "a324" and look at the response from the page.
</thought>
<action>
click('a324')
</action>
"""
        if preliminary_answer is not None:
            text += f"""
## Preliminary Anser
Here is a preliminary anser, which might be incorrect or inaccurate. \
Refine it to get the final answer.
<thought>
{preliminary_answer['thought']}
</thought>
<action>
{preliminary_answer['action']}
</action>
"""
        self.text = text

    def get_message_items(self) -> list[dict]:
        return [{"type": "text", "text": self.text}]


@dataclass
class UIPrompt(VLPrompt):
    instructions: Union[dp.ChatInstructions, dp.GoalInstructions]
    screenshot: Optional[Image.Image]
    observation: dp.Observation
    history: dp.History
    think: dp.Think
    action_prompt: dp.ActionPrompt
    abstract_example: Optional[str]
    concrete_example: Optional[str]
    preliminary_answer: Optional[str]

    def get_messages(self) -> list[Union[SystemMessage, HumanMessage]]:
        message = HumanMessage(self.instructions.prompt)
        if self.screenshot is not None:
            message.add_text("# Screenshot:\n")
            message.add_image(self.screenshot)
        message.add_text(self.observation.prompt)
        message.add_text(self.history.prompt)
        message.add_text(self.think.prompt)
        message.add_text(self.action_prompt.prompt)
        if self.abstract_example is not None:
            message.add_text(self.abstract_example)
        if self.concrete_example is not None:
            message.add_text(self.concrete_example)
        if self.preliminary_answer is not None:
            message.add_text(self.preliminary_answer)
        return message

    def parse_answer(self, answer_text: str) -> dict:
        answer_dict = {}
        answer_dict.update(self.think.parse_answer(answer_text))
        answer_dict.update(self.action_prompt.parse_answer(answer_text))
        return answer_dict


@dataclass
class UIPromptArgs(VLPromptArgs):
    action_set_args: HighLevelActionSetArgs
    enable_chat: bool
    use_screenshot: bool
    use_som: bool
    use_tabs: bool
    use_error: bool
    use_history: bool
    use_abstract_example: bool
    use_concrete_example: bool

    def make_prompt(
        self,
        obs: dict,
        actions: list[str],
        thoughts: list[str],
        extra_instruction: Optional[str],
        preliminary_answer: Optional[dict],
    ) -> UIPrompt:
        if self.enable_chat:
            instruction_prompt_part = ChatInstructionPromptPart(
                chat_messages=obs["chat_messages"],
                extra_instruction=extra_instruction,
            )
        else:
            instruction_prompt_part = GoalInstructionPromptPart(
                goal_object=obs["goal_object"],
                extra_instruction=extra_instruction,
            )
        if self.use_screenshot:
            if self.use_som:
                screenshot = obs["screenshot_som"]
            else:
                screenshot = obs["screenshot"]
            screenshot_prompt_part = ScreenshotPromptPart(screenshot)
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
        if self.use_error and obs["last_action_error"]:
            error_prompt_part = ErrorPromptPart(obs["last_action_error"])
        else:
            error_prompt_part = None
        if self.use_history:
            history_prompt_part = HistoryPromptPart(thoughts=thoughts, actions=actions)
        else:
            history_prompt_part = None
        answer_prompt_part = AnswerPromptPart(
            action_set=self.action_set_args.make_action_set(),
            use_abstract_example=self.use_abstract_example,
            use_concrete_example=self.use_concrete_example,
            preliminary_answer=preliminary_answer,
        )
        return UIPrompt(
            instruction_prompt_part=instruction_prompt_part,
            screenshot_prompt_part=screenshot_prompt_part,
            tabs_prompt_part=tabs_prompt_part,
            error_prompt_part=error_prompt_part,
            history_prompt_part=history_prompt_part,
            answer_prompt_part=answer_prompt_part,
        )
