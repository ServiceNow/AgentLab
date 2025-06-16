from agentlab.llm.llm_utils import (
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
        self.message_content = [
            {
                "type": "text",
                "text": """\
You are an agent working to address a web-based task through step-by-step interaction with the browser. \
To achieve the goal of the task, at each step, you need to submit an action based on the current state of the browser. \
This action will be executed to update the state of the browser, and you will proceed to the next step.
""",
            }
        ]

    def get_message_content(self) -> list[dict]:
        return self.message_content


class GoalPromptPart(VLPromptPart):
    def __init__(self, goal_object: list[dict]):
        self.message_content = [
            {
                "type": "text",
                "text": """
# The goal of the task
""",
            }
        ]
        self.message_content.extend(goal_object)

    def get_message_content(self) -> list[dict]:
        return self.message_content


class InteractionPromptPart(VLPromptPart):
    def __init__(
        self,
        current_screenshot: Union[Image.Image, np.ndarray],
        screenshot_history: list[Union[Image.Image, np.ndarray]],
        thought_history: list[str],
        action_history: list[str],
        use_screenshot_history: bool,
    ):
        self.message_content = [
            {
                "type": "text",
                "text": """
# The previous steps to achieve the goal
""",
            }
        ]
        for index, (screenshot, thought, action) in enumerate(
            zip(screenshot_history, thought_history, action_history)
        ):
            self.message_content.append(
                {
                    "type": "text",
                    "text": f"""
## Step {index}
""",
                }
            )
            if use_screenshot_history:
                self.message_content.append(
                    {
                        "type": "text",
                        "text": """
### Screenshot
""",
                    }
                )
                self.message_content.append(
                    {"type": "image_url", "image_url": {"url": image_to_image_url(screenshot)}}
                )
            self.message_content.append(
                {
                    "type": "text",
                    "text": f"""
### Thought

{thought}

### Action

{action}
""",
                }
            )
        self.message_content.append(
            {
                "type": "text",
                "text": """
# The screenshot of the current step
""",
            }
        )
        self.message_content.append(
            {"type": "image_url", "image_url": {"url": image_to_image_url(current_screenshot)}}
        )

    def get_message_content(self) -> list[dict]:
        return self.message_content


class TabsPromptPart(VLPromptPart):
    def __init__(
        self, open_pages_titles: list[str], open_pages_urls: list[str], active_page_index: int
    ):
        self.message_content = [
            {
                "type": "text",
                "text": """
# The open tabs of the browser
""",
            }
        ]
        for index, (title, url) in enumerate(zip(open_pages_titles, open_pages_urls)):
            self.message_content.append(
                {
                    "type": "text",
                    "text": f"""
## Tab {index}{' (active)' if index == active_page_index else ''}

### Title

{title}

### URL

{url}
""",
                }
            )

    def get_message_content(self) -> list[dict]:
        return self.message_content


class ErrorPromptPart(VLPromptPart):
    def __init__(
        self,
        last_action_error: str,
        logs_separator: str = "Call log:",
        logs_limit: int = 5,
    ):
        self.message_content = [
            {
                "type": "text",
                "text": """
# The error caused by the last action
""",
            }
        ]
        if logs_separator in last_action_error:
            error, logs = last_action_error.split(logs_separator)
            logs = logs.split("\n")[:logs_limit]
            self.message_content.append(
                {
                    "type": "text",
                    "text": f"""
{error}

{logs_separator}
""",
                }
            )
            for log in logs:
                self.message_content.append(
                    {
                        "type": "text",
                        "text": f"""
{log}
""",
                    }
                )
        else:
            self.message_content.append(
                {
                    "type": "text",
                    "text": f"""
{last_action_error}
""",
                }
            )

    def get_message_content(self) -> list[dict]:
        return self.message_content


class PreliminaryAnswerPromptPart(VLPromptPart):
    def __init__(
        self, action_set_description: str, use_abstract_example: bool, use_concrete_example: bool
    ):
        self.message_content = [
            {
                "type": "text",
                "text": f"""
# The action space

Here are all the actions you can take to interact with the browser.

{action_set_description}

# The answer requirements

Think about the action, and describe the location where the action is to be taken. \
Your answer should contain one thought and one location.
""",
            }
        ]
        if use_abstract_example:
            self.message_content.append(
                {
                    "type": "text",
                    "text": """
# An abstract example of the answer

<thought>
The thought about the action.
</thought>
<location>
The description of the location where the action is to be taken.
</location>
""",
                }
            )
        if use_concrete_example:
            self.message_content.append(
                {
                    "type": "text",
                    "text": """
# A concrete example of the answer

<thought>
The goal is to click on the numbers in ascending order. \
The smallest number visible on the screen is '1'. \
I will use the 'mouse_click' action to directly click on the number '1'.
</thought>
<location>
The number '1' in the top-left quadrant of the white area.
</location>
""",
                }
            )

    def get_message_content(self) -> list[dict]:
        return self.message_content


class FinalAnswerPromptPart(VLPromptPart):
    def __init__(
        self,
        action_set_description: str,
        extra_info: dict,
        use_abstract_example: bool,
        use_concrete_example: bool,
    ):
        self.message_content = [
            {
                "type": "text",
                "text": f"""
# The action space

Here are all the actions you can take to interact with the browser.

{action_set_description}

# The thought about the action

{extra_info['thought']}

# The coordinates where the action is to be taken

{extra_info['coordinates']}

# The answer requirements

Formulate the action to be taken. \
Your answer should contain only one action.
""",
            }
        ]
        if use_abstract_example:
            self.message_content.append(
                {
                    "type": "text",
                    "text": """
# An abstract example of the answer

<action>
The action to be taken.
</action>
""",
                }
            )
        if use_concrete_example:
            self.message_content.append(
                {
                    "type": "text",
                    "text": """
# A concrete example of the answer

<action>
mouse_click(50, 50)
</action>
""",
                }
            )

    def get_message_content(self) -> list[dict]:
        return self.message_content


@dataclass
class MainUIPrompt(VLPrompt):
    introduction_prompt_part: IntroductionPromptPart
    goal_prompt_part: GoalPromptPart
    interaction_prompt_part: InteractionPromptPart
    tabs_prompt_part: Optional[TabsPromptPart]
    error_prompt_part: Optional[ErrorPromptPart]
    answer_prompt_part: Union[PreliminaryAnswerPromptPart, FinalAnswerPromptPart]
    action_validator: Callable

    def get_message(self) -> HumanMessage:
        message_content = self.introduction_prompt_part.get_message_content()
        message_content.extend(self.goal_prompt_part.get_message_content())
        message_content.extend(self.interaction_prompt_part.get_message_content())
        if self.tabs_prompt_part is not None:
            message_content.extend(self.tabs_prompt_part.get_message_content())
        if self.error_prompt_part is not None:
            message_content.extend(self.error_prompt_part.get_message_content())
        message_content.extend(self.answer_prompt_part.get_message_content())
        message = HumanMessage(message_content)
        message.merge()
        return message

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

    def get_message(self) -> HumanMessage:
        message_content = [
            {"type": "image_url", "image_url": {"url": image_to_image_url(self.screenshot)}},
            {
                "type": "text",
                "text": f"""\
Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on this screen based on a description. \
Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible. \
If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose. \
Your answer should be a single string (x, y) corresponding to the point of interest.

Description: {self.location}
""",
            },
        ]
        return HumanMessage(message_content)

    def parse_answer(self, answer_content: list[dict]) -> dict:
        answer_text = answer_content[0]["text"]
        answer_dict = {"coordinates": answer_text}
        return answer_dict


@dataclass
class UIPromptArgs(VLPromptArgs):
    use_screenshot_history: bool
    use_tabs: bool
    use_error: bool
    use_abstract_example: bool
    use_concrete_example: bool

    def make_prompt(
        self,
        obs: dict,
        screenshot_history: list[Union[Image.Image, np.ndarray]],
        thought_history: list[str],
        action_history: list[str],
        action_set: HighLevelActionSet,
        extra_info: Optional[dict] = None,
    ) -> Union[MainUIPrompt, AuxiliaryUIPrompt]:
        introduction_prompt_part = IntroductionPromptPart()
        goal_prompt_part = GoalPromptPart(obs["goal_object"])
        interaction_prompt_part = InteractionPromptPart(
            obs["screenshot"],
            screenshot_history,
            thought_history,
            action_history,
            self.use_screenshot_history,
        )
        if self.use_tabs and len(obs["open_pages_titles"]) == len(obs["open_pages_urls"]) > 1:
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
        if extra_info is None:
            preliminary_answer_prompt_part = PreliminaryAnswerPromptPart(
                action_set_description=action_set.describe(
                    with_long_description=True, with_examples=False
                ),
                use_abstract_example=self.use_abstract_example,
                use_concrete_example=self.use_concrete_example,
            )
            return MainUIPrompt(
                introduction_prompt_part=introduction_prompt_part,
                goal_prompt_part=goal_prompt_part,
                interaction_prompt_part=interaction_prompt_part,
                tabs_prompt_part=tabs_prompt_part,
                error_prompt_part=error_prompt_part,
                answer_prompt_part=preliminary_answer_prompt_part,
                action_validator=action_set.to_python_code,
            )
        else:
            if "coordinates" in extra_info:
                final_answer_prompt_part = FinalAnswerPromptPart(
                    action_set_description=action_set.describe(
                        with_long_description=True, with_examples=False
                    ),
                    extra_info=extra_info,
                    use_abstract_example=self.use_abstract_example,
                    use_concrete_example=self.use_concrete_example,
                )
                return MainUIPrompt(
                    introduction_prompt_part=introduction_prompt_part,
                    goal_prompt_part=goal_prompt_part,
                    interaction_prompt_part=interaction_prompt_part,
                    tabs_prompt_part=tabs_prompt_part,
                    error_prompt_part=error_prompt_part,
                    answer_prompt_part=final_answer_prompt_part,
                    action_validator=action_set.to_python_code,
                )
            else:
                return AuxiliaryUIPrompt(
                    screenshot=obs["screenshot"], location=extra_info["location"]
                )
