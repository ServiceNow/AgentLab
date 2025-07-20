from agentlab.llm.llm_utils import (
    extract_code_blocks,
    HumanMessage,
    ParseError,
    parse_html_tags_raise,
)
from ast import literal_eval
from dataclasses import dataclass
from PIL import Image
from typing import Callable, Optional, Union
from .base import VLPrompt, VLPromptArgs, VLPromptPart
from ..utils import image_to_image_url
import numpy as np


class IntroductionPromptPart(VLPromptPart):
    def __init__(self):
        self._message_content = [
            {
                "type": "text",
                "text": """\
You are an agent working to address a web-based task through step-by-step interaction with the browser. \
To achieve the goal of the task, at each step, you need to submit an action based on the current state of the browser. \
This action will be executed to update the state of the browser, and you will proceed to the next step.
""",
            }
        ]

    @property
    def message_content(self) -> list[dict]:
        return self._message_content


class GoalPromptPart(VLPromptPart):
    def __init__(self, goal_object: list[dict]):
        self._message_content = [
            {
                "type": "text",
                "text": """
# The goal of the task
""",
            }
        ]
        self._message_content.extend(goal_object)

    @property
    def message_content(self) -> list[dict]:
        return self._message_content


class InteractionPromptPart(VLPromptPart):
    def __init__(
        self,
        current_screenshot: Union[Image.Image, np.ndarray],
        screenshot_history: list[Union[Image.Image, np.ndarray]],
        think_history: list[str],
        action_history: list[str],
        use_screenshot_history: bool,
    ):
        self._message_content = []
        if len(screenshot_history) == len(think_history) == len(action_history) != 0:
            self._message_content.append(
                {
                    "type": "text",
                    "text": """
# The previous steps
""",
                }
            )
            for index, (screenshot, think, action) in enumerate(
                zip(screenshot_history, think_history, action_history)
            ):
                self._message_content.append(
                    {
                        "type": "text",
                        "text": f"""
## Step {index}
""",
                    }
                )
                if use_screenshot_history:
                    self._message_content.append(
                        {
                            "type": "text",
                            "text": """
### Screenshot
""",
                        }
                    )
                    self._message_content.append(
                        {"type": "image_url", "image_url": {"url": image_to_image_url(screenshot)}}
                    )
                self._message_content.append(
                    {
                        "type": "text",
                        "text": f"""
### Think

{think}

### Action

{action}
""",
                    }
                )
        self._message_content.append(
            {
                "type": "text",
                "text": """
# The current screenshot
""",
            }
        )
        self._message_content.append(
            {"type": "image_url", "image_url": {"url": image_to_image_url(current_screenshot)}}
        )

    @property
    def message_content(self) -> list[dict]:
        return self._message_content


class TabsPromptPart(VLPromptPart):
    def __init__(
        self, open_pages_titles: list[str], open_pages_urls: list[str], active_page_index: int
    ):
        self._message_content = []
        if len(open_pages_titles) == len(open_pages_urls) != 0:
            self._message_content.append(
                {
                    "type": "text",
                    "text": """
# The open tabs of the browser
""",
                }
            )
            for index, (title, url) in enumerate(zip(open_pages_titles, open_pages_urls)):
                self._message_content.append(
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

    @property
    def message_content(self) -> list[dict]:
        return self._message_content


class ErrorPromptPart(VLPromptPart):
    def __init__(
        self,
        last_action_error: str,
        logs_separator: str = "Call log:",
        logs_limit: int = 5,
    ):
        self._message_content = []
        if len(last_action_error) != 0:
            self._message_content.append(
                {
                    "type": "text",
                    "text": """
# The error caused by the last action
""",
                }
            )
            if logs_separator in last_action_error:
                error, logs = last_action_error.split(logs_separator)
                logs = "\n".join(logs.split("\n")[:logs_limit])
                self._message_content.append(
                    {
                        "type": "text",
                        "text": f"""
{error}

{logs_separator}

{logs}
""",
                    }
                )
            else:
                self._message_content.append(
                    {
                        "type": "text",
                        "text": f"""
{last_action_error}
""",
                    }
                )

    @property
    def message_content(self) -> list[dict]:
        return self._message_content


class PreliminaryAnswerPromptPart(VLPromptPart):
    def __init__(
        self, action_set_description: str, use_abstract_example: bool, use_concrete_example: bool
    ):
        self._message_content = [
            {
                "type": "text",
                "text": f"""
# The action space

Here are all the actions you can take to interact with the browser.

{action_set_description}

# The answer requirements

Think about the action, and describe the location where the action is to be taken. \
Your answer should contain one think and one location.
""",
            }
        ]
        if use_abstract_example:
            self._message_content.append(
                {
                    "type": "text",
                    "text": """
# An abstract example of the answer

<think>
The think about the action.
</think>
<location>
The description of the location where the action is to be taken.
</location>
""",
                }
            )
        if use_concrete_example:
            self._message_content.append(
                {
                    "type": "text",
                    "text": """
# A concrete example of the answer

<think>
The goal is to click on the numbers in ascending order. \
The smallest number visible on the screen is '1'. \
I will use the 'mouse_click' action to directly click on the number '1'.
</think>
<location>
The number '1' in the top-left quadrant of the white area.
</location>
""",
                }
            )

    @property
    def message_content(self) -> list[dict]:
        return self._message_content


class FinalAnswerPromptPart(VLPromptPart):
    def __init__(
        self,
        action_set_description: str,
        main_think: str,
        auxiliary_location: str,
        use_abstract_example: bool,
        use_concrete_example: bool,
    ):
        self._message_content = [
            {
                "type": "text",
                "text": f"""
# The action space

Here are all the actions you can take to interact with the browser.

{action_set_description}

# The think about the action

{main_think}

# The location where the action is to be taken

{auxiliary_location}

# The answer requirements

Formulate the action to be taken. \
Your answer should contain only one action.
""",
            }
        ]
        if use_abstract_example:
            self._message_content.append(
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
            self._message_content.append(
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

    @property
    def message_content(self) -> list[dict]:
        return self._message_content


@dataclass
class MainUIPrompt(VLPrompt):
    introduction_prompt_part: IntroductionPromptPart
    goal_prompt_part: GoalPromptPart
    interaction_prompt_part: InteractionPromptPart
    tabs_prompt_part: Optional[TabsPromptPart]
    error_prompt_part: Optional[ErrorPromptPart]
    answer_prompt_part: Union[PreliminaryAnswerPromptPart, FinalAnswerPromptPart]
    action_validator: Callable

    def __post_init__(self):
        message_content = []
        message_content.extend(self.introduction_prompt_part.message_content)
        message_content.extend(self.goal_prompt_part.message_content)
        message_content.extend(self.interaction_prompt_part.message_content)
        if self.tabs_prompt_part is not None:
            message_content.extend(self.tabs_prompt_part.message_content)
        if self.error_prompt_part is not None:
            message_content.extend(self.error_prompt_part.message_content)
        message_content.extend(self.answer_prompt_part.message_content)
        merged_message_content = []
        for item in message_content:
            if (
                item["type"] == "text"
                and len(merged_message_content) != 0
                and merged_message_content[-1]["type"] == "text"
            ):
                merged_message_content[-1]["text"] += item["text"]
            else:
                merged_message_content.append(item)
        self._message = HumanMessage(merged_message_content)

    @property
    def message(self) -> HumanMessage:
        return self._message

    def parse_answer(self, answer_content: list[dict]) -> dict:
        answer_text = answer_content[0]["text"]
        if isinstance(self.answer_prompt_part, PreliminaryAnswerPromptPart):
            try:
                result = parse_html_tags_raise(answer_text, keys=["think", "location"])
                main_think = result["think"]
                main_location = result["location"]
            except ParseError as error:
                raise error
            answer_dict = {"main_think": main_think, "main_location": main_location}
        else:
            try:
                main_action = parse_html_tags_raise(answer_text, keys=["action"])["action"]
            except ParseError as error:
                code_blocks = extract_code_blocks(answer_text)
                if len(code_blocks) == 0:
                    raise error
                else:
                    main_action = "\n".join([block for _, block in code_blocks])
            answer_dict = {"main_action": main_action}
            if answer_dict["main_action"] == "None":
                answer_dict["main_action"] = None
            else:
                try:
                    self.action_validator(answer_dict["main_action"])
                except:
                    raise ParseError(f"Invalid action: {answer_dict['main_action']}")
        return answer_dict


@dataclass
class AuxiliaryUIPrompt(VLPrompt):
    current_screenshot: Union[Image.Image, np.ndarray]
    screenshot_history: list[Union[Image.Image, np.ndarray]]
    main_think: str
    main_location: str
    use_screenshot_history: bool
    use_location_reasoning: bool
    location_adapter: Callable

    def __post_init__(self):
        message_content = []
        screenshots = []
        if self.use_screenshot_history:
            screenshots.extend(self.screenshot_history)
        screenshots.append(self.current_screenshot)
        for screenshot in screenshots:
            message_content.append(
                {"type": "image_url", "image_url": {"url": image_to_image_url(screenshot)}}
            )
        if self.use_location_reasoning:
            message_content.append(
                {
                    "type": "text",
                    "text": "\n".join(
                        [
                            f"You are StarVLM-R1, a reasoning GUI Agent Assistant. In the UI screenshot, I want you to continue executing the command '{self.main_think}'.",
                            "Please provide the action to perform (enumerate from ['click']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.",
                            "Output the response as as a JSON list follows:",
                            "[{'action': enum['complete', 'close/delete', 'press_home', 'click', 'press_back', 'type', 'select', 'scroll', 'enter'], 'point': [x, y], 'input_text': 'no input text [default]', 'explanation': 'explanation and reason for why selecting this action and region'}]",
                            "Example:",
                            "[{'action': enum['click'], 'point': [123, 300], 'input_text': 'no input text', 'explanation': 'explanation and reason for why selecting this action and region'}]",
                            "Please strictly follow the format.",
                        ]
                    ),
                }
            )
        else:
            message_content.append(
                {
                    "type": "text",
                    "text": f"""\
Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.

- Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
- If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
- Your answer should be a single string (x, y) corresponding to the point of the interest.

Description: {self.main_location}

Answer:""",
                }
            )
        self._message = HumanMessage(message_content)

    @property
    def message(self) -> HumanMessage:
        return self._message

    def parse_answer(self, answer_content: list[dict]) -> dict:
        answer_text = answer_content[0]["text"]
        try:
            if self.use_location_reasoning:
                x, y = literal_eval(answer_text.strip())[0]["point"]
            else:
                x, y = literal_eval(answer_text.strip())
            x, y = self.location_adapter(self.current_screenshot, int(x), int(y))
        except:
            raise ParseError(f"Invalid answer: {answer_text}")
        return {"auxiliary_location": f"({x}, {y})", "auxiliary_response": answer_text}


@dataclass
class UIPromptArgs(VLPromptArgs):
    use_screenshot_history: bool
    use_tabs: bool
    use_error: bool
    use_abstract_example: bool
    use_concrete_example: bool
    use_location_reasoning: bool

    def make_main_prompt(
        self,
        obs: dict,
        screenshot_history: list[Union[Image.Image, np.ndarray]],
        think_history: list[str],
        action_history: list[str],
        action_set_description: str,
        action_validator: Callable,
        extra_info: Optional[dict] = None,
    ) -> MainUIPrompt:
        introduction_prompt_part = IntroductionPromptPart()
        goal_prompt_part = GoalPromptPart(obs["goal_object"])
        interaction_prompt_part = InteractionPromptPart(
            obs["screenshot"],
            screenshot_history=screenshot_history,
            think_history=think_history,
            action_history=action_history,
            use_screenshot_history=self.use_screenshot_history,
        )
        if self.use_tabs:
            tabs_prompt_part = TabsPromptPart(
                obs["open_pages_titles"],
                open_pages_urls=obs["open_pages_urls"],
                active_page_index=obs["active_page_index"],
            )
        else:
            tabs_prompt_part = None
        if self.use_error:
            error_prompt_part = ErrorPromptPart(obs["last_action_error"])
        else:
            error_prompt_part = None
        if extra_info is None:
            answer_prompt_part = PreliminaryAnswerPromptPart(
                action_set_description,
                use_abstract_example=self.use_abstract_example,
                use_concrete_example=self.use_concrete_example,
            )
        else:
            answer_prompt_part = FinalAnswerPromptPart(
                action_set_description,
                main_think=extra_info["main_think"],
                auxiliary_location=extra_info["auxiliary_location"],
                use_abstract_example=self.use_abstract_example,
                use_concrete_example=self.use_concrete_example,
            )
        return MainUIPrompt(
            introduction_prompt_part=introduction_prompt_part,
            goal_prompt_part=goal_prompt_part,
            interaction_prompt_part=interaction_prompt_part,
            tabs_prompt_part=tabs_prompt_part,
            error_prompt_part=error_prompt_part,
            answer_prompt_part=answer_prompt_part,
            action_validator=action_validator,
        )

    def make_auxiliary_prompt(
        self,
        obs: dict,
        screenshot_history: list[Union[Image.Image, np.ndarray]],
        location_adapter: Callable,
        extra_info: dict,
    ) -> AuxiliaryUIPrompt:
        return AuxiliaryUIPrompt(
            current_screenshot=obs["screenshot"],
            screenshot_history=screenshot_history,
            main_think=extra_info["main_think"],
            main_location=extra_info["main_location"],
            use_screenshot_history=self.use_screenshot_history,
            use_location_reasoning=self.use_location_reasoning,
            location_adapter=location_adapter,
        )
