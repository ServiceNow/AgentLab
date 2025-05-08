import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import anthropic
import openai
from anthropic import Anthropic
from openai import OpenAI

from agentlab.llm import tracking

from .base_api import BaseModelArgs

type ContentItem = Dict[str, Any]
type Message = Dict[str, Union[str, List[ContentItem]]]


@dataclass
class ResponseLLMOutput:
    """Serializable object for the output of a response LLM."""

    raw_response: Any
    think: str
    action: str
    last_computer_call_id: str
    assistant_message: Any


class MessageBuilder:
    def __init__(self, role: str):
        self.role = role
        self.content: List[ContentItem] = []
        self.tool_call_id = None

    @staticmethod
    def system() -> "MessageBuilder":
        return MessageBuilder(role="system")

    @staticmethod
    def user() -> "MessageBuilder":
        return MessageBuilder(role="user")

    @staticmethod
    def assistant() -> "MessageBuilder":
        return MessageBuilder(role="assistant")

    @staticmethod
    def tool() -> "MessageBuilder":
        return MessageBuilder(role="tool")

    def add_text(self, text: str) -> "MessageBuilder":
        self.content.append({"text": text})
        return self

    def add_image(self, image: str) -> "MessageBuilder":
        self.content.append({"image": image})
        return self

    def add_tool_id(self, tool_id: str) -> "MessageBuilder":
        self.tool_call_id = tool_id
        return self

    def to_openai(self) -> List[Message]:
        content = []
        for item in self.content:
            if "text" in item:
                content.append({"type": "input_text", "text": item["text"]})
            elif "image" in item:
                content.append({"type": "input_image", "image_url": item["image"]})
        res = [{"role": self.role, "content": content}]

        if self.role == "tool":
            assert self.tool_call_id is not None, "Tool call ID is required for tool messages"
            # tool messages can only take text with openai
            # we need to split the first content element if it's text and use it
            # then open a new (user) message with the rest
            # a function_call_output dict has keys "call_id", "type" and "output"
            res[0]["call_id"] = self.tool_call_id
            res[0]["type"] = "function_call_output"
            res[0].pop("role", None)  # make sure to remove role
            text_content = (
                content.pop(0)["text"]
                if "text" in content[0]
                else "Tool call answer in next message"
            )
            res[0]["output"] = text_content
            res[0].pop("content", None)  # make sure to remove content
            res.append({"role": "user", "content": content})

        return res

    def to_anthropic(self) -> List[Message]:
        content = []

        if self.role == "system":
            logging.warning(
                "In the Anthropic API, system messages should be passed as a direct input to the client."
            )
            return []

        for item in self.content:
            if "text" in item:
                content.append({"type": "text", "text": item["text"]})
            elif "image" in item:
                img_str: str = item["image"]
                # make sure to get rid of the image type for anthropic
                # e.g. "data:image/png;base64"
                if img_str.startswith("data:image/png;base64,"):
                    img_str = img_str[len("data:image/png;base64,") :]
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",  # currently only base64 is supported
                            "media_type": "image/png",  # currently only png is supported
                            "data": img_str,
                        },
                    }
                )
        res = [{"role": self.role, "content": content}]

        if self.role == "tool":
            assert self.tool_call_id is not None, "Tool call ID is required for tool messages"
            res[0]["role"] = "user"
            res[0]["content"] = [
                {
                    "type": "tool_result",
                    "tool_use_id": self.tool_call_id,
                    "content": res[0]["content"],
                }
            ]
        return res

    def to_chat_completion(self) -> List[Message]: ...

    def to_markdown(self) -> str:
        content = []
        for item in self.content:
            if "text" in item:
                content.append(item["text"])
            elif "image" in item:
                content.append(f"![Image]({item['image']})")

        # Format the role as a header
        res = f"## {self.role.capitalize()} Message\n\n"

        # Add content with spacing between items
        res += "\n\n---\n\n".join(content)

        # Add tool call ID if the role is "tool"
        if self.role == "tool":
            assert self.tool_call_id is not None, "Tool call ID is required for tool messages"
            res += f"\n\n---\n\n**Tool Call ID:** `{self.tool_call_id}`"

        return res


class BaseResponseModel(ABC):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 100,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_kwargs = extra_kwargs or {}

    def __call__(self, messages: list[dict | MessageBuilder]) -> dict:
        """Make a call to the model and return the parsed response."""
        response = self._call_api(messages)
        return self._parse_response(response)

    @abstractmethod
    def _call_api(self, messages: list[dict | MessageBuilder]) -> Any:
        """Make a call to the model API and return the raw response."""
        pass

    @abstractmethod
    def _parse_response(self, response: Any) -> ResponseLLMOutput:
        """Parse the raw response from the model API and return a structured response."""
        pass


class OpenAIResponseModel(BaseResponseModel):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 100,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_kwargs=extra_kwargs,
        )
        self.client = OpenAI(api_key=api_key)

    def _call_api(self, messages: list[Any | MessageBuilder]) -> dict:
        input = []
        for msg in messages:
            if isinstance(msg, MessageBuilder):
                input += msg.to_openai()
            else:
                input.append(msg)
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=input,
                temperature=self.temperature,
                # previous_response_id=content.get("previous_response_id", None),
                max_output_tokens=self.max_tokens,
                **self.extra_kwargs,
                tool_choice="required",
                # reasoning={
                #     "effort": "low",
                #     "summary": "detailed",
                # },
            )
            return response
        except openai.OpenAIError as e:
            logging.error(f"Failed to get a response from the API: {e}")
            raise e

    def _parse_response(self, response: dict) -> dict:
        result = ResponseLLMOutput(
            raw_response=response,
            think="",
            action="noop()",
            last_computer_call_id=None,
            assistant_message=None,
        )
        for output in response.output:
            if output.type == "function_call":
                arguments = json.loads(output.arguments)
                result.action = (
                    f"{output.name}({", ".join([f"{k}={v}" for k, v in arguments.items()])})"
                )
                result.last_computer_call_id = output.call_id
                result.assistant_message = output
                break
            elif output.type == "reasoning":
                if len(output.summary) > 0:
                    result.think += output.summary[0].text + "\n"
        return result


class ClaudeResponseModel(BaseResponseModel):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 100,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_kwargs=extra_kwargs,
        )

        # Get pricing information

        try:
            pricing = tracking.get_pricing_anthropic()
            self.input_cost = float(pricing[model_name]["prompt"])
            self.output_cost = float(pricing[model_name]["completion"])
        except KeyError:
            logging.warning(
                f"Model {model_name} not found in the pricing information, prices are set to 0. Maybe try upgrading langchain_community."
            )
            self.input_cost = 0.0
            self.output_cost = 0.0

        self.client = Anthropic(api_key=api_key)

    def _call_api(self, messages: list[dict | MessageBuilder]) -> dict:
        input = []
        for msg in messages:
            if isinstance(msg, MessageBuilder):
                input += msg.to_anthropic()
            else:
                input.append(msg)
        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=input,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **self.extra_kwargs,
            )
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = input_tokens * self.input_cost + output_tokens * self.output_cost

            print(f"response.usage: {response.usage}")

            if hasattr(tracking.TRACKER, "instance") and isinstance(
                tracking.TRACKER.instance, tracking.LLMTracker
            ):
                tracking.TRACKER.instance(input_tokens, output_tokens, cost)

            return response
        except Exception as e:
            logging.error(f"Failed to get a response from the API: {e}")
            raise e

    def _parse_response(self, response: dict) -> dict:
        result = ResponseLLMOutput(
            raw_response=response,
            think="",
            action="noop()",
            last_computer_call_id=None,
            assistant_message={
                "role": "assistant",
                "content": response.content,
            },
        )
        for output in response.content:
            if output.type == "tool_use":
                result.action = f"{output.name}({', '.join([f'{k}=\"{v}\"' if isinstance(v, str) else f'{k}={v}' for k, v in output.input.items()])})"
                result.last_computer_call_id = output.id
            elif output.type == "text":
                result.think += output.text
        return result


def cua_response_to_text(action):
    """
    Given a computer action (e.g., click, double_click, scroll, etc.),
    convert it to a text description.
    """
    action_type = action.type

    try:
        match action_type:

            case "click":
                x, y = action.x, action.y
                button = action.button
                print(f"Action: click at ({x}, {y}) with button '{button}'")
                # Not handling things like middle click, etc.
                if button != "left" and button != "right":
                    button = "left"
                return f"mouse_click({x}, {y}, button='{button}')"

            case "scroll":
                x, y = action.x, action.y
                scroll_x, scroll_y = action.scroll_x, action.scroll_y
                print(
                    f"Action: scroll at ({x}, {y}) with offsets (scroll_x={scroll_x}, scroll_y={scroll_y})"
                )
                return f"mouse_move({x}, {y})\nscroll({scroll_x}, {scroll_y})"

            case "keypress":
                keys = action.keys
                for k in keys:
                    print(f"Action: keypress '{k}'")
                    # A simple mapping for common keys; expand as needed.
                    if k.lower() == "enter":
                        return "keyboard_press('Enter')"
                    elif k.lower() == "space":
                        return "keyboard_press(' ')"
                    else:
                        return f"keyboard_press('{k}')"

            case "type":
                text = action.text
                print(f"Action: type text: {text}")
                return f"keyboard_type('{text}')"

            case "wait":
                print(f"Action: wait")
                return "noop()"

            case "screenshot":
                # Nothing to do as screenshot is taken at each turn
                print(f"Action: screenshot")

            # Handle other actions here

            case "drag":
                x1, y1 = action.path[0].x, action.path[0].y
                x2, y2 = action.path[1].x, action.path[1].y
                print(f"Action: drag from ({x1}, {y1}) to ({x2}, {y2})")
                return f"mouse_drag_and_drop({x1}, {y1}, {x2}, {y2})"

            case _:
                print(f"Unrecognized action: {action}")

    except Exception as e:
        print(f"Error handling action {action}: {e}")


@dataclass
class OpenAIResponseModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenAI
    model."""

    api = "openai"

    def make_model(self, extra_kwargs=None):
        return OpenAIResponseModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            extra_kwargs=extra_kwargs,
        )


@dataclass
class ClaudeResponseModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenAI
    model."""

    api = "anthropic"

    def make_model(self, extra_kwargs=None):
        return ClaudeResponseModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            extra_kwargs=extra_kwargs,
        )
