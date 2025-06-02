import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

import openai
from anthropic import Anthropic
from openai import OpenAI

from agentlab.llm.llm_utils import image_to_png_base64_url

from .base_api import BaseModelArgs
from .llm_utils import (
    call_anthropic_api_with_retries,
    call_openai_api_with_retries,
)
from .tracking import TrackAPIPricingMixin

"""This module contains utlity classes for building input messages and interacting with LLM APIs. 
It includes:
    1. Message Builder for building input messages
    2. Base Reponse class for different LLM APIs (OpenAI, Anthropic, etc.)
    3. Factory classes (inherits from BaseModelArgs) for creating instances of LLM Response models.
"""


type ContentItem = Dict[str, Any]
type Message = Dict[str, Union[str, List[ContentItem]]]


@dataclass
class LLMOutput:
    """Serializable object for the output of a response LLM."""

    raw_response: Any = field(default_factory=dict)
    think: str = field(default="")
    action: str = field(default="noop()")  # Default action if no tool call is made
    tool_calls: Any = field(default=None)  # This will hold the tool call response if any


class MessageBuilder:
    def __init__(self, role: str):

        self.role = role
        self.last_raw_response: LLMOutput = None
        self.content: List[ContentItem] = []
        self.tool_call_id: Optional[str] = None

    @classmethod
    def system(cls) -> "MessageBuilder":
        return cls("system")

    @classmethod
    def user(cls) -> "MessageBuilder":
        return cls("user")

    @classmethod
    def assistant(cls) -> "MessageBuilder":
        return cls("assistant")

    @classmethod
    def tool(cls, last_raw_response) -> "MessageBuilder":
        return cls("tool").update_last_raw_response(last_raw_response)

    @abstractmethod
    def prepare_message(self) -> List[Message]:
        """Prepare the message for the API call."""
        raise NotImplementedError("Subclasses must implement this method.")

    def update_last_raw_response(self, last_raw_response: Any) -> "MessageBuilder":
        self.last_raw_response = last_raw_response
        return self

    # def add_tool_id(self, id: str) -> "MessageBuilder":
    #     self.tool_call_id = id
    #     return self

    def add_text(self, text: str) -> "MessageBuilder":
        self.content.append({"text": text})
        return self

    def add_image(self, image: str) -> "MessageBuilder":
        self.content.append({"image": image})
        return self

    def to_markdown(self) -> str:
        parts = []
        for item in self.content:
            if "text" in item:
                parts.append(item["text"])
            elif "image" in item:
                parts.append(f"![Image]({item['image']})")

        markdown = f"### {self.role.capitalize()}\n"
        markdown += "\n\n---\n\n".join(parts)

        # if self.role == "tool":
        #     assert self.tool_call_id is not None, "Tool call ID is required for tool messages"
        #     markdown += f"\n\n---\n\n**Tool Call ID:** `{self.tool_call_id}`"

        return markdown

    def add_image_url(self, image_url: str) -> "MessageBuilder":
        """Add an image URL to the message content."""
        self.content.append({"image": image_to_png_base64_url(image_url)})
        return self


# TODO: Support parallel tool calls.


class OpenAIResponseAPIMessageBuilder(MessageBuilder):
    @classmethod
    def system(cls) -> "OpenAIResponseAPIMessageBuilder":
        # OpenAI Responses API uses 'developer' role for system messages
        return cls("developer")

    def prepare_message(self) -> List[Message]:
        content = []
        for item in self.content:
            if "text" in item:
                content_type = "input_text" if self.role != "assistant" else "output_text"
                content.append({"type": content_type, "text": item["text"]})

            elif "image" in item:
                content.append({"type": "input_image", "image_url": item["image"]})

        output = [{"role": self.role, "content": content}]
        if self.role != "tool":
            return output
        else:
            tool_call_response = self.handle_tool_call(content)
            return tool_call_response

    def handle_tool_call(self, content):
        """Handle the tool call response from the last raw response."""
        output = []
        head_content, *tail_content = content
        api_response = self.last_raw_response
        fn_calls = [content for content in api_response.output if content.type == "function_call"]
        assert len(fn_calls) > 0, "No function calls found in the last response"
        if len(fn_calls) > 1:
            logging.warning("Using only the first tool call from many.")

        first_fn_call_id = fn_calls[0].call_id
        fn_output = head_content.get("text", "Function call answer in next message")
        fn_call_response = {
            "type": "function_call_output",
            "call_id": first_fn_call_id,
            "output": fn_output,
        }
        output.append(fn_call_response)
        if tail_content:
            # if there are more content items, add them as a new user message
            output.append({"role": "user", "content": tail_content})
        return output


class AnthropicAPIMessageBuilder(MessageBuilder):

    def prepare_message(self) -> List[Message]:
        content = [self.transform_content(item) for item in self.content]
        output = {"role": self.role, "content": content}

        if self.role == "system":
            logging.info(
                "Treating system message as 'user'. In the Anthropic API, system messages should be passed as a direct input to the client."
            )
            output["role"] = "user"

        if self.role == "tool":
            # assert self.tool_call_id is not None, "Tool call ID is required for tool messages"
            api_response = self.last_raw_response
            fn_calls = [content for content in api_response.content if content.type == "tool_use"]
            assert len(fn_calls) > 0, "No tool calls found in the last response"
            if len(fn_calls) > 1:
                logging.warning("Using only the first tool call from many.")
            tool_call_id = fn_calls[0].id  # Using the first tool call ID

            output["role"] = "user"
            output["content"] = [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": output["content"],
                }
            ]
        return [output]

    def transform_content(self, content: ContentItem) -> ContentItem:
        """Transform content item to the format expected by Anthropic API."""
        if "text" in content:
            return {"type": "text", "text": content["text"]}
        elif "image" in content:
            img_str: str = content["image"]
            # make sure to get rid of the image type for anthropic
            # e.g. "data:image/png;base64"
            if img_str.startswith("data:image/png;base64,"):
                img_str = img_str[len("data:image/png;base64,") :]
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_str,
                },
            }
        else:
            raise ValueError(f"Unsupported content type: {content}")


class OpenAIChatCompletionAPIMessageBuilder(MessageBuilder):

    def prepare_message(self) -> List[Message]:
        """Prepare the message for the OpenAI API."""
        content = [self.transform_content(item) for item in self.content]
        if self.role == "tool":
            return self.handle_tool_call(content)
        else:
            return [{"role": self.role, "content": content}]

    def transform_content(self, content: ContentItem) -> ContentItem:
        """Transform content item to the format expected by OpenAI ChatCompletion."""
        if "text" in content:
            return {"type": "text", "text": content["text"]}
        elif "image" in content:
            return {"type": "image_url", "image_url": {"url": content["image"]}}
        else:
            raise ValueError(f"Unsupported content type: {content}")

    def handle_tool_call(self, content) -> List[Message]:
        """Handle the tool call response from the last raw response."""
        output = []
        content_head, *content_tail = content
        api_response = self.last_raw_response.choices[0].message
        fn_calls = getattr(api_response, "tool_calls", None)
        assert fn_calls is not None, "Tool calls not found in the last response"
        if len(fn_calls) > 1:
            logging.warning("Using only the first tool call from many.")

        # a function_call_output dict has keys "role", "tool_call_id" and "content"
        tool_call_reponse = {
            "role": "tool",
            "tool_call_id": fn_calls[0].id,  # using the first tool call ID
            "content": content_head.get("text", "Tool call answer in next message"),
            "name": fn_calls[0].function.name,  # required with OpenRouter
        }

        output.append(tool_call_reponse)
        if content_tail:
            # if there are more content items, add them as a new user message
            output.append({"role": "user", "content": content_tail})
        return output


# # Base class for all API Endpoints
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

        super().__init__()

    def __call__(self, messages: list[dict | MessageBuilder], **kwargs) -> dict:
        """Make a call to the model and return the parsed response."""
        response = self._call_api(messages, **kwargs)
        return self._parse_response(response)

    @abstractmethod
    def _call_api(self, messages: list[dict | MessageBuilder], **kwargs) -> Any:
        """Make a call to the model API and return the raw response."""
        pass

    @abstractmethod
    def _parse_response(self, response: Any) -> LLMOutput:
        """Parse the raw response from the model API and return a structured response."""
        pass


class BaseModelWithPricing(TrackAPIPricingMixin, BaseResponseModel):
    pass


class OpenAIResponseModel(BaseModelWithPricing):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 100,
        extra_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.tools = kwargs.pop("tools", None)
        self.tool_choice = kwargs.pop("tool_choice", None)
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_kwargs=extra_kwargs,
            **kwargs,
        )
        self.client = OpenAI(api_key=api_key)

    def _call_api(self, messages: list[Any | MessageBuilder], **kwargs) -> dict:
        input = []
        for msg in messages:
            input.extend(msg.prepare_message() if isinstance(msg, MessageBuilder) else [msg])

        api_params: Dict[str, Any] = {
            "model": self.model_name,
            "input": input,
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            **self.extra_kwargs,
        }

        if self.tools is not None:
            api_params["tools"] = self.tools
        if self.tool_choice is not None:
            api_params["tool_choice"] = self.tool_choice

        # api_params |= kwargs  # Merge any additional parameters passed
        response = call_openai_api_with_retries(
            self.client.responses.create,
            api_params,
        )

        return response

    def _parse_response(self, response: dict) -> dict:
        result = LLMOutput(
            raw_response=response,
            think="",
            action="noop()",
            tool_calls=None,
        )
        interesting_keys = ["output_text"]
        for output in response.output:
            if output.type == "function_call":
                arguments = json.loads(output.arguments)
                result.action = (
                    # f"{output.name}({", ".join([f"{k}={v}" for k, v in arguments.items()])})"
                    f"{output.name}({', '.join([f'{k}=\"{v}\"' if isinstance(v, str) else f'{k}={v}' for k, v in arguments.items()])})"
                )
                result.tool_calls = output
                break
            elif output.type == "reasoning":
                if len(output.summary) > 0:
                    result.think += output.summary[0].text + "\n"

            elif output.type == "message" and output.content:  # Why did i add a 'message' here?
                result.think += output.content[0].text + "\n"
        for key in interesting_keys:
            if key_content := getattr(output, "output_text", None) is not None:
                result.think += f"<{key}>{key_content}</{key}>"
        return result


class OpenAIChatCompletionModel(BaseModelWithPricing):
    def __init__(
        self,
        model_name: str,
        client_args: Optional[Dict[str, Any]] = {},
        temperature: float = 0.5,
        max_tokens: int = 100,
        extra_kwargs: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):

        self.tools = self.format_tools_for_chat_completion(kwargs.pop("tools", None))
        self.tool_choice = kwargs.pop("tool_choice", None)

        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_kwargs=extra_kwargs,
            *args,
            **kwargs,
        )

        self.client = OpenAI(
            **client_args
        )  # Ensures client_args is a dict or defaults to an empty dict

    def _call_api(self, messages: list[dict | MessageBuilder]) -> openai.types.chat.ChatCompletion:
        input = []
        for msg in messages:
            input.extend(msg.prepare_message() if isinstance(msg, MessageBuilder) else [msg])
        api_params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": input,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.extra_kwargs,  # Pass tools, tool_choice, etc. here
        }
        if self.tools is not None:
            api_params["tools"] = self.tools
        if self.tool_choice is not None:
            api_params["tool_choice"] = self.tool_choice

        response = call_openai_api_with_retries(self.client.chat.completions.create, api_params)

        return response

    def _parse_response(self, response: openai.types.chat.ChatCompletion) -> LLMOutput:

        output = LLMOutput(
            raw_response=response,
            think="",
            action="noop()",  # Default if no tool call
            tool_calls=None,
        )
        message = response.choices[0].message.to_dict()
        output.think = self.extract_content_with_reasoning(message)

        if tool_calls := message.get("tool_calls", None):
            for tool_call in tool_calls:
                function = tool_call["function"]
                arguments = json.loads(function["arguments"])
                output.action = f"{function['name']}({', '.join([f'{k}=\"{v}\"' if isinstance(v, str) else f'{k}={v}' for k, v in arguments.items()])})"
                output.tool_calls = {
                    "role": "assistant",
                    "tool_calls": [message["tool_calls"][0]],  # Use only the first tool call
                }
                break
        return output

    @staticmethod
    def format_tools_for_chat_completion(tools):
        """Formats response tools format for OpenAI Chat Completion API.
        Why we need this?
        Ans: actionset.to_tool_description() in bgym only returns description
        format valid for OpenAI Response API.
        """
        formatted_tools = None
        if tools is not None:
            formatted_tools = [
                {
                    "type": tool["type"],
                    "function": {k: tool[k] for k in ("name", "description", "parameters")},
                }
                for tool in tools
            ]
        return formatted_tools

    @staticmethod
    def extract_content_with_reasoning(message, wrap_tag="think"):
        """Extracts the content from the message, including reasoning if available.
        It wraps the reasoning around <think>...</think> for backward compatibility."""
        if not isinstance(message, dict):
            message = message.to_dict()

        reasoning_content = message.get("reasoning", None)
        msg_content = message.get("text", "")  # works for OR

        if reasoning_content:
            # Wrap reasoning in <think> tags with newlines for clarity
            reasoning_content = f"<{wrap_tag}>{reasoning_content}</{wrap_tag}>\n"  # why I do need to enclose reasoning in <think> tags?
            logging.debug("Extracting content from response.choices[i].message.reasoning")
        else:
            reasoning_content = ""
        return f"{reasoning_content}{msg_content}{message.get('content', '')}"


# To Do: Double check the expected action format by browsergym.
# openai action output do not have parenthesis but the antropic action parsing does.
# Confirm with allac if this is the expected format.


class ClaudeResponseModel(BaseModelWithPricing):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 100,
        extra_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.tools = kwargs.pop("tools", None)
        self.tool_choice = kwargs.pop("tool_choice", None)

        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_kwargs=extra_kwargs,
            **kwargs,
        )

        self.client = Anthropic(api_key=api_key)

    def _call_api(self, messages: list[dict | MessageBuilder]) -> dict:
        input = []
        for msg in messages:
            input.extend(msg.prepare_message() if isinstance(msg, MessageBuilder) else [msg])

        api_params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": input,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.extra_kwargs,  # Pass tools, tool_choice, etc. here
        }
        if self.tools is not None:
            api_params["tools"] = self.tools
        if self.extra_kwargs.get("reasoning", None) is not None:
            api_params["reasoning"] = self.extra_kwargs["reasoning"]

        response = call_anthropic_api_with_retries(self.client.messages.create, api_params)

        return response

    def _parse_response(self, response: dict) -> dict:
        result = LLMOutput(
            raw_response=response,
            think="",
            action="noop()",
            tool_calls={
                "role": "assistant",
                "content": response.content,
            },
        )
        for output in response.content:
            if output.type == "tool_use":
                result.action = f"{output.name}({', '.join([f'{k}=\"{v}\"' if isinstance(v, str) else f'{k}={v}' for k, v in output.input.items()])})"
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


# Factory classes to create the appropriate model based on the API endpoint.
@dataclass
class OpenAIResponseModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenAI
    model."""

    api = "openai"

    def make_model(self, extra_kwargs=None, **kwargs):
        return OpenAIResponseModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            extra_kwargs=extra_kwargs,
            pricing_api="openai",
            **kwargs,
        )

    def get_message_builder(self) -> MessageBuilder:
        return OpenAIResponseAPIMessageBuilder


@dataclass
class ClaudeResponseModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenAI
    model."""

    api = "anthropic"

    def make_model(self, extra_kwargs=None, **kwargs):
        return ClaudeResponseModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            extra_kwargs=extra_kwargs,
            pricing_api="anthropic",
            **kwargs,
        )

    def get_message_builder(self) -> MessageBuilder:
        return AnthropicAPIMessageBuilder


@dataclass
class OpenAIChatModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenAI
    model."""

    api = "openai"

    def make_model(self, extra_kwargs=None, **kwargs):
        return OpenAIChatCompletionModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            extra_kwargs=extra_kwargs,
            pricing_api="openai",
            **kwargs,
        )

    def get_message_builder(self) -> MessageBuilder:
        return OpenAIChatCompletionAPIMessageBuilder


@dataclass
class OpenRouterModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenRouter
    model."""

    api: str = "openai"  # tool description format used by actionset.to_tool_description() in bgym

    def make_model(self, extra_kwargs=None, **kwargs):
        return OpenAIChatCompletionModel(
            client_args={
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": os.getenv("OPENROUTER_API_KEY"),
            },
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            extra_kwargs=extra_kwargs,
            pricing_api="openrouter",
            **kwargs,
        )

    def get_message_builder(self) -> MessageBuilder:
        return OpenAIChatCompletionAPIMessageBuilder


class VLLMModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with a VLLM
    model."""

    api = "openai"  # tool description format used by actionset.to_tool_description() in bgym

    def make_model(self, extra_kwargs=None, **kwargs):
        return OpenAIChatCompletionModel(
            client_args={
                "base_url": "http://localhost:8000/v1",
                "api_key": os.getenv("VLLM_API_KEY", "EMPTY"),
            },
            model_name=self.model_name,  # this needs to be set
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            extra_kwargs=extra_kwargs,
            pricing_api="vllm",
            **kwargs,
        )

    def get_message_builder(self) -> MessageBuilder:
        return OpenAIChatCompletionAPIMessageBuilder
