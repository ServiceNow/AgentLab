import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

import openai
from anthropic import Anthropic
from openai import OpenAI


from agentlab.llm import tracking

from .base_api import BaseModelArgs
from .llm_utils import (
    call_anthropic_api_with_retries,
    call_openai_api_with_retries,
    supports_tool_calling_for_openrouter,
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
        self.last_response: ResponseLLMOutput = None
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
    def tool(cls) -> "MessageBuilder":
        return cls("tool")

    def update_last_raw_response(self, raw_response: Any) -> "MessageBuilder":
        self.last_response = raw_response
        return self

    def add_tool_id(self, id: str) -> "MessageBuilder":
        self.tool_call_id = id
        return self

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

        markdown = f"## {self.role.capitalize()} Message\n\n"
        markdown += "\n\n---\n\n".join(parts)

        if self.role == "tool":
            assert self.tool_call_id is not None, "Tool call ID is required for tool messages"
            markdown += f"\n\n---\n\n**Tool Call ID:** `{self.tool_call_id}`"

        return markdown


class OpenAIResponseAPIMessageBuilder(MessageBuilder):

    def __init__(self, role: str):
        super().__init__(role)
        self.tool_call_id = None

    def add_tool_id(self, id: str) -> "MessageBuilder":
        self.tool_call_id = id
        return self

    def prepare_message(self) -> List[Message]:
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


class AnthropicAPIMessageBuilder(MessageBuilder):

    def __init__(self, role: str):
        super().__init__(role)
        self.tool_call_id = None

    def add_tool_id(self, id: str) -> "MessageBuilder":
        self.tool_call_id = id
        return self

    def prepare_message(self) -> List[Message]:
        content = []

        if self.role == "system":
            logging.info(
                "Treating system message as 'user'. In the Anthropic API, system messages should be passed as a direct input to the client."
            )
            return [{"role": "user", "content": content}]

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


class OpenAIChatCompletionAPIMessageBuilder(MessageBuilder):

    def __init__(self, role: str):
        super().__init__(role)
        self.tool_call_id = None
        self.tool_name = None
        self.last_response = None

    def update_tool_info(self, id: str) -> "MessageBuilder":
        self.tool_call_id = id
        return self

    def prepare_message(self) -> List[Message]:
        """Prepare the message for the OpenAI API."""
        content = []
        for item in self.content:
            if "text" in item:
                content.append({"type": "text", "text": item["text"]})
            elif "image" in item:
                content.append({"type": "image_url", "image_url": {"url": item["image"]}})
        res = [{"role": self.role, "content": content}]

        if self.role == "tool":
            assert self.tool_call_id is not None, "Tool call ID is required for tool messages"
            # tool messages can only take text with openai
            # we need to split the first content element if it's text and use it
            # then open a new (user) message with the rest
            # a function_call_output dict has keys "call_id", "type" and "output"
            res[0]["tool_call_id"] = self.tool_call_id
            res[0]["type"] = "function_call_output"
            message = self.last_response.raw_response.choices[0].message.to_dict()
            res[0]["tool_name"] = message["tool_calls"][0]["function"]["name"]
            text_content = (
                content.pop(0)["text"]
                if "text" in content[0]
                else "Tool call answer in next message"
            )
            res[0]["content"] = text_content
            res.append({"role": "user", "content": content})
        return res


# # Base class for all API Endpoints
class BaseResponseModel(ABC):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 100,
        extra_kwargs: Optional[Dict[str, Any]] = None,
        *args, **kwargs
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_kwargs = extra_kwargs or {}
        
        super().__init__(*args, **kwargs)

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


class BaseModelWithPricing(TrackAPIPricingMixin, BaseResponseModel):
    pass


# To Do: Add the call_with_tries in the openAI response model.

class OpenAIResponseModel(BaseModelWithPricing):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 100,
        extra_kwargs: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_kwargs=extra_kwargs,
            *args,
            **kwargs,
        )
        self.client = OpenAI(api_key=api_key)

    def _call_api(self, messages: list[Any | MessageBuilder]) -> dict:
        input = []
        for msg in messages:
            if isinstance(msg, MessageBuilder):
                input += msg.prepare_message()
            else:
                input.append(msg)
        
        api_params: Dict[str, Any] = {
            "model": self.model_name,
            "input": input,
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            **self.extra_kwargs,  # Pass tools, tool_choice, etc. here
        }
        if self.extra_kwargs.get("tool_choice", None) == "required":
            api_params["tool_choice"] = "required"
        if self.extra_kwargs.get("reasoning", None) is not None:
            api_params["reasoning"] = self.extra_kwargs["reasoning"]

        response = call_openai_api_with_retries(
            self.client.responses.create,
            api_params,
        )

        return response


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


class OpenAIChatCompletionModel(BaseModelWithPricing):
    def __init__(
        self,
        model_name: str,
        client_args: Optional[Dict[str, Any]] = {},
        temperature: float = 0.5,
        max_tokens: int = 100,
        extra_kwargs: Optional[Dict[str, Any]] = None,
        *args,**kwargs,
    ):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_kwargs=extra_kwargs,
            *args, **kwargs,
        )

        self.extra_kwargs["tools"] = self.format_tools_for_chat_completion(
            self.extra_kwargs.get("tools", [])
        )

        self.client = OpenAI(
            **client_args
        )  # Ensures client_args is a dict or defaults to an empty dict

    def _call_api(self, messages: list[dict | MessageBuilder]) -> openai.types.chat.ChatCompletion:
        chat_messages: List[Message] = []
        for msg in messages:
            if isinstance(msg, MessageBuilder):
                chat_messages.extend(msg.prepare_message())
            else:
                # Assuming msg is already in OpenAI Chat Completion message format
                chat_messages.append(msg)  # type: ignore

        api_params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": chat_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.extra_kwargs,  # Pass tools, tool_choice, etc. here
        }
        if self.extra_kwargs.get("tool_choice", None) == "required":
            api_params["tool_choice"] = "required"
        if self.extra_kwargs.get("reasoning", None) is not None:
            api_params["reasoning"] = self.extra_kwargs["reasoning"]

        response = call_openai_api_with_retries(self.client.chat.completions.create, api_params)

        return response

    def _parse_response(self, response: openai.types.chat.ChatCompletion) -> ResponseLLMOutput:

        output = ResponseLLMOutput(
            raw_response=response,
            think="",
            action="noop()",  # Default if no tool call
            last_computer_call_id=None,
            assistant_message={
                "role": "assistant",
                "content": response.choices[0].message.content,
            },
        )
        message = response.choices[0].message.to_dict()

        if tool_calls := message.get("tool_calls", None):
            for tool_call in tool_calls:
                function = tool_call["function"]
                arguments = json.loads(function["arguments"])
                output.action = (
                    f"{function['name']}({', '.join([f'{k}={v}' for k, v in arguments.items()])})"
                )
                output.last_computer_call_id = tool_call["id"]
                output.assistant_message = {
                    "role": "assistant",
                    "tool_calls": message["tool_calls"],
                }
                break  # only first tool call is used

        elif "content" in message and message["content"]:
            output.think = message["content"]

        return output

    @staticmethod
    def format_tools_for_chat_completion(tools_flat):
        """Formats response tools format for OpenAI Chat Completion API.
        Why we need this?
        Ans: actionset.to_tool_description() in bgym only returns description
        format valid for OpenAI Response API.
        """
        return [
            {
                "type": tool["type"],
                "function": {k: tool[k] for k in ("name", "description", "parameters")},
            }
            for tool in tools_flat
        ]


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
        *args, **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_kwargs=extra_kwargs,
            *args,
            **kwargs,
        )

        self.client = Anthropic(api_key=api_key)

    def _call_api(self, messages: list[dict | MessageBuilder]) -> dict:
        input = []
        for msg in messages:
            if isinstance(msg, MessageBuilder):
                input += msg.prepare_message()
            else:
                input.append(msg)

        api_params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": input,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.extra_kwargs,  # Pass tools, tool_choice, etc. here
        }
        if self.extra_kwargs.get("tool_choice", None) == "required":
            api_params["tool_choice"] = "required"
        if self.extra_kwargs.get("reasoning", None) is not None:
            api_params["reasoning"] = self.extra_kwargs["reasoning"]

        response = call_anthropic_api_with_retries(self.client.messages.create, api_params)

        return response

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


# Factory classes to create the appropriate model based on the API endpoint.
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
            pricing_api="openai",
        )

    def get_message_builder(self) -> MessageBuilder:
        return OpenAIResponseAPIMessageBuilder


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
            pricing_api="anthropic",
        )

    def get_message_builder(self) -> MessageBuilder:
        return AnthropicAPIMessageBuilder


@dataclass
class OpenAIChatModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenAI
    model."""

    api = "openai"

    def make_model(self, extra_kwargs=None):
        return OpenAIChatCompletionModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            extra_kwargs=extra_kwargs,
            pricing_api="openai",
        )

    def get_message_builder(self) -> MessageBuilder:
        return OpenAIChatCompletionAPIMessageBuilder


@dataclass
class OpenRouterModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenRouter
    model."""

    api: str = "openai"  # tool description format used by actionset.to_tool_description() in bgym

    def make_model(self, extra_kwargs=None):
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
        )

    def get_message_builder(self) -> MessageBuilder:
        return OpenAIChatCompletionAPIMessageBuilder

    def __post_init__(self):
        # Some runtime checks
        assert supports_tool_calling_for_openrouter(
            self.model_name
        ), f"Model {self.model_name} does not support tool calling."


class VLLMModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with a VLLM
    model."""

    api = "openai"  # tool description format used by actionset.to_tool_description() in bgym

    def __post_init__(self):
        # error handeling
        assert self.is_model_available(
            self.model_name
        ), f"Model {self.model_name} is not available on the VLLM server. \
                Please check the model name or server configuration."

    def make_model(self, extra_kwargs=None):
        return OpenAIChatCompletionModel(
            client_args={
                "base_url": "http://localhost:8000/v1",
                "api_key": os.getenv("VLLM_API_KEY", "EMPTY"),
            },
            model_name=self.model_name,  # this needs to be set
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            extra_kwargs=extra_kwargs,
        )

    def get_message_builder(self) -> MessageBuilder:
        return OpenAIChatCompletionAPIMessageBuilder

    ## Some Tests for VLLM server in the works!
    def test_vllm_server_reachability(self):
        import requests

        try:
            response = requests.get(
                f"{self.client_args['base_url']}/v1/models",
                headers={"Authorization": f"Bearer {self.client_args['api_key']}"},
            )
            if response.status_code == 200:
                return True
            else:
                return False
        except requests.RequestException as e:
            logging.error(f"Error checking VLLM server reachability: {e}")
            return False

    def is_model_available(self, model_name: str) -> bool:
        # import requests

        # """Check if the model is available on the VLLM server."""
        # if not self.test_vllm_server_reachability():
        #     logging.error("VLLM server is not reachable.")
        #     return False
        # try:
        #     response = requests.get(
        #         f"{self.client_args['base_url']}/v1/models",
        #         headers={"Authorization": f"Bearer {self.client_args['api_key']}"},
        #     )
        #     if response.status_code == 200:
        #         models = response.json().get("data", [])
        #         return any(model.get("id") == model_name for model in models)
        #     else:
        #         logging.error(
        #             f"Failed to fetch vllm hosted models: {response.status_code} - {response.text}"
        #         )
        #         return False
        # except requests.RequestException as e:
        #     logging.error(f"Error checking model availability: {e}")
        #     return False
        return True
