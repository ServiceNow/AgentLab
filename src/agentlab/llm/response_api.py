import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Type
import openai
from anthropic import Anthropic
from openai import OpenAI

from agentlab.llm import tracking

from .base_api import BaseModelArgs

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
                "Treating system message as 'user'. In the Anthropic API, system messages should be passed as a direct input to the client." \
                
            )
            return [{"role": 'user', "content": content}]

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
                content.append({"type": "image_url", 
                                "image_url": {"url": item["image"]}})
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
            res[0]["tool_name"] = message['tool_calls'][0]['function']['name']
            text_content = (
                content.pop(0)["text"]
                if "text" in content[0]
                else "Tool call answer in next message"
            )
            res[0]["content"] = text_content
            res.append({"role": "user", "content": content})
        return res

class OpenRouterAPIMessageBuilder(MessageBuilder):

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
                input += msg.prepare_message()
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

class OpenAIChatCompletionModel(BaseResponseModel):
    def __init__(
        self,
        model_name: str,
        client_args: Optional[Dict[str, Any]] = {},
        temperature: float = 0.5,
        max_tokens: int = 100,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_kwargs=extra_kwargs,
        )
        self.extra_kwargs['tools'] = self.format_tools_for_chat_completion(
                                            self.extra_kwargs.get('tools', []))
        self.client = OpenAI(**client_args)  # Ensures client_args is a dict or defaults to an empty dict
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
            "tool_choice":"auto",
            **self.extra_kwargs,  # Pass tools, tool_choice, etc. here
        }

        response = self.call_with_retries(self.client.chat.completions.create, api_params)
            # Basic token tracking (if usage information is available)
        if response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            # Cost calculation would require pricing data
            # cost = ...
            # if hasattr(tracking.TRACKER, "instance") and isinstance(
            #     tracking.TRACKER.instance, tracking.LLMTracker
            # ):
            #     tracking.TRACKER.instance(input_tokens, output_tokens, cost) # Placeholder for cost

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
                output.assistant_message = {"role": "assistant",
                                             "tool_calls": message["tool_calls"]}
                break # only first tool call is used

        elif "content" in message and message["content"]:
            output.think = message["content"]

        return output

    @staticmethod
    def format_tools_for_chat_completion(tools_flat):
        """Formats response tools format for OpenAI Chat Completion API."""
        return [
            {
                "type": tool["type"],
                "function": {k: tool[k] for k in ("name", "description", "parameters")},
            }
            for tool in tools_flat
        ]

    @staticmethod
    def call_with_retries(client_function, api_params, max_retries=5):
        """
        Makes a API call with retries for transient failures,
        rate limiting, and invalid or error-containing responses.

        Args:
            client_function (Callable): Function to call the API (e.g., openai.ChatCompletion.create).
            api_params (dict): Parameters to pass to the API function.
            max_retries (int): Maximum number of retry attempts.

        Returns:
            response: Valid API response object.
        """
        for attempt in range(1, max_retries + 1):
            try:
                response = client_function(**api_params)

                # Check for explicit error field in response object
                if getattr(response, "error", None):
                    logging.warning(f"[Attempt {attempt}] API returned error: {response.error}. Retrying...")
                    continue

                # Check for valid response with choices
                if hasattr(response, "choices") and response.choices:
                    logging.info(f"[Attempt {attempt}] API call succeeded.")
                    return response

                logging.warning(f"[Attempt {attempt}] API returned empty or malformed response. Retrying...")

            except openai.APIError as e:
                logging.error(f"[Attempt {attempt}] APIError: {e}")
                if e.http_status == 429:
                    logging.warning("Rate limit exceeded. Retrying...")
                elif e.http_status >= 500:
                    logging.warning("Server error encountered. Retrying...")
                else:
                    logging.error("Non-retriable API error occurred.")
                    raise

            except Exception as e:
                logging.exception(f"[Attempt {attempt}] Unexpected exception occurred.")
                raise

        logging.error("Exceeded maximum retry attempts. API call failed.")
        raise RuntimeError("API call failed after maximum retries.")


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
                input += msg.prepare_message()
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
            action=None,
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
        )

    def get_message_builder(self) -> MessageBuilder:
        return OpenAIChatCompletionAPIMessageBuilder


@dataclass
class OpenRouterModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenRouter
    model."""

    api: str = "openai" # tool description format used by actionset.to_tool_description() in bgym
    open_router_args: Dict = field(default_factory=dict)

    def make_model(self, extra_kwargs=None):
        import os

        extra_kwargs = self.open_router_args.copy()
        if extra_kwargs:
            extra_kwargs.update(extra_kwargs)

        return OpenAIChatCompletionModel(
            client_args={
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": os.getenv("OPENROUTER_API_KEY"),
            },
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            extra_kwargs= extra_kwargs,
        )
    
    def get_message_builder(self) -> MessageBuilder:
        return OpenRouterAPIMessageBuilder
