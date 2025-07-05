import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

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


ContentItem = Dict[str, Any]
Message = Dict[str, Union[str, List[ContentItem]]]


@dataclass
class LLMOutput:
    """Serializable object for the output of a response LLM."""

    raw_response: Any = field(default_factory=dict)
    think: str = field(default="")
    action: str = field(default=None)  # Default action if no tool call is made
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
                parts.append(f"\n```\n{item['text']}\n```\n")
            elif "image" in item:
                parts.append(f"![Image]({item['image']})")

        markdown = f"### {self.role.capitalize()}\n"
        markdown += "\n".join(parts)

        return markdown

    def add_image_url(self, image_url: str) -> "MessageBuilder":
        """Add an image URL to the message content."""
        self.content.append({"image": image_to_png_base64_url(image_url)})
        return self

    def mark_all_previous_msg_for_caching(self):
        """Insert a cache breakpoint in the message content."""
        # This is a placeholder for future implementation.
        raise NotImplementedError


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

    def mark_all_previous_msg_for_caching(self) -> List[Message]:
        pass


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
        if self.role == "assistant":
            # Strip whitespace from assistant text responses. See anthropic error code 400.
            for c in output["content"]:
                if "text" in c:
                    c["text"] = c["text"].strip()
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

    def mark_all_previous_msg_for_caching(self) -> List[Message]:
        """Insert a cache breakpoint in the message content to mark all previous messages for caching."""
        self._cache_breakpoint = True


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
            action=None,
            tool_calls=None,
        )
        interesting_keys = ["output_text"]
        for output in response.output:
            if output.type == "function_call":
                arguments = json.loads(output.arguments)
                func_args_str = ", ".join(
                    [
                        f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
                        for k, v in arguments.items()
                    ]
                )
                result.action = f"{output.name}({func_args_str})"
                result.tool_calls = output
                break
            elif output.type == "reasoning":
                if len(output.summary) > 0:
                    result.think += output.summary[0].text + "\n"

            elif output.type == "message" and output.content:
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
            action=None,  # Default if no tool call
            tool_calls=None,
        )
        message = response.choices[0].message.to_dict()
        output.think = self.extract_content_with_reasoning(message)

        if tool_calls := message.get("tool_calls", None):
            for tool_call in tool_calls:
                function = tool_call["function"]
                arguments = json.loads(function["arguments"])
                func_args_str = ", ".join(
                    [
                        f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
                        for k, v in arguments.items()
                    ]
                )
                output.action = f"{function['name']}({func_args_str})"
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

        Args:
            tools: List of tool descriptions to format for Chat Completion API.

        Returns:
            Formatted tools list compatible with OpenAI Chat Completion API, or None if tools is None.
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
        It wraps the reasoning around <think>...</think> for easy identification of reasoning content,
        When LLM produces 'text' and 'reasoning' in the same message.
        Note: The wrapping of 'thinking' content may not be nedeed and may be reconsidered.

        Args:
            message: The message object or dict containing content and reasoning.
            wrap_tag: The tag name to wrap reasoning content (default: "think").

        Returns:
            str: The extracted content with reasoning wrapped in specified tags.
        """
        if not isinstance(message, dict):
            message = message.to_dict()

        reasoning_content = message.get("reasoning", None)
        msg_content = message.get("text", "")  # works for OR

        if reasoning_content:
            # Wrap reasoning in <think> tags with newlines for clarity
            reasoning_content = f"<{wrap_tag}>{reasoning_content}</{wrap_tag}>\n"
            logging.debug("Extracting content from response.choices[i].message.reasoning")
        else:
            reasoning_content = ""
        return f"{reasoning_content}{msg_content}{message.get('content', '')}"


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

    def _call_api(
        self, messages: list[dict | MessageBuilder], tool_choice="auto", **kwargs
    ) -> dict:
        input = []

        sys_msg, other_msgs = self.filter_system_messages(messages)
        sys_msg_text = "\n".join(c["text"] for m in sys_msg for c in m.content)
        for msg in other_msgs:
            temp = msg.prepare_message() if isinstance(msg, MessageBuilder) else [msg]
            if kwargs.pop("use_cache_breakpoints", False):
                temp = self.apply_cache_breakpoints(msg, temp)
            input.extend(temp)

        api_params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": input,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system": sys_msg_text,  # Anthropic API expects system message as a string
            "tool_choice": {"type": tool_choice},  # Tool choice for Claude API
            **self.extra_kwargs,  # Pass tools, tool_choice, etc. here
        }
        if self.tools is not None:
            api_params["tools"] = self.tools
        if kwargs.pop("cache_tool_definition", False):
            # Indicating cache control for the last tool enables caching of all previous tool definitions.
            api_params["tools"][-1]["cache_control"] = {"type": "ephemeral"}
        if kwargs.pop("cache_complete_prompt", False):
            # Indicating cache control for the last message enables caching of the complete prompt.
            api_params["messages"][-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
        if self.extra_kwargs.get("reasoning", None) is not None:
            api_params["reasoning"] = self.extra_kwargs["reasoning"]

        response = call_anthropic_api_with_retries(self.client.messages.create, api_params)

        return response

    @staticmethod
    def filter_system_messages(messages: list[dict | MessageBuilder]) -> tuple[MessageBuilder]:
        """Filter system messages from the list of messages."""
        # System message cannot have an image in the middle of the text sequences.
        # Images can be appended in the end of the system message.

        sys_msgs, other_msgs = [], []
        for msg in messages:
            if isinstance(msg, MessageBuilder) and msg.role == "system":
                sys_msgs.append(msg)
                for c in msg.content:
                    if c.get("type") == "image":
                        raise TypeError("System messages cannot contain images.")
            else:
                other_msgs.append(msg)
        return sys_msgs, other_msgs

    def _parse_response(self, response: dict) -> dict:
        result = LLMOutput(
            raw_response=response,
            think="",
            action=None,
            tool_calls={
                "role": "assistant",
                "content": response.content,
            },
        )
        for output in response.content:
            if output.type == "tool_use":
                func_args_str = ", ".join(
                    [
                        f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
                        for k, v in output.input.items()
                    ]
                )
                result.action = f"{output.name}({func_args_str})"
            elif output.type == "text":
                result.think += output.text
        return result

    # def ensure_cache_conditions(self, msgs: List[Message]) -> bool:
    #     """Ensure API specific cache conditions are met."""
    #     assert sum(getattr(msg, "_cache_breakpoint", 0) for msg in msgs) <= 4, "Too many cache breakpoints in the message."

    def apply_cache_breakpoints(self, msg: Message, prepared_msg: dict) -> List[Message]:
        """Apply cache breakpoints to the messages."""
        if getattr(msg, "_cache_breakpoint", False):
            prepared_msg[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
        return prepared_msg


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
