import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

import openai
from anthropic import Anthropic
from anthropic.types import Completion
from anthropic.types import Message as AnthrophicMessage
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

logger = logging.getLogger(__name__)

ContentItem = Dict[str, Any]
Message = Dict[str, Union[str, List[ContentItem]]]


@dataclass
class ToolCall:
    """Represents a tool call made by the LLM.
    Attributes:
    name: Name of the tool called.
    arguments: Arguments passed to the tool.
    raw_call: The raw call object from the LLM API.
    tool_response: Output of the tool call goes here. It can be only one content item.
    """

    name: str = field(default=None)
    arguments: Dict[str, Any] = field(default_factory=dict)
    raw_call: Any = field(default=None)
    tool_response: ContentItem = None

    @property
    def is_response_set(self) -> bool:
        """Check if the tool response is set."""
        return self.tool_response is not None

    def response_text(self, text: str) -> "MessageBuilder":
        self.tool_response = {"text": text}
        return self

    def response_image(self, image: str) -> "MessageBuilder":
        self.tool_response = {"image": image}
        return self

    def __repr__(self):
        return f"ToolCall(name={self.name}, arguments={self.arguments})"


@dataclass
class ToolCalls:
    """A collection of tool calls made by the LLM.

    Attributes:
    tool_calls: List of ToolCall objects.
    raw_calls: Represents raw tool calls object returned by a LLM API, may contain one or more tool calls.
    """

    tool_calls: List[ToolCall] = field(default_factory=list)
    raw_calls: List[Any] = field(default_factory=list)

    def add_tool_call(self, tool_call: ToolCall) -> "ToolCalls":
        self.tool_calls.append(tool_call)
        return self

    @property
    def all_responses_set(self) -> bool:
        """Check if all tool calls have responses set."""
        return all(call.is_response_set for call in self.tool_calls)

    def __len__(self) -> int:
        """Return the number of tool calls."""
        return len(self.tool_calls)

    def __iter__(self):
        """Make ToolCalls iterable."""
        return iter(self.tool_calls)

    def __bool__(self):
        """Check if there are any tool calls."""
        return len(self.tool_calls) > 0


@dataclass
class LLMOutput:
    """Serializable object for the output of a response LLM."""

    raw_response: Any = field(default=None)
    think: str = field(default="")
    action: str | None = field(default=None)  # Default action if no tool call is made
    tool_calls: ToolCalls | None = field(
        default=None
    )  # This will hold the tool call response if any


class MessageBuilder:
    def __init__(self, role: str):
        self.role = role
        self.content: List[ContentItem] = []
        self.responded_tool_calls: ToolCalls = None

    @classmethod
    def system(cls) -> "MessageBuilder":
        return cls("system")

    @classmethod
    def user(cls) -> "MessageBuilder":
        return cls("user")

    @classmethod
    def assistant(cls) -> "MessageBuilder":
        return cls("assistant")

    @abstractmethod
    def prepare_message(self) -> List[Message]:
        """Prepare the message for the API call."""
        raise NotImplementedError("Subclasses must implement this method.")

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

        # Tool call markdown repr
        if self.responded_tool_calls is not None:
            for i, tool_call in enumerate(self.responded_tool_calls.tool_calls, 1):
                parts.append(
                    f"\n**Tool Call {i}**: {tool_call_to_python_code(tool_call.name, tool_call.arguments)}"
                )
                response = tool_call.tool_response
                if response is not None:
                    parts.append(f"\n**Tool Response {i}:**")
                    content = (
                        f"```\n{response['text']}\n```"
                        if "text" in response
                        else f"![Tool Response Image]({response['image']})"
                    )
                    parts.append(content)

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

    @classmethod
    def add_responded_tool_calls(cls, responded_tool_calls: ToolCalls) -> "MessageBuilder":
        """Add tool calls to the message content."""
        assert responded_tool_calls.all_responses_set, "All tool calls must have a response."
        msg = cls("tool")
        msg.responded_tool_calls = responded_tool_calls
        return msg


class OpenAIResponseAPIMessageBuilder(MessageBuilder):
    @classmethod
    def system(cls) -> "OpenAIResponseAPIMessageBuilder":
        # OpenAI Responses API uses 'developer' role for system messages
        return cls("developer")

    def prepare_message(self) -> List[Message]:
        content = []
        for item in self.content:
            content.append(self.convert_content_to_expected_format(item))
        output = [{"role": self.role, "content": content}]

        return output if self.role != "tool" else self.handle_tool_call()

    def convert_content_to_expected_format(self, content: ContentItem) -> ContentItem:
        """Convert the content item to the expected format for OpenAI Responses."""
        if "text" in content:
            content_type = "input_text" if self.role != "assistant" else "output_text"
            return {"type": content_type, "text": content["text"]}
        elif "image" in content:
            return {"type": "input_image", "image_url": content["image"]}
        else:
            raise ValueError(f"Unsupported content type: {content}")

    def handle_tool_call(self) -> List[Message]:
        """Handle the tool call response from the last raw response."""
        if self.responded_tool_calls is None:
            raise ValueError("No tool calls found in responded_tool_calls")

        output = []
        output.extend(self.responded_tool_calls.raw_calls.output)  # this contains response
        for fn_call in self.responded_tool_calls:
            call_type = fn_call.raw_call.type
            call_id = fn_call.raw_call.call_id
            call_response = fn_call.tool_response

            match call_type:
                case "function_call":
                    # image output is not supported in function calls response.
                    assert (
                        "image" not in call_response
                    ), "Image output is not supported in function calls response."
                    fn_call_response = {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": self.convert_content_to_expected_format(call_response)["text"],
                    }
                    output.append(fn_call_response)

                case "computer_call":
                    # For computer calls, use only images are expected.
                    assert (
                        "text" not in call_response
                    ), "Text output is not supported in computer calls response."
                    computer_call_output = {
                        "type": "computer_call_output",
                        "call_id": call_id,
                        "output": self.convert_content_to_expected_format(call_response),
                    }
                    output.append(computer_call_output)  # this needs to be a screenshot

        return output

    def mark_all_previous_msg_for_caching(self):
        """Nothing special to do here for openAI. They do not have a notion of cache breakpoints."""
        pass


class AnthropicAPIMessageBuilder(MessageBuilder):
    def prepare_message(self) -> List[Message]:
        content = [self.transform_content(item) for item in self.content]
        output = {"role": self.role, "content": content}

        if self.role == "tool":
            return self.handle_tool_call()

        if self.role == "assistant":
            # Strip whitespace from assistant text responses. See anthropic error code 400.
            for c in output["content"]:
                if "text" in c:
                    c["text"] = c["text"].strip()
        return [output]

    def handle_tool_call(self) -> List[Message]:
        """Handle the tool call response from the last raw response."""
        if self.responded_tool_calls is None:
            raise ValueError("No tool calls found in responded_tool_calls")

        llm_tool_call = {
            "role": "assistant",
            "content": self.responded_tool_calls.raw_calls.content,
        }  # Add the toolcall block
        tool_response = {"role": "user", "content": []}  # Anthropic expects a list of messages
        for call in self.responded_tool_calls:
            assert (
                "image" not in call.tool_response
            ), "Image output is not supported in tool calls response."
            tool_response["content"].append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.raw_call.id,
                    "content": self.transform_content(call.tool_response)[
                        "text"
                    ],  # needs to be str
                }
            )

        return [llm_tool_call, tool_response]

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
        content = []
        for item in self.content:
            content.append(self.convert_content_to_expected_format(item))
        output = [{"role": self.role, "content": content}]
        return output if self.role != "tool" else self.handle_tool_call()

    def convert_content_to_expected_format(self, content: ContentItem) -> ContentItem:
        """Transform content item to the format expected by OpenAI ChatCompletion."""
        if "text" in content:
            return {"type": "text", "text": content["text"]}
        elif "image" in content:
            return {"type": "image_url", "image_url": {"url": content["image"]}}
        else:
            raise ValueError(f"Unsupported content type: {content}")

    def handle_tool_call(self) -> List[Message]:
        """Handle the tool call response from the last raw response."""
        if self.responded_tool_calls is None:
            raise ValueError("No tool calls found in responded_tool_calls")
        output = []
        output.append(
            self.responded_tool_calls.raw_calls.choices[0].message
        )  # add raw calls to output
        for fn_call in self.responded_tool_calls:
            raw_call = fn_call.raw_call
            assert (
                "image" not in fn_call.tool_response
            ), "Image output is not supported in function calls response."
            # a function_call_output dict has keys "role", "tool_call_id" and "content"
            tool_call_reponse = {
                "name": raw_call["function"]["name"],  # required with OpenRouter
                "role": "tool",
                "tool_call_id": raw_call["id"],
                "content": self.convert_content_to_expected_format(fn_call.tool_response)["text"],
            }
            output.append(tool_call_reponse)

        return output

    def mark_all_previous_msg_for_caching(self):
        """Nothing special to do here for openAI. They do not have a notion of cache breakpoints."""
        pass


@dataclass
class APIPayload:
    messages: List[MessageBuilder] | None = None
    tools: List[Dict[str, Any]] | None = None
    tool_choice: Literal["none", "auto", "any", "required"] | None = None
    force_call_tool: str | None = (
        None  # Name of the tool to call # If set, will force the LLM to call this tool.
    )
    use_cache_breakpoints: bool = (
        False  # If True, will apply cache breakpoints to the messages. # applicable for Anthropic
    )
    cache_tool_definition: bool = (
        False  # If True, will cache the tool definition in the last message.
    )
    cache_complete_prompt: bool = (
        False  # If True, will cache the complete prompt in the last message.
    )
    reasoning_effort: Literal["low", "medium", "high"] | None = None

    def __post_init__(self):
        if self.tool_choice and self.force_call_tool:
            raise ValueError("tool_choice and force_call_tool are mutually exclusive")
        if self.reasoning_effort is not None:
            logger.info(
                "In agentlab reasoning_effort is used by LiteLLM API only. We will eventually shift to LiteLLM API for all LLMs."
            )


# # Base class for all API Endpoints
class BaseResponseModel(ABC):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        super().__init__()

    def __call__(self, payload: APIPayload) -> LLMOutput:
        """Make a call to the model and return the parsed response."""
        response = self._call_api(payload)
        return self._parse_response(response)

    @abstractmethod
    def _call_api(self, payload: APIPayload) -> Any:
        """Make a call to the model API and return the raw response."""
        pass

    @abstractmethod
    def _parse_response(self, response: Any) -> LLMOutput:
        """Parse the raw response from the model API and return a structured response."""
        pass


class AgentlabAction:
    """
    Collection of utility function to convert tool calls to Agentlab action format.
    """

    @staticmethod
    def convert_toolcall_to_agentlab_action_format(toolcall: ToolCall) -> str:
        """Convert a tool call to an Agentlab environment action string.

        Args:
            toolcall: ToolCall object containing the name and arguments of the tool call.

        Returns:
            A string representing the action in Agentlab format i.e. python function call string.
        """

        tool_name, tool_args = toolcall.name, toolcall.arguments
        return tool_call_to_python_code(tool_name, tool_args)

    @staticmethod
    def convert_multiactions_to_agentlab_action_format(actions: list[str]) -> str | None:
        """Convert multiple actions list to a format that env supports.

        Args:
            actions: List of action strings to be joined.

        Returns:
            Joined actions separated by newlines, or None if empty.
        """
        return "\n".join(actions) if actions else None


class BaseModelWithPricing(TrackAPIPricingMixin, BaseResponseModel):
    pass


class OpenAIResponseModel(BaseModelWithPricing):
    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float | None = None,
        max_tokens: int | None = 100,
    ):
        self.action_space_as_tools = True  # this should be a config
        super().__init__(  # This is passed to BaseModel
            model_name=model_name, api_key=api_key, temperature=temperature, max_tokens=max_tokens
        )
        client_args = {}
        if base_url is not None:
            client_args["base_url"] = base_url
        if api_key is not None:
            client_args["api_key"] = api_key
        self.client = OpenAI(**client_args)
        # Init pricing tracker after super() so that all attributes have been set.
        self.init_pricing_tracker(pricing_api="openai")  # Use the PricingMixin

    def _call_api(self, payload: APIPayload) -> "OpenAIResponseObject":

        input = []
        for msg in payload.messages:
            input.extend(msg.prepare_message())
        api_params: Dict[str, Any] = {
            "model": self.model_name,
            "input": input,
        }
        # Not all Open AI models support these parameters (example: o3), so we check if they are set.
        if self.temperature is not None:
            api_params["temperature"] = self.temperature
        if self.max_tokens is not None:
            api_params["max_output_tokens"] = self.max_tokens
        if payload.tools is not None:
            api_params["tools"] = payload.tools
        if payload.tool_choice is not None and payload.force_call_tool is None:
            api_params["tool_choice"] = (
                "required" if payload.tool_choice in ("required", "any") else payload.tool_choice
            )
        if payload.force_call_tool is not None:
            api_params["tool_choice"] = {"type": "function", "name": payload.force_call_tool}

        response = call_openai_api_with_retries(
            self.client.responses.create,
            api_params,
        )

        return response

    def _parse_response(self, response: "OpenAIResponseObject") -> LLMOutput:
        """Parse the raw response from the OpenAI Responses API."""

        think_output = self._extract_thinking_content_from_response(response)
        toolcalls = self._extract_tool_calls_from_response(response)

        if self.action_space_as_tools:
            env_action = self._extract_env_actions_from_toolcalls(toolcalls)
        else:
            env_action = self._extract_env_actions_from_text_response(response)

        return LLMOutput(
            raw_response=response,
            think=think_output,
            action=env_action if env_action is not None else None,
            tool_calls=toolcalls if toolcalls is not None else None,
        )

    def _extract_tool_calls_from_response(self, response: "OpenAIResponseObject") -> ToolCalls:
        """Extracts tool calls from the response."""
        tool_calls = []
        for output in response.output:
            if output.type == "function_call":
                tool_name = output.name
                tool_args = json.loads(output.arguments)
            elif output.type == "computer_call":
                tool_name, tool_args = self.cua_action_to_env_tool_name_and_args(output.action)
            else:
                continue
            tool_calls.append(ToolCall(name=tool_name, arguments=tool_args, raw_call=output))

        return ToolCalls(tool_calls=tool_calls, raw_calls=response)

    def _extract_env_actions_from_toolcalls(self, toolcalls: ToolCalls) -> Any | None:
        """Extracts actions from the response."""
        if not toolcalls:
            return None

        actions = [
            AgentlabAction.convert_toolcall_to_agentlab_action_format(call) for call in toolcalls
        ]
        actions = (
            AgentlabAction.convert_multiactions_to_agentlab_action_format(actions)
            if len(actions) > 1
            else actions[0]
        )
        return actions

    def _extract_thinking_content_from_response(self, response: "OpenAIResponseObject") -> str:
        """Extracts the thinking content from the response."""
        thinking_content = ""
        for output in response.output:
            if output.type == "reasoning":
                if len(output.summary) > 0:
                    thinking_content += output.summary[0].text + "\n"
            elif output.type == "message" and output.content:
                thinking_content += output.content[0].text + "\n"
            elif hasattr(output, "output_text") and output.output_text:
                thinking_content += f"{output.output_text}\n"
        return thinking_content

    def cua_action_to_env_tool_name_and_args(self, action: str) -> tuple[str, Dict[str, Any]]:
        """ "Overwrite this method to convert a computer action to agentlab action string"""
        raise NotImplementedError(
            "This method should be implemented in the subclass to convert a computer action to agentlab action string."
        )

    def _extract_env_actions_from_text_response(
        self, response: "OpenAIResponseObject"
    ) -> str | None:
        """Extracts environment actions from the text response."""
        # Use when action space is not given as tools.
        pass


class OpenAIChatCompletionModel(BaseModelWithPricing):
    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float | None = None,
        max_tokens: int | None = 100,
    ):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.action_space_as_tools = True  # this should be a config
        client_args = {}
        if base_url is not None:
            client_args["base_url"] = base_url
        if api_key is not None:
            client_args["api_key"] = api_key
        self.client = OpenAI(**client_args)
        self.init_pricing_tracker(pricing_api="openai")  # Use the PricingMixin

    def _call_api(self, payload: APIPayload) -> "openai.types.chat.ChatCompletion":
        input = []
        for msg in payload.messages:
            input.extend(msg.prepare_message())
        api_params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": input,
        }
        if self.temperature is not None:
            api_params["temperature"] = self.temperature

        if self.max_tokens is not None:
            api_params["max_completion_tokens"] = self.max_tokens

        if payload.tools is not None:
            # tools format is OpenAI Response API format.
            api_params["tools"] = self.format_tools_for_chat_completion(payload.tools)

        if payload.tool_choice is not None and payload.force_call_tool is None:
            api_params["tool_choice"] = (
                "required" if payload.tool_choice in ("required", "any") else payload.tool_choice
            )

        if payload.force_call_tool is not None:
            api_params["tool_choice"] = {
                "type": "function",
                "function": {"name": payload.force_call_tool},
            }

        response = call_openai_api_with_retries(self.client.chat.completions.create, api_params)

        return response

    def _parse_response(self, response: "openai.types.chat.ChatCompletion") -> LLMOutput:
        think_output = self._extract_thinking_content_from_response(response)
        tool_calls = self._extract_tool_calls_from_response(response)

        if self.action_space_as_tools:
            env_action = self._extract_env_actions_from_toolcalls(tool_calls)
        else:
            env_action = self._extract_env_actions_from_text_response(response)
        return LLMOutput(
            raw_response=response,
            think=think_output,
            action=env_action if env_action is not None else None,
            tool_calls=tool_calls if tool_calls is not None else None,
        )

    def _extract_thinking_content_from_response(
        self, response: openai.types.chat.ChatCompletion, wrap_tag="think"
    ):
        """Extracts the content from the message, including reasoning if available.
        It wraps the reasoning around <think>...</think> for easy identification of reasoning content,
        When LLM produces 'text' and 'reasoning' in the same message.
        Note: The wrapping of 'thinking' content may not be nedeed and may be reconsidered.

        Args:
            response: The message object or dict containing content and reasoning.
            wrap_tag: The tag name to wrap reasoning content (default: "think").

        Returns:
            str: The extracted content with reasoning wrapped in specified tags.
        """
        message = response.choices[0].message
        if not isinstance(message, dict):
            message = message.to_dict()

        reasoning_content = message.get("reasoning", None)
        msg_content = message.get("text", "")  # works for Open-router
        if reasoning_content:
            # Wrap reasoning in <think> tags with newlines for clarity
            reasoning_content = f"<{wrap_tag}>{reasoning_content}</{wrap_tag}>\n"
            logging.debug("Extracting content from response.choices[i].message.reasoning")
        else:
            reasoning_content = ""
        return f"{reasoning_content}{msg_content}{message.get('content', '')}"

    def _extract_tool_calls_from_response(
        self, response: openai.types.chat.ChatCompletion
    ) -> ToolCalls | None:
        """Extracts tool calls from the response."""
        message = response.choices[0].message.to_dict()
        tool_calls = message.get("tool_calls", None)
        if tool_calls is None:
            return None
        tool_call_list = []
        for tc in tool_calls:
            tool_call_list.append(
                ToolCall(
                    name=tc["function"]["name"],
                    arguments=json.loads(tc["function"]["arguments"]),
                    raw_call=tc,
                )
            )
        return ToolCalls(tool_calls=tool_call_list, raw_calls=response)

    def _extract_env_actions_from_toolcalls(self, toolcalls: ToolCalls) -> Any | None:
        """Extracts actions from the response."""
        if not toolcalls:
            return None

        actions = [
            AgentlabAction.convert_toolcall_to_agentlab_action_format(call) for call in toolcalls
        ]
        actions = (
            AgentlabAction.convert_multiactions_to_agentlab_action_format(actions)
            if len(actions) > 1
            else actions[0]
        )
        return actions

    def _extract_env_actions_from_text_response(
        self, response: "openai.types.chat.ChatCompletion"
    ) -> str | None:
        """Extracts environment actions from the text response."""
        # Use when action space is not given as tools.
        pass

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


class ClaudeResponseModel(BaseModelWithPricing):
    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float | None = None,
        max_tokens: int | None = 100,
    ):
        self.action_space_as_tools = True  # this should be a config

        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        client_args = {}
        if base_url is not None:
            client_args["base_url"] = base_url
        if api_key is not None:
            client_args["api_key"] = api_key
        self.client = Anthropic(**client_args)
        self.init_pricing_tracker(pricing_api="anthropic")  # Use the PricingMixin

    def _call_api(self, payload: APIPayload) -> Completion:
        sys_msg, other_msgs = self.filter_system_messages(payload.messages)
        sys_msg_text = "\n".join(c["text"] for m in sys_msg for c in m.content)
        input = []
        for msg in other_msgs:
            temp = msg.prepare_message()
            if payload.use_cache_breakpoints:
                temp = self.apply_cache_breakpoints(msg, temp)
            input.extend(temp)

        api_params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": input,
            "system": sys_msg_text,
        }  # Anthropic API expects system message as a string

        if self.temperature is not None:
            api_params["temperature"] = self.temperature
        if self.max_tokens is not None:
            api_params["max_tokens"] = self.max_tokens

        if payload.tools is not None:
            api_params["tools"] = payload.tools
        if payload.tool_choice is not None and payload.force_call_tool is None:
            api_params["tool_choice"] = (
                {"type": "any"}
                if payload.tool_choice in ("required", "any")
                else {"type": payload.tool_choice}
            )
        if payload.force_call_tool is not None:
            api_params["tool_choice"] = {"type": "tool", "name": payload.force_call_tool}
        if payload.cache_tool_definition:
            # Indicating cache control for the last message enables caching of the last message.
            api_params["tools"][-1]["cache_control"] = {"type": "ephemeral"}
        if payload.cache_complete_prompt:
            # Indicating cache control for the last message enables caching of the complete prompt.
            api_params["messages"][-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}

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

    def _parse_response(self, response: "AnthrophicMessage") -> LLMOutput:

        toolcalls = self._extract_tool_calls_from_response(response)
        think_output = self._extract_thinking_content_from_response(response)
        if self.action_space_as_tools:
            env_action = self._extract_env_actions_from_toolcalls(toolcalls)
        else:
            env_action = self._extract_env_actions_from_text_response(response)
        return LLMOutput(
            raw_response=response,
            think=think_output,
            action=env_action if env_action is not None else None,
            tool_calls=toolcalls if toolcalls is not None else None,
        )

    def _extract_tool_calls_from_response(self, response: "AnthrophicMessage") -> ToolCalls:
        """Extracts tool calls from the response."""
        tool_calls = []
        for output in response.content:
            if output.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        name=output.name,
                        arguments=output.input,
                        raw_call=output,
                    )
                )
        return ToolCalls(tool_calls=tool_calls, raw_calls=response)

    def _extract_thinking_content_from_response(self, response: "AnthrophicMessage"):
        """Extracts the thinking content from the response."""
        return "".join(output.text for output in response.content if output.type == "text")

    def _extract_env_actions_from_toolcalls(self, toolcalls: ToolCalls) -> Any | None:
        """Extracts actions from the response."""
        if not toolcalls:
            return None

        actions = [
            AgentlabAction.convert_toolcall_to_agentlab_action_format(call) for call in toolcalls
        ]
        actions = (
            AgentlabAction.convert_multiactions_to_agentlab_action_format(actions)
            if len(actions) > 1
            else actions[0]
        )
        return actions

    def _extract_env_actions_from_text_response(self, response: "AnthrophicMessage") -> str | None:
        """Extracts environment actions from the text response."""
        # Use when action space is not given as tools.
        pass

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

    def make_model(self):
        return OpenAIResponseModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )

    def get_message_builder(self) -> MessageBuilder:
        return OpenAIResponseAPIMessageBuilder


@dataclass
class ClaudeResponseModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenAI
    model."""

    api = "anthropic"

    def make_model(self):
        return ClaudeResponseModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )

    def get_message_builder(self) -> MessageBuilder:
        return AnthropicAPIMessageBuilder


@dataclass
class OpenAIChatModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenAI
    model."""

    api = "openai"

    def make_model(self):
        return OpenAIChatCompletionModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )

    def get_message_builder(self) -> MessageBuilder:
        return OpenAIChatCompletionAPIMessageBuilder


@dataclass
class OpenRouterModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenRouter
    model."""

    api: str = "openai"  # tool description format used by actionset.to_tool_description() in bgym

    def make_model(self):
        return OpenAIChatCompletionModel(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )

    def get_message_builder(self) -> MessageBuilder:
        return OpenAIChatCompletionAPIMessageBuilder


def tool_call_to_python_code(func_name, kwargs):
    """Format a function name and kwargs dict into a Python function call string."""
    if kwargs is None:
        kwargs = {}

    if not kwargs:
        return f"{func_name}()"

    args_str = ", ".join(f"{key}={repr(value)}" for key, value in kwargs.items())
    return f"{func_name}({args_str})"


# ___Not__Tested__#

# class VLLMModelArgs(BaseModelArgs):
#     """Serializable object for instantiating a generic chat model with a VLLM
#     model."""

#     api = "openai"  # tool description format used by actionset.to_tool_description() in bgym

#     def make_model(self, extra_kwargs=None, **kwargs):
#         return OpenAIChatCompletionModel(
#             client_args={
#                 "base_url": "http://localhost:8000/v1",
#                 "api_key": os.getenv("VLLM_API_KEY", "EMPTY"),
#             },
#             model_name=self.model_name,  # this needs to be set
#             temperature=self.temperature,
#             max_tokens=self.max_new_tokens,
#             extra_kwargs=extra_kwargs,
#             pricing_api="vllm",
#             **kwargs,
#         )

#     def get_message_builder(self) -> MessageBuilder:
#         return OpenAIChatCompletionAPIMessageBuilder
