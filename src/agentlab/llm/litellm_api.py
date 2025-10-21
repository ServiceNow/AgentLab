import json
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Type

import litellm
from litellm import completion
from openai.types.chat import ChatCompletion as OpenAIChatCompletion

from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.response_api import (
    AgentlabAction,
    APIPayload,
    BaseModelWithPricing,
    LLMOutput,
    Message,
    MessageBuilder,
    OpenAIChatCompletionAPIMessageBuilder,
    ToolCall,
    ToolCalls,
)

litellm.modify_params = True


class LiteLLMModel(BaseModelWithPricing):
    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float | None = None,
        max_tokens: int | None = 100,
        use_only_first_toolcall: bool = False,
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
        self.client = partial(completion, **client_args)
        self.init_pricing_tracker(pricing_api="litellm")
        self.use_only_first_toolcall = use_only_first_toolcall
        try:
            self.litellm_info = litellm.get_model_info(model_name)
            # maybe log this in xray

        except Exception as e:
            logging.error(f"Failed to get litellm model info: {e}")

    def _call_api(self, payload: APIPayload) -> "OpenAIChatCompletion":
        """
        Calls the LiteLLM API with the given payload.

        Args:
            payload (APIPayload): The payload to send to the API.

        Returns:
            OpenAIChatCompletion: An object with the same keys as OpenAIChatCompletion.
        """
        input = []
        for msg in payload.messages:  # type: ignore
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
            api_params["tools"] = (
                self.format_tools_for_chat_completion(payload.tools)
                if "function" not in payload.tools[0]  # convert if responses_api_tools
                else payload.tools
            )

        if payload.tool_choice is not None and payload.force_call_tool is None:
            api_params["tool_choice"] = (
                "required" if payload.tool_choice in ("required", "any") else payload.tool_choice
            )

        if payload.force_call_tool is not None:
            api_params["tool_choice"] = {
                "type": "function",
                "function": {"name": payload.force_call_tool},
            }

        if payload.reasoning_effort is not None:
            api_params["reasoning_effort"] = payload.reasoning_effort

        if "tools" in api_params and payload.cache_tool_definition:
            api_params["tools"][-1]["cache_control"] = {"type": "ephemeral"}  # type: ignore

        if payload.cache_complete_prompt:
            # Indicating cache control for the last message enables caching of the complete prompt.
            api_params["messages"][-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        response = self.client(**api_params, num_retries=5)

        return response  # type: ignore

    def _parse_response(self, response: "OpenAIChatCompletion") -> LLMOutput:
        think_output = self._extract_thinking_content_from_response(response)
        tool_calls = self._extract_tool_calls_from_response(response)

        if self.action_space_as_tools:
            env_action = self._extract_env_actions_from_toolcalls(tool_calls)  # type: ignore
        else:
            env_action = self._extract_env_actions_from_text_response(response)
        return LLMOutput(
            raw_response=response,
            think=think_output,
            action=env_action if env_action is not None else None,
            tool_calls=tool_calls if tool_calls is not None else None,
        )

    def _extract_thinking_content_from_response(
        self, response: OpenAIChatCompletion, wrap_tag="think"
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

    def _extract_tool_calls_from_response(self, response: OpenAIChatCompletion) -> ToolCalls | None:
        """Extracts tool calls from the response."""
        message = response.choices[0].message.to_dict()
        tool_calls = message.get("tool_calls", None)
        if tool_calls is None:
            return None
        tool_call_list = []
        for tc in tool_calls:  # type: ignore
            tool_call_list.append(
                ToolCall(
                    name=tc["function"]["name"],
                    arguments=json.loads(tc["function"]["arguments"]),
                    raw_call=tc,
                )
            )
            if self.use_only_first_toolcall:
                break
        return ToolCalls(tool_calls=tool_call_list, raw_calls=response)  # type: ignore

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
        self, response: "OpenAIChatCompletion"
    ) -> str | None:
        """Extracts environment actions from the text response."""
        # Use when action space is not given as tools.
        # TODO: Add support to pass action space as prompt in LiteLLM.
        # Check: https://docs.litellm.ai/docs/completion/function_call#function-calling-for-models-wout-function-calling-support
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


class LiteLLMAPIMessageBuilder(OpenAIChatCompletionAPIMessageBuilder):
    """Message builder for LiteLLM API, extending OpenAIChatCompletionAPIMessageBuilder."""

    def prepare_message(self, use_only_first_toolcall: bool = False) -> List[Message]:
        """Prepare the message for the OpenAI API."""
        content = []
        for item in self.content:
            content.append(self.convert_content_to_expected_format(item))
        output = [{"role": self.role, "content": content}]
        return output if self.role != "tool" else self.handle_tool_call(use_only_first_toolcall)

    def handle_tool_call(self, use_only_first_toolcall: bool = False) -> List[Message]:
        """Handle the tool call response from the last raw response."""
        if self.responded_tool_calls is None:
            raise ValueError("No tool calls found in responded_tool_calls")
        output = []
        raw_call = self.responded_tool_calls.raw_calls.choices[0].message  # type: ignore
        if use_only_first_toolcall:
            raw_call.tool_calls = raw_call.tool_calls[:1]
        output.append(raw_call)  # add raw calls to output
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


@dataclass
class LiteLLMModelArgs(BaseModelArgs):
    """Serializable arguments for LiteLMMModel."""

    api = "openai"  # tool description format used by actionset.to_tool_description() in bgym
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    use_only_first_toolcall: bool = False

    def make_model(self):
        return LiteLLMModel(
            model_name=self.model_name,
            base_url=self.base_url,
            api_key=self.api_key,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            use_only_first_toolcall=self.use_only_first_toolcall,
        )

    def get_message_builder(self) -> Type[MessageBuilder]:
        """Returns a message builder for the LiteLMMModel."""
        return LiteLLMAPIMessageBuilder


if __name__ == "__main__":
    """
    Some simple tests to run the LiteLLMModel with different models.
    """

    import os

    from agentlab.agents.tool_use_agent import DEFAULT_PROMPT_CONFIG, ToolUseAgentArgs
    from agentlab.experiments.study import Study
    from agentlab.llm.litellm_api import LiteLLMModelArgs

    os.environ["LITELLM_LOG"] = "WARNING"

    def get_agent(model_name: str) -> ToolUseAgentArgs:
        return ToolUseAgentArgs(
            model_args=LiteLLMModelArgs(
                model_name=model_name,
                max_new_tokens=2000,
                temperature=None,
            ),
            config=DEFAULT_PROMPT_CONFIG,
        )

    models = [
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini",
        "openai/gpt-4.1-nano",
        "openai/o3-2025-04-16",
        "anthropic/claude-3-7-sonnet-20250219",
        "anthropic/claude-sonnet-4-20250514",
        ## Add more models to test.
    ]
    agent_args = [get_agent(model) for model in models]

    study = Study(agent_args, "miniwob_tiny_test", logging_level_stdout=logging.WARNING)
    study.run(
        n_jobs=5,
        parallel_backend="ray",
        strict_reproducibility=False,
        n_relaunch=3,
    )
