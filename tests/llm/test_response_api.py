import os
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import anthropic
import openai
import pytest

from agentlab.llm import tracking
from agentlab.llm.response_api import (
    AnthropicAPIMessageBuilder,
    ClaudeResponseModelArgs,
    LLMOutput,
    OpenAIChatCompletionAPIMessageBuilder,
    OpenAIChatModelArgs,
    OpenAIResponseAPIMessageBuilder,
    OpenAIResponseModelArgs,
)


# Helper to create a mock OpenAI ChatCompletion response
def create_mock_openai_chat_completion(
    content=None, tool_calls=None, prompt_tokens=10, completion_tokens=20
):
    completion = MagicMock(spec=openai.types.chat.ChatCompletion)
    choice = MagicMock()
    message = MagicMock(spec=openai.types.chat.ChatCompletionMessage)
    message.content = content
    message.tool_calls = None
    if tool_calls:
        message.tool_calls = []
        for tc in tool_calls:
            tool_call_mock = MagicMock(
                spec=openai.types.chat.chat_completion_message_tool_call.ChatCompletionMessageToolCall
            )
            tool_call_mock.id = tc["id"]
            tool_call_mock.type = tc["type"]
            tool_call_mock.function = MagicMock(
                spec=openai.types.chat.chat_completion_message_tool_call.Function
            )
            tool_call_mock.function.name = tc["function"]["name"]
            tool_call_mock.function.arguments = tc["function"]["arguments"]
            message.tool_calls.append(tool_call_mock)

    choice.message = message
    completion.choices = [choice]

    completion.usage = MagicMock()
    # Explicitly set the attributes that get_tokens_counts_from_response will try first.
    # These are the generic names.
    completion.usage.input_tokens = prompt_tokens
    completion.usage.output_tokens = completion_tokens

    # Also set the OpenAI-specific names if any other part of the code might look for them directly,
    # or if get_tokens_counts_from_response had different fallback logic.
    completion.usage.prompt_tokens = prompt_tokens
    completion.usage.completion_tokens = completion_tokens

    completion.model_dump.return_value = {
        "id": "chatcmpl-xxxx",
        "choices": [
            {"message": {"role": "assistant", "content": content, "tool_calls": tool_calls}}
        ],
        # Ensure the usage dict in model_dump also reflects the token counts accurately.
        # The get_tokens_counts_from_response also has a path for dict style.
        "usage": {
            "input_tokens": prompt_tokens,  # Generic name
            "output_tokens": completion_tokens,  # Generic name
            "prompt_tokens": prompt_tokens,  # OpenAI specific
            "completion_tokens": completion_tokens,  # OpenAI specific
        },
    }
    message.to_dict.return_value = {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
    }
    return completion


# Helper to create a mock Anthropic response
def create_mock_anthropic_response(
    text_content=None, tool_use=None, input_tokens=15, output_tokens=25
):

    response = MagicMock(spec=anthropic.types.Message)
    response.type = "message"  # Explicitly set the type attribute
    response.content = []
    response.content = []
    if text_content:
        text_block = MagicMock(spec=anthropic.types.TextBlock)
        text_block.type = "text"
        text_block.text = text_content
        response.content.append(text_block)
    if tool_use:
        tool_use_block = MagicMock(spec=anthropic.types.ToolUseBlock)
        tool_use_block.type = "tool_use"
        tool_use_block.id = tool_use["id"]
        tool_use_block.name = tool_use["name"]
        tool_use_block.input = tool_use["input"]
        response.content.append(tool_use_block)
    response.usage = MagicMock()
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    return response


def create_mock_openai_responses_api_response(
    outputs: Optional[List[Dict[str, Any]]] = None, input_tokens: int = 10, output_tokens: int = 20
) -> MagicMock:
    """
    Helper to create a mock response object similar to what
    openai.resources.Responses.create() would return.
    Compatible with OpenAIResponseModel and TrackAPIPricingMixin.
    """

    response_mock = MagicMock(openai.types.responses.response)
    response_mock.type = "response"
    response_mock.output = []

    if outputs:
        for out_data in outputs:
            output_item_mock = MagicMock()
            output_item_mock.type = out_data.get("type")

            if output_item_mock.type == "function_call":
                # You can adapt this depending on your expected object structure
                output_item_mock.name = out_data.get("name")
                output_item_mock.arguments = out_data.get("arguments")
                output_item_mock.call_id = out_data.get("call_id")
            elif output_item_mock.type == "reasoning":
                output_item_mock.summary = []
                for text_content in out_data.get("summary", []):
                    summary_text_mock = MagicMock()
                    summary_text_mock.text = text_content
                    output_item_mock.summary.append(summary_text_mock)

            response_mock.output.append(output_item_mock)

    # Token usage for pricing tracking
    response_mock.usage = MagicMock()
    response_mock.usage.input_tokens = input_tokens
    response_mock.usage.output_tokens = output_tokens
    response_mock.usage.prompt_tokens = input_tokens
    response_mock.usage.completion_tokens = output_tokens

    return response_mock


# --- Test MessageBuilders ---


def test_openai_response_api_message_builder_text():
    builder = OpenAIResponseAPIMessageBuilder.user()
    builder.add_text("Hello, world!")
    messages = builder.prepare_message()
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [{"type": "input_text", "text": "Hello, world!"}]


def test_openai_response_api_message_builder_image():
    builder = OpenAIResponseAPIMessageBuilder.user()
    builder.add_image("data:image/png;base64,SIMPLEBASE64STRING")
    messages = builder.prepare_message()
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [
        {"type": "input_image", "image_url": "data:image/png;base64,SIMPLEBASE64STRING"}
    ]


def test_anthropic_api_message_builder_text():
    builder = AnthropicAPIMessageBuilder.user()
    builder.add_text("Hello, Anthropic!")
    messages = builder.prepare_message()
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [{"type": "text", "text": "Hello, Anthropic!"}]


def test_anthropic_api_message_builder_image():
    builder = AnthropicAPIMessageBuilder.user()
    builder.add_image("data:image/png;base64,ANTHROPICBASE64")
    messages = builder.prepare_message()
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert len(messages[0]["content"]) == 1
    image_content = messages[0]["content"][0]
    assert image_content["type"] == "image"
    assert image_content["source"]["type"] == "base64"
    assert image_content["source"]["media_type"] == "image/png"
    assert image_content["source"]["data"] == "ANTHROPICBASE64"  # Base64 prefix should be stripped


def test_openai_chat_completion_api_message_builder_text():
    builder = OpenAIChatCompletionAPIMessageBuilder.user()
    builder.add_text("Hello, ChatCompletion!")
    # Mock last_response as it's used by tool role
    builder.last_raw_response = MagicMock(spec=LLMOutput)
    builder.last_raw_response.raw_response = MagicMock()
    builder.last_raw_response.raw_response.choices = [MagicMock()]
    builder.last_raw_response.raw_response.choices[0].message.to_dict.return_value = {
        "tool_calls": [{"function": {"name": "some_function"}}]
    }
    messages = builder.prepare_message()

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [{"type": "text", "text": "Hello, ChatCompletion!"}]


def test_openai_chat_completion_api_message_builder_image():
    builder = OpenAIChatCompletionAPIMessageBuilder.user()
    builder.add_image("data:image/jpeg;base64,CHATCOMPLETIONBASE64")
    # Mock last_response
    builder.last_raw_response = MagicMock(spec=LLMOutput)
    builder.last_raw_response.raw_response = MagicMock()
    builder.last_raw_response.raw_response.choices = [MagicMock()]
    builder.last_raw_response.raw_response.choices[0].message.to_dict.return_value = {
        "tool_calls": [{"function": {"name": "some_function"}}]
    }
    messages = builder.prepare_message()

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,CHATCOMPLETIONBASE64"}}
    ]


def test_openai_chat_completion_model_parse_and_cost():
    args = OpenAIChatModelArgs(model_name="gpt-3.5-turbo")  # A cheap model for testing
    # Mock the OpenAI client to avoid needing OPENAI_API_KEY
    with patch("agentlab.llm.response_api.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        model = args.make_model()

    # Mock the API call
    mock_response = create_mock_openai_chat_completion(
        content="This is a test thought.",
        tool_calls=[
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
            }
        ],
        prompt_tokens=50,
        completion_tokens=30,
    )

    with patch.object(
        model.client.chat.completions, "create", return_value=mock_response
    ) as mock_create:
        with tracking.set_tracker() as global_tracker:  # Use your global tracker
            messages = [
                OpenAIChatCompletionAPIMessageBuilder.user()
                .add_text("What's the weather in Paris?")
                .prepare_message()[0]
            ]
            parsed_output = model(messages)

    mock_create.assert_called_once()
    assert parsed_output.raw_response.choices[0].message.content == "This is a test thought."
    assert parsed_output.action == 'get_weather(location="Paris")'
    assert parsed_output.raw_response.choices[0].message.tool_calls[0].id == "call_123"
    # Check cost tracking (token counts)
    assert global_tracker.stats["input_tokens"] == 50
    assert global_tracker.stats["output_tokens"] == 30
    assert global_tracker.stats["cost"] > 0


def test_claude_response_model_parse_and_cost():
    args = ClaudeResponseModelArgs(model_name="claude-3-haiku-20240307")  # A cheap model
    model = args.make_model()

    mock_anthropic_api_response = create_mock_anthropic_response(
        text_content="Thinking about the request.",
        tool_use={"id": "tool_abc", "name": "search_web", "input": {"query": "latest news"}},
        input_tokens=40,
        output_tokens=20,
    )

    with patch.object(
        model.client.messages, "create", return_value=mock_anthropic_api_response
    ) as mock_create:
        with tracking.set_tracker() as global_tracker:
            messages = [
                AnthropicAPIMessageBuilder.user()
                .add_text("Search for latest news")
                .prepare_message()[0]
            ]
            parsed_output = model(messages)

    mock_create.assert_called_once()
    fn_calls = [
        content for content in parsed_output.raw_response.content if content.type == "tool_use"
    ]
    assert "Thinking about the request." in parsed_output.think
    assert parsed_output.action == 'search_web(query="latest news")'
    assert fn_calls[0].id == "tool_abc"
    assert global_tracker.stats["input_tokens"] == 40
    assert global_tracker.stats["output_tokens"] == 20
    # assert global_tracker.stats["cost"] > 0 # Verify cost is calculated


def test_openai_response_model_parse_and_cost():
    """
    Tests OpenAIResponseModel output parsing and cost tracking with both
    function_call and reasoning outputs.
    """
    args = OpenAIResponseModelArgs(model_name="gpt-4.1")
    
    # Mock outputs
    mock_function_call_output = {
        "type": "function_call",
        "name": "get_current_weather",
        "arguments": '{"location": "Boston, MA", "unit": "celsius"}',
        "call_id": "call_abc123",
    }

    mock_api_resp = create_mock_openai_responses_api_response(
        outputs=[mock_function_call_output],
        input_tokens=70,
        output_tokens=40,
    )
    
    # Mock the OpenAI client to avoid needing OPENAI_API_KEY
    with patch('agentlab.llm.response_api.OpenAI') as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        model = args.make_model()

    with patch.object(
        model.client.responses, "create", return_value=mock_api_resp
    ) as mock_create_method:
        with tracking.set_tracker() as global_tracker:
            messages = [
                OpenAIResponseAPIMessageBuilder.user()
                .add_text("What's the weather in Boston?")
                .prepare_message()[0]
            ]
            parsed_output = model(messages)

    mock_create_method.assert_called_once()
    fn_calls = [
        content for content in parsed_output.raw_response.output if content.type == "function_call"
    ]
    assert parsed_output.action == 'get_current_weather(location="Boston, MA", unit="celsius")'
    assert fn_calls[0].call_id == "call_abc123"
    assert parsed_output.raw_response == mock_api_resp
    assert global_tracker.stats["input_tokens"] == 70
    assert global_tracker.stats["output_tokens"] == 40


# --- Test Response Models (Pricy - require API keys and actual calls) ---


@pytest.mark.pricy
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_chat_completion_model_pricy_call():
    """Tests OpenAIChatCompletionModel with a real API call."""
    args = OpenAIChatModelArgs(
        model_name="gpt-4.1",
        temperature=1e-5,
        max_new_tokens=100,
    )

    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather in a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the weather for.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature.",
                    },
                },
                "required": ["location"],
            },
        }
    ]

    model = args.make_model(tools=tools, tool_choice="required")

    with tracking.set_tracker() as global_tracker:
        messages = [
            OpenAIChatCompletionAPIMessageBuilder.user()
            .add_text("What is the weather in Paris?")
            .prepare_message()[0]
        ]
        parsed_output = model(messages)

    assert parsed_output.raw_response is not None
    assert (
        parsed_output.action == 'get_weather(location="Paris")'
    ), f""" Expected get_weather(location="Paris") but got {parsed_output.action}"""
    assert global_tracker.stats["input_tokens"] > 0
    assert global_tracker.stats["output_tokens"] > 0
    assert global_tracker.stats["cost"] > 0


@pytest.mark.pricy
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_claude_response_model_pricy_call():
    """Tests ClaudeResponseModel with a real API call."""

    args = ClaudeResponseModelArgs(
        model_name="claude-3-haiku-20240307",
        temperature=1e-5,
        max_new_tokens=100,
    )
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the weather for.",
                    },
                },
                "required": ["location"],
            },
        }
    ]
    model = args.make_model(tools=tools)

    with tracking.set_tracker() as global_tracker:
        messages = [
            AnthropicAPIMessageBuilder.user()
            .add_text("What is the weather in Paris?")
            .prepare_message()[0]
        ]
        parsed_output = model(messages)

    assert parsed_output.raw_response is not None
    assert (
        parsed_output.action == 'get_weather(location="Paris")'
    ), f'Expected get_weather("Paris") but got {parsed_output.action}'
    assert global_tracker.stats["input_tokens"] > 0
    assert global_tracker.stats["output_tokens"] > 0
    assert global_tracker.stats["cost"] > 0


@pytest.mark.pricy
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_response_model_pricy_call():
    """
    Tests OpenAIResponseModel output parsing and cost tracking with both
    function_call and reasoning outputs.
    """
    args = OpenAIResponseModelArgs(model_name="gpt-4.1", temperature=1e-5, max_new_tokens=100)

    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather in a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the weather for.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature.",
                    },
                },
                "required": ["location"],
            },
        }
    ]
    model = args.make_model(tools=tools)

    with tracking.set_tracker() as global_tracker:
        messages = [
            OpenAIResponseAPIMessageBuilder.user()
            .add_text("What is the weather in Paris?")
            .prepare_message()[0]
        ]
        parsed_output = model(messages)

    assert parsed_output.raw_response is not None
    assert (
        parsed_output.action == """get_weather(location="Paris")"""
    ), f""" Expected get_weather(location="Paris") but got {parsed_output.action}"""
    assert global_tracker.stats["input_tokens"] > 0
    assert global_tracker.stats["output_tokens"] > 0
    assert global_tracker.stats["cost"] > 0


@pytest.mark.pricy
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_response_model_with_multiple_messages_and_cost_tracking():
    """
    Test OpenAIResponseModel's output parsing and cost tracking
    with a tool-using assistant and follow-up interaction.
    """
    args = OpenAIResponseModelArgs(model_name="gpt-4.1", temperature=1e-5, max_new_tokens=100)

    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather in a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the weather for.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature.",
                    },
                },
                "required": ["location"],
            },
        }
    ]

    model = args.make_model(tools=tools, tool_choice="required")
    builder = args.get_message_builder()

    messages = [builder.user().add_text("What is the weather in Paris?")]

    with tracking.set_tracker() as tracker:
        # First turn: get initial tool call
        parsed = model(messages)
        prev_input = tracker.stats["input_tokens"]
        prev_output = tracker.stats["output_tokens"]
        prev_cost = tracker.stats["cost"]

        # Simulate tool execution and user follow-up
        messages += [
            parsed.tool_calls,  # Add tool call from the model
            builder.tool(parsed.raw_response).add_text("Its sunny! 25°C"),
            builder.user().add_text("What is the weather in Delhi?"),
        ]

        parsed = model(messages)

        # Token and cost deltas
        delta_input = tracker.stats["input_tokens"] - prev_input
        delta_output = tracker.stats["output_tokens"] - prev_output
        delta_cost = tracker.stats["cost"] - prev_cost

    # Assertions
    assert prev_input > 0
    assert prev_output > 0
    assert prev_cost > 0
    assert parsed.raw_response is not None
    assert parsed.action == 'get_weather(location="Delhi")', f"Unexpected action: {parsed.action}"
    assert delta_input > 0
    assert delta_output > 0
    assert delta_cost > 0
    assert tracker.stats["input_tokens"] == prev_input + delta_input
    assert tracker.stats["output_tokens"] == prev_output + delta_output
    assert tracker.stats["cost"] == pytest.approx(prev_cost + delta_cost)


@pytest.mark.pricy
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_chat_completion_model_with_multiple_messages_and_cost_tracking():
    """
    Test OpenAIResponseModel's output parsing and cost tracking
    with a tool-using assistant and follow-up interaction.
    """
    args = OpenAIChatModelArgs(model_name="gpt-4.1", temperature=1e-5, max_new_tokens=100)

    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather in a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the weather for.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature.",
                    },
                },
                "required": ["location"],
            },
        }
    ]

    model = args.make_model(tools=tools, tool_choice="required")
    builder = args.get_message_builder()

    messages = [builder.user().add_text("What is the weather in Paris?")]

    with tracking.set_tracker() as tracker:
        # First turn: get initial tool call
        parsed = model(messages)
        prev_input = tracker.stats["input_tokens"]
        prev_output = tracker.stats["output_tokens"]
        prev_cost = tracker.stats["cost"]

        # Simulate tool execution and user follow-up
        messages += [
            parsed.tool_calls,  # Add tool call from the model
            builder.tool(parsed.raw_response).add_text("Its sunny! 25°C"),
            builder.user().add_text("What is the weather in Delhi?"),
        ]

        parsed = model(messages)

        # Token and cost deltas
        delta_input = tracker.stats["input_tokens"] - prev_input
        delta_output = tracker.stats["output_tokens"] - prev_output
        delta_cost = tracker.stats["cost"] - prev_cost

    # Assertions
    assert prev_input > 0
    assert prev_output > 0
    assert prev_cost > 0
    assert parsed.raw_response is not None
    assert parsed.action == 'get_weather(location="Delhi")', f"Unexpected action: {parsed.action}"
    assert delta_input > 0
    assert delta_output > 0
    assert delta_cost > 0
    assert tracker.stats["input_tokens"] == prev_input + delta_input
    assert tracker.stats["output_tokens"] == prev_output + delta_output
    assert tracker.stats["cost"] == pytest.approx(prev_cost + delta_cost)


@pytest.mark.pricy
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_claude_model_with_multiple_messages_pricy_call():
    model_factory = ClaudeResponseModelArgs(
        model_name="claude-3-haiku-20240307", temperature=1e-5, max_new_tokens=100
    )
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the weather for.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature.",
                    },
                },
                "required": ["location"],
            },
        }
    ]
    model = model_factory.make_model(tools=tools)
    msg_builder = model_factory.get_message_builder()
    messages = []

    messages.append(msg_builder.user().add_text("What is the weather in Paris?"))
    with tracking.set_tracker() as global_tracker:
        llm_output1 = model(messages)

        prev_input = global_tracker.stats["input_tokens"]
        prev_output = global_tracker.stats["output_tokens"]
        prev_cost = global_tracker.stats["cost"]

        messages.append(llm_output1.tool_calls)
        messages.append(msg_builder.tool(llm_output1.raw_response).add_text("Its sunny! 25°C"))
        messages.append(msg_builder.user().add_text("What is the weather in Delhi?"))
        llm_output2 = model(messages)
        # Token and cost deltas
        delta_input = global_tracker.stats["input_tokens"] - prev_input
        delta_output = global_tracker.stats["output_tokens"] - prev_output
        delta_cost = global_tracker.stats["cost"] - prev_cost

    # Assertions
    assert prev_input > 0, "Expected previous input tokens to be greater than 0"
    assert prev_output > 0, "Expected previous output tokens to be greater than 0"
    assert prev_cost > 0, "Expected previous cost value to be greater than 0"
    assert llm_output2.raw_response is not None
    assert (
        llm_output2.action == 'get_weather(location="Delhi", unit="celsius")'
    ), f'Expected get_weather("Delhi") but got {llm_output2.action}'
    assert delta_input > 0, "Expected new input tokens to be greater than 0"
    assert delta_output > 0, "Expected new output tokens to be greater than 0"
    assert delta_cost > 0, "Expected new cost value to be greater than 0"
    assert global_tracker.stats["input_tokens"] == prev_input + delta_input
    assert global_tracker.stats["output_tokens"] == prev_output + delta_output
    assert global_tracker.stats["cost"] == pytest.approx(prev_cost + delta_cost)


# TODO: Add tests for image token costing (this is complex and model-specific)
#       - For OpenAI, you'd need to know how they bill for images (e.g., fixed cost per image + tokens for text parts)
#       - You'd likely need to mock the response from client.chat.completions.create to include specific usage for images.
