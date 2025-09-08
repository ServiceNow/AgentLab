import os
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import anthropic
import openai
import pytest

from agentlab.llm import tracking
from agentlab.llm.response_api import (
    AnthropicAPIMessageBuilder,
    APIPayload,
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
            tool_call_mock.function = MagicMock()
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
    prompt_tokens_details_mock = MagicMock()
    prompt_tokens_details_mock.cached_tokens = 0
    completion.usage.prompt_tokens_details = prompt_tokens_details_mock

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
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }
    message.to_dict.return_value = {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
    }
    return completion


responses_api_tools = [
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

chat_api_tools = [
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
anthropic_tools = [
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
    response.usage.cache_input_tokens = 0
    response.usage.cache_creation_input_tokens = 0
    return response


def create_mock_openai_responses_api_response(
    outputs: Optional[List[Dict[str, Any]]] = None, input_tokens: int = 10, output_tokens: int = 20
) -> MagicMock:
    """
    Helper to create a mock response object similar to what
    openai.resources.Responses.create() would return.
    Compatible with OpenAIResponseModel and TrackAPIPricingMixin.
    """

    response_mock = MagicMock(spec=openai.types.responses.response.Response)
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
    response_mock.usage = MagicMock(spec=openai.types.responses.response.ResponseUsage)
    response_mock.usage.input_tokens = input_tokens
    response_mock.usage.output_tokens = output_tokens
    response_mock.usage.prompt_tokens = input_tokens
    response_mock.usage.completion_tokens = output_tokens
    input_tokens_details_mock = MagicMock()
    input_tokens_details_mock.cached_tokens = 0
    response_mock.usage.input_tokens_details = input_tokens_details_mock

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
    messages = builder.prepare_message()

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [{"type": "text", "text": "Hello, ChatCompletion!"}]


def test_openai_chat_completion_api_message_builder_image():
    builder = OpenAIChatCompletionAPIMessageBuilder.user()
    builder.add_image("data:image/jpeg;base64,CHATCOMPLETIONBASE64")
    messages = builder.prepare_message()

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,CHATCOMPLETIONBASE64"}}
    ]


def test_openai_chat_completion_model_parse_and_cost():
    args = OpenAIChatModelArgs(model_name="gpt-3.5-turbo")
    with patch("agentlab.llm.response_api.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        model = args.make_model()

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
        with tracking.set_tracker() as global_tracker:
            messages = [
                OpenAIChatCompletionAPIMessageBuilder.user().add_text(
                    "What's the weather in Paris?"
                )
            ]
            payload = APIPayload(messages=messages)
            parsed_output = model(payload)

    mock_create.assert_called_once()
    assert parsed_output.raw_response.choices[0].message.content == "This is a test thought."
    assert parsed_output.action == """get_weather(location='Paris')"""
    assert parsed_output.raw_response.choices[0].message.tool_calls[0].id == "call_123"
    # Check cost tracking (token counts)
    assert global_tracker.stats["input_tokens"] == 50
    assert global_tracker.stats["output_tokens"] == 30
    assert global_tracker.stats["cost"] > 0


def test_claude_response_model_parse_and_cost():
    args = ClaudeResponseModelArgs(model_name="claude-3-haiku-20240307")
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
            messages = [AnthropicAPIMessageBuilder.user().add_text("Search for latest news")]
            payload = APIPayload(messages=messages)
            parsed_output = model(payload)

    mock_create.assert_called_once()
    fn_call = next(iter(parsed_output.tool_calls))

    assert "Thinking about the request." in parsed_output.think
    assert parsed_output.action == """search_web(query='latest news')"""
    assert fn_call.name == "search_web"
    assert global_tracker.stats["input_tokens"] == 40
    assert global_tracker.stats["output_tokens"] == 20


def test_openai_response_model_parse_and_cost():
    args = OpenAIResponseModelArgs(model_name="gpt-4.1")

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

    with patch("agentlab.llm.response_api.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        model = args.make_model()

    with patch.object(
        model.client.responses, "create", return_value=mock_api_resp
    ) as mock_create_method:
        with tracking.set_tracker() as global_tracker:
            messages = [
                OpenAIResponseAPIMessageBuilder.user().add_text("What's the weather in Boston?")
            ]
            payload = APIPayload(messages=messages)
            parsed_output = model(payload)

    mock_create_method.assert_called_once()
    fn_calls = [
        content
        for content in parsed_output.tool_calls.raw_calls.output
        if content.type == "function_call"
    ]
    assert parsed_output.action == "get_current_weather(location='Boston, MA', unit='celsius')"
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

    tools = chat_api_tools
    model = args.make_model()

    with tracking.set_tracker() as global_tracker:
        messages = [
            OpenAIChatCompletionAPIMessageBuilder.user().add_text("What is the weather in Paris?")
        ]
        payload = APIPayload(messages=messages, tools=tools, tool_choice="required")
        parsed_output = model(payload)

    assert parsed_output.raw_response is not None
    assert (
        parsed_output.action == "get_weather(location='Paris')"
    ), f""" Expected get_weather(location='Paris') but got {parsed_output.action}"""
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
    tools = anthropic_tools
    model = args.make_model()

    with tracking.set_tracker() as global_tracker:
        messages = [AnthropicAPIMessageBuilder.user().add_text("What is the weather in Paris?")]
        payload = APIPayload(messages=messages, tools=tools)
        parsed_output = model(payload)

    assert parsed_output.raw_response is not None
    assert (
        parsed_output.action == "get_weather(location='Paris')"
    ), f"""Expected get_weather('Paris') but got {parsed_output.action}"""
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

    tools = responses_api_tools
    model = args.make_model()

    with tracking.set_tracker() as global_tracker:
        messages = [
            OpenAIResponseAPIMessageBuilder.user().add_text("What is the weather in Paris?")
        ]
        payload = APIPayload(messages=messages, tools=tools)
        parsed_output = model(payload)

    assert parsed_output.raw_response is not None
    assert (
        parsed_output.action == """get_weather(location='Paris', unit='celsius')"""
    ), f""" Expected get_weather(location='Paris', unit='celsius') but got {parsed_output.action}"""
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

    tools = responses_api_tools
    model = args.make_model()
    builder = args.get_message_builder()

    messages = [builder.user().add_text("What is the weather in Paris?")]

    with tracking.set_tracker() as tracker:
        payload = APIPayload(messages=messages, tools=tools, tool_choice="required")
        parsed = model(payload)
        prev_input = tracker.stats["input_tokens"]
        prev_output = tracker.stats["output_tokens"]
        prev_cost = tracker.stats["cost"]

        assert parsed.tool_calls, "Expected tool calls in the response"
        # Set tool responses
        for tool_call in parsed.tool_calls:
            tool_call.response_text("Its sunny! 25°C")
        # Simulate tool execution and user follow-up
        messages += [
            builder.add_responded_tool_calls(parsed.tool_calls),
            builder.user().add_text("What is the weather in Delhi?"),
        ]

        payload = APIPayload(messages=messages, tools=tools, tool_choice="required")
        parsed = model(payload)

        delta_input = tracker.stats["input_tokens"] - prev_input
        delta_output = tracker.stats["output_tokens"] - prev_output
        delta_cost = tracker.stats["cost"] - prev_cost

    assert prev_input > 0
    assert prev_output > 0
    assert prev_cost > 0
    assert parsed.raw_response is not None
    assert (
        parsed.action == """get_weather(location='Delhi', unit='celsius')"""
    ), f"Unexpected action: {parsed.action}"
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

    model = args.make_model()
    builder = args.get_message_builder()

    messages = [builder.user().add_text("What is the weather in Paris?")]

    with tracking.set_tracker() as tracker:
        payload = APIPayload(messages=messages, tools=tools, tool_choice="required")
        parsed = model(payload)
        prev_input = tracker.stats["input_tokens"]
        prev_output = tracker.stats["output_tokens"]
        prev_cost = tracker.stats["cost"]

        for tool_call in parsed.tool_calls:
            tool_call.response_text("Its sunny! 25°C")
        # Simulate tool execution and user follow-up
        messages += [
            builder.add_responded_tool_calls(parsed.tool_calls),
            builder.user().add_text("What is the weather in Delhi?"),
        ]
        # Set tool responses

        payload = APIPayload(messages=messages, tools=tools, tool_choice="required")
        parsed = model(payload)

        delta_input = tracker.stats["input_tokens"] - prev_input
        delta_output = tracker.stats["output_tokens"] - prev_output
        delta_cost = tracker.stats["cost"] - prev_cost

    assert prev_input > 0
    assert prev_output > 0
    assert prev_cost > 0
    assert parsed.raw_response is not None
    assert (
        parsed.action == """get_weather(location='Delhi')"""
    ), f"Unexpected action: {parsed.action}"
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
    model = model_factory.make_model()
    msg_builder = model_factory.get_message_builder()
    messages = []

    messages.append(msg_builder.user().add_text("What is the weather in Paris?"))
    with tracking.set_tracker() as global_tracker:
        payload = APIPayload(messages=messages, tools=tools)
        llm_output1 = model(payload)

        prev_input = global_tracker.stats["input_tokens"]
        prev_output = global_tracker.stats["output_tokens"]
        prev_cost = global_tracker.stats["cost"]

        for tool_call in llm_output1.tool_calls:
            tool_call.response_text("It's sunny! 25°C")
        messages += [
            msg_builder.add_responded_tool_calls(llm_output1.tool_calls),
            msg_builder.user().add_text("What is the weather in Delhi?"),
        ]
        payload = APIPayload(messages=messages, tools=tools)
        llm_output2 = model(payload)
        delta_input = global_tracker.stats["input_tokens"] - prev_input
        delta_output = global_tracker.stats["output_tokens"] - prev_output
        delta_cost = global_tracker.stats["cost"] - prev_cost

    assert prev_input > 0, "Expected previous input tokens to be greater than 0"
    assert prev_output > 0, "Expected previous output tokens to be greater than 0"
    assert prev_cost > 0, "Expected previous cost value to be greater than 0"
    assert llm_output2.raw_response is not None
    assert (
        llm_output2.action == """get_weather(location='Delhi', unit='celsius')"""
    ), f"""Expected get_weather('Delhi') but got {llm_output2.action}"""
    assert delta_input > 0, "Expected new input tokens to be greater than 0"
    assert delta_output > 0, "Expected new output tokens to be greater than 0"
    assert delta_cost > 0, "Expected new cost value to be greater than 0"
    assert global_tracker.stats["input_tokens"] == prev_input + delta_input
    assert global_tracker.stats["output_tokens"] == prev_output + delta_output
    assert global_tracker.stats["cost"] == pytest.approx(prev_cost + delta_cost)


## Test multiaction
@pytest.mark.pricy
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Skipping as OpenAI API key not set")
def test_multi_action_tool_calls():
    """
    Test that the model can produce multiple tool calls in parallel.
    Uncomment commented lines to see the full behaviour of models and tool choices.
    """
    # test_config (setting name, BaseModelArgs, model_name, tools)
    tool_test_configs = [
        (
            "gpt-4.1-responses API",
            OpenAIResponseModelArgs,
            "gpt-4.1-2025-04-14",
            responses_api_tools,
        ),
        ("gpt-4.1-chat Completions API", OpenAIChatModelArgs, "gpt-4.1-2025-04-14", chat_api_tools),
        # ("claude-3", ClaudeResponseModelArgs, "claude-3-haiku-20240307", anthropic_tools),   # fails
        # ("claude-3.7", ClaudeResponseModelArgs, "claude-3-7-sonnet-20250219", anthropic_tools), # fails
        ("claude-4-sonnet", ClaudeResponseModelArgs, "claude-sonnet-4-20250514", anthropic_tools),
        # add more models as needed
    ]

    def add_user_messages(msg_builder):
        return [
            msg_builder.user().add_text("What is the weather in Paris and Delhi?"),
            msg_builder.user().add_text("You must call multiple tools to achieve the task."),
        ]

    res_df = []

    for tool_choice in [
        # 'none',
        # 'required', # fails for Responses API
        # 'any',  # fails for Responses API
        "auto",
        # 'get_weather'
    ]:
        for name, llm_class, checkpoint_name, tools in tool_test_configs:
            print(name, "tool choice:", tool_choice, "\n", "**" * 10)
            model_args = llm_class(model_name=checkpoint_name, max_new_tokens=200, temperature=None)
            llm, msg_builder = model_args.make_model(), model_args.get_message_builder()
            messages = add_user_messages(msg_builder)
            if tool_choice == "get_weather":  # force a specific tool call
                response: LLMOutput = llm(
                    APIPayload(messages=messages, tools=tools, force_call_tool=tool_choice)
                )
            else:
                response: LLMOutput = llm(
                    APIPayload(messages=messages, tools=tools, tool_choice=tool_choice)
                )
                num_tool_calls = len(response.tool_calls) if response.tool_calls else 0
            res_df.append(
                {
                    "model": name,
                    "checkpoint": checkpoint_name,
                    "tool_choice": tool_choice,
                    "num_tool_calls": num_tool_calls,
                    "action": response.action,
                }
            )
            assert (
                num_tool_calls == 2
            ), f"Expected 2 tool calls, but got {num_tool_calls} for {name} with tool choice {tool_choice}"
        # import pandas as pd
        # print(pd.DataFrame(res_df))


EDGE_CASES = [
    # 1. Empty kwargs dict
    ("valid_function", {}, "valid_function()"),
    # 2. Kwargs with problematic string values (quotes, escapes, unicode)
    (
        "send_message",
        {
            "text": 'He said "Hello!" and used a backslash: \\',
            "unicode": "Café naïve résumé 🚀",
            "newlines": "Line1\nLine2\tTabbed",
        },
        "send_message(text='He said \"Hello!\" and used a backslash: \\\\', unicode='Café naïve résumé 🚀', newlines='Line1\\nLine2\\tTabbed')",
    ),
    # 3. Mixed types including problematic float values
    (
        "complex_call",
        {
            "infinity": float("inf"),
            "nan": float("nan"),
            "negative_zero": -0.0,
            "scientific": 1.23e-45,
        },
        "complex_call(infinity=inf, nan=nan, negative_zero=-0.0, scientific=1.23e-45)",
    ),
    # 4. Deeply nested structures that could stress repr()
    (
        "process_data",
        {
            "nested": {"level1": {"level2": {"level3": [1, 2, {"deep": True}]}}},
            "circular_ref_like": {"a": {"b": {"c": "back_to_start"}}},
        },
        "process_data(nested={'level1': {'level2': {'level3': [1, 2, {'deep': True}]}}}, circular_ref_like={'a': {'b': {'c': 'back_to_start'}}})",
    ),
]


def test_tool_call_to_python_code():
    from agentlab.llm.response_api import tool_call_to_python_code

    for edge_case in EDGE_CASES:
        func_name, kwargs, expected = edge_case
        result = tool_call_to_python_code(func_name, kwargs)
        print(result)
        assert result == expected, f"Expected {expected} but got {result}"


if __name__ == "__main__":
    test_tool_call_to_python_code()
    # test_openai_chat_completion_model_parse_and_cost()
