import os
from functools import partial

import pytest
from agentlab.llm.litellm_api import LiteLLMModelArgs
from agentlab.llm.response_api import APIPayload, LLMOutput

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
    },
    {
        "type": "function",
        "name": "get_time",
        "description": "Get the current time in a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get the time for.",
                }
            },
            "required": ["location"],
        },
    },
]


# test_config (setting name, BaseModelArgs, model_name, tools)
tool_test_configs = [
    ("gpt-4.1", LiteLLMModelArgs, "openai/gpt-4.1-2025-04-14", chat_api_tools),
    # ("claude-3", LiteLLMModelArgs, "anthropic/claude-3-haiku-20240307", anthropic_tools),   # fails for parallel tool calls
    # ("claude-3.7", LiteLLMModelArgs, "anthropic/claude-3-7-sonnet-20250219", anthropic_tools), # fails for parallel tool calls
    ("claude-4-sonnet", LiteLLMModelArgs, "anthropic/claude-sonnet-4-20250514", chat_api_tools),
    # ("gpt-o3", LiteLLMModelArgs, "openai/o3-2025-04-16", chat_api_tools), # fails for parallel tool calls
    # add more models as needed
]


def add_user_messages(msg_builder):
    return [
        msg_builder.user().add_text("What is the weather in Paris and Delhi?"),
        msg_builder.user().add_text("You must call multiple tools to achieve the task."),
    ]


## Test multiaction
@pytest.mark.pricy
def test_multi_action_tool_calls():
    """
    Test that the model can produce multiple tool calls in parallel.
    Note: Remove assert and Uncomment commented lines to see the full behaviour of models and tool choices.
    """
    res_df = []
    for tool_choice in [
        # "none",
        "required",  # fails for Responses API
        "any",  # fails for Responses API
        "auto",
        # "get_weather",  # force a specific tool call
    ]:
        for name, llm_class, checkpoint_name, tools in tool_test_configs:
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
            row = {
                "model": name,
                "checkpoint": checkpoint_name,
                "tool_choice": tool_choice,
                "num_tool_calls": num_tool_calls,
                "action": response.action,
            }
            res_df.append(row)
            assert (
                num_tool_calls == 2
            ), f"Expected 2 tool calls, but got {num_tool_calls} for {name} with tool choice {tool_choice}"
    # import pandas as pd
    # print(pd.DataFrame(res_df))


@pytest.mark.pricy
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Skipping as OpenAI API key not set")
def test_single_tool_call():
    """
    Test that the LLMOutput contains only one tool call when use_only_first_toolcall is True.
    """
    for tool_choice in [
        # 'none',
        "required",
        "any",
        "auto",
    ]:
        for name, llm_class, checkpoint_name, tools in tool_test_configs:
            print(name, "tool choice:", tool_choice, "\n", "**" * 10)
            llm_class = partial(llm_class, use_only_first_toolcall=True)
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
            assert (
                num_tool_calls == 1
            ), f"Expected 1 tool calls, but got {num_tool_calls} for {name} with tool choice {tool_choice }"


@pytest.mark.pricy
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Skipping as OpenAI API key not set")
def test_force_tool_call():
    """
    Test that the model can produce a specific tool call when requested.
    The user message asks the 'weather' but we force call tool "get_time".
    We test if 'get_time' is present in the tool calls.
    Note: Model can have other tool calls as well.
    """
    force_call_tool = "get_time"
    for name, llm_class, checkpoint_name, tools in tool_test_configs:
        model_args = llm_class(model_name=checkpoint_name, max_new_tokens=200, temperature=None)
        llm, msg_builder = model_args.make_model(), model_args.get_message_builder()
        messages = add_user_messages(msg_builder)  # asks weather in Paris and Delhi
        response: LLMOutput = llm(
            APIPayload(messages=messages, tools=tools, force_call_tool=force_call_tool)
        )
        called_fn_names = [call.name for call in response.tool_calls] if response.tool_calls else []
        assert response.tool_calls is not None
        assert any(
            fn_name == "get_time" for fn_name in called_fn_names
        ), f"Model:{name},Expected all tool calls to be 'get_time', but got {called_fn_names} with force call {force_call_tool}"


if __name__ == "__main__":
    test_multi_action_tool_calls()
    test_force_tool_call()
    test_single_tool_call()
