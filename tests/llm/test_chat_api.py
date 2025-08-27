import os

import pytest

from agentlab.llm.chat_api import (
    AnthropicModelArgs,
    AzureModelArgs,
    OpenAIModelArgs,
    make_system_message,
    make_user_message,
)

# TODO(optimass): figure out a good model for all tests


if "AGENTLAB_LOCAL_TEST" in os.environ:
    skip_tests = os.environ["AGENTLAB_LOCAL_TEST"] != "1"
else:
    skip_tests = False


@pytest.mark.pricy
@pytest.mark.skipif(skip_tests, reason="Skipping on remote as Azure is pricy")
@pytest.mark.skipif(
    not os.getenv("AZURE_OPENAI_API_KEY"), reason="Skipping as Azure API key not set"
)
def test_api_model_args_azure():
    model_args = AzureModelArgs(
        model_name="gpt-4.1-nano",
        deployment_name="gpt-4.1-nano",
        max_total_tokens=8192,
        max_input_tokens=8192 - 512,
        max_new_tokens=512,
        temperature=1e-1,
    )
    model = model_args.make_model()

    messages = [
        make_system_message("You are an helpful virtual assistant"),
        make_user_message("Give the third prime number"),
    ]
    answer = model(messages)

    assert "5" in answer.get("content")


@pytest.mark.pricy
@pytest.mark.skipif(skip_tests, reason="Skipping on remote as Azure is pricy")
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Skipping as OpenAI API key not set")
def test_api_model_args_openai():
    model_args = OpenAIModelArgs(
        model_name="gpt-4o-mini",
        max_total_tokens=8192,
        max_input_tokens=8192 - 512,
        max_new_tokens=512,
        temperature=1e-1,
    )
    model = model_args.make_model()

    messages = [
        make_system_message("You are an helpful virtual assistant"),
        make_user_message("Give the third prime number"),
    ]
    answer = model(messages)

    assert "5" in answer.get("content")


@pytest.mark.pricy
@pytest.mark.skipif(skip_tests, reason="Skipping on remote as Anthropic is pricy")
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="Skipping as Anthropic API key not set"
)
def test_api_model_args_anthropic():
    model_args = AnthropicModelArgs(
        model_name="claude-3-haiku-20240307",
        max_total_tokens=8192,
        max_input_tokens=8192 - 512,
        max_new_tokens=512,
        temperature=1e-1,
    )
    model = model_args.make_model()

    messages = [
        make_system_message("You are an helpful virtual assistant"),
        make_user_message("Give the third prime number. Just the number, no explanation."),
    ]
    answer = model(messages)
    assert "5" in answer.get("content")


if __name__ == "__main__":
    test_api_model_args_anthropic()
