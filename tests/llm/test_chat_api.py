import os

import pytest
from langchain.schema import HumanMessage, SystemMessage

from agentlab.llm.chat_api import AzureModelArgs, HuggingFaceModelArgs, OpenAIModelArgs
from agentlab.llm.llm_utils import download_and_save_model
from agentlab.llm.prompt_templates import STARCHAT_PROMPT_TEMPLATE

# TODO(optimass): figure out a good model for all tests


if "AGENTLAB_LOCAL_TEST" in os.environ:
    skip_tests = os.environ["AGENTLAB_LOCAL_TEST"] != "1"
else:
    skip_tests = False


@pytest.mark.pricy
@pytest.mark.skipif(skip_tests, reason="Skipping on remote as Azure is pricy")
def test_api_model_args_azure():
    model_args = AzureModelArgs(
        model_name="gpt-35-turbo",
        deployment_name="gpt-35-turbo",
        max_total_tokens=8192,
        max_input_tokens=8192 - 512,
        max_new_tokens=512,
        temperature=1e-1,
    )
    model = model_args.make_model()

    messages = [
        SystemMessage(content="You are an helpful virtual assistant"),
        HumanMessage(content="Give the third prime number"),
    ]
    answer = model.invoke(messages)

    assert "5" in answer.content


@pytest.mark.pricy
@pytest.mark.skip(reason="Skipping atm for lack of better marking")
def test_api_model_args_openai():
    model_args = OpenAIModelArgs(
        model_name="gpt-3.5-turbo-0125",
        max_total_tokens=8192,
        max_input_tokens=8192 - 512,
        max_new_tokens=512,
        temperature=1e-1,
    )
    model = model_args.make_model()

    messages = [
        SystemMessage(content="You are an helpful virtual assistant"),
        HumanMessage(content="Give the third prime number"),
    ]
    answer = model.invoke(messages)

    assert "5" in answer.content
