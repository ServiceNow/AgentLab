import pytest
from langchain.schema import HumanMessage, SystemMessage

from agentlab.llm.chat_api import APIModelArgs
from agentlab.llm.llm_utils import download_and_save_model
from agentlab.llm.prompt_templates import STARCHAT_PROMPT_TEMPLATE

# TODO(optimass): figure out a good model for all tests


@pytest.mark.pricy
@pytest.mark.skip(reason="Skipping atm for lack of better marking")
def test_api_model_args_hf():
    model_name = "huggingface/HuggingFaceH4/starchat-beta"

    model_args = APIModelArgs(
        model_name=model_name,
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
def test_api_model_args_azure():
    model_args = APIModelArgs(
        model_name="azure/gpt-35-turbo/gpt-35-turbo",
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
