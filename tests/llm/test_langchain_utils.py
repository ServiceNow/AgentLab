import pytest

from agentlab.llm.chat_api import HuggingFaceAPIChatModel, HuggingFaceURLChatModel
from agentlab.llm.llm_utils import download_and_save_model
from agentlab.llm.prompt_templates import STARCHAT_PROMPT_TEMPLATE

# TODO(optimass): figure out a good model for all tests


@pytest.mark.skip(reason="We can quickly hit the free tier limit on HuggingFace Hub")
def test_CustomLLMChatbot_remotely():
    # model_path = "google/flan-t5-base"  # remote model on HuggingFace Hub
    model_path = "HuggingFaceH4/starchat-beta"  # remote model on HuggingFace Hub

    chatbot = HuggingFaceAPIChatModel(
        model_path=model_path,
        prompt_template=STARCHAT_PROMPT_TEMPLATE,
        temperature=1e-3,
    )

    messages = [
        dict(role="system", content="You are an helpful virtual assistant"),
        dict(role="user", content="Is python a programming language?"),
    ]

    answer = chatbot(messages)

    print(answer.content)


@pytest.mark.skip(reason="Requires a local model checkpoint")
def test_CustomLLMChatbot_locally():
    # model_path = "google/flan-t5-base"  # remote model on HuggingFace Hub
    model_path = "/mnt/ui_copilot/data_rw/models/starcoderbase-1b-ft"  # local model in shared volum

    chatbot = HuggingFaceURLChatModel(model_path=model_path, temperature=1e-3)

    messages = [
        dict(role="system", content="Please tell me back the following word: "),
        dict(role="user", content="bird"),
    ]

    answer = chatbot(messages)

    print(answer.content)


@pytest.mark.skip(reason="Requires downloading a large file on disk local model checkpoint")
def test_download_and_save_model():
    model_path = "meta-llama/Llama-2-70b-chat"
    save_dir = "test_models"

    download_and_save_model(model_path, save_dir)
