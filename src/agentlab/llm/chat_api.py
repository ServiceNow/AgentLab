import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain.schema import AIMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from agentlab.llm.langchain_utils import HuggingFaceAPIChatModel, HuggingFaceURLChatModel

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


class CheatMiniWoBLLM:
    """For unit-testing purposes only. It only work with miniwob.click-test task."""

    def invoke(self, messages) -> str:
        prompt = messages[-1].content
        match = re.search(r"^\s*\[(\d+)\].*button", prompt, re.MULTILINE | re.IGNORECASE)

        if match:
            bid = match.group(1)
            action = f'click("{bid}")'
        else:
            raise Exception("Can't find the button's bid")

        answer = f"""I'm clicking the button as requested.
<action>
{action}
</action>
"""
        return AIMessage(content=answer)

    def __call__(self, messages) -> str:
        return self.invoke(messages)


@dataclass
class CheatMiniWoBLLMArgs:
    model_name = "test/cheat_miniwob_click_test"
    max_total_tokens = 10240
    max_input_tokens = 8000
    max_new_tokens = 128

    def make_model(self):
        return CheatMiniWoBLLM()

    def prepare_server(self):
        pass

    def close_server(self):
        pass


@dataclass
class BaseModelArgs(ABC):
    """Base class for all model arguments."""

    model_name: str
    max_total_tokens: int = None
    max_input_tokens: int = None
    max_new_tokens: int = None
    temperature: float = 0.1
    vision_support: bool = False

    @abstractmethod
    def make_model(self) -> "BaseChatModel":
        pass

    def prepare_server(self):
        pass

    def close_server(self):
        pass


@dataclass
class OpenAIModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenAI
    model."""

    def make_model(self):
        return ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )


@dataclass
class HuggingFaceModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with a HuggingFace model."""

    def make_model(self):
        return HuggingFaceAPIChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            n_retry_server=4,
        )


@dataclass
class AzureModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an Azure model."""

    deployment_name: str = None

    def make_model(self):
        return AzureChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            deployment_name=self.deployment_name,
        )


@dataclass
class SelfHostedModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with a self-hosted model."""

    model_url: str = None
    token: str = None
    backend: str = "huggingface"
    n_retry_server: int = 4

    def make_model(self):
        if self.backend == "huggingface":
            # currently only huggingface tgi servers are supported
            if self.model_url is None:
                self.model_url = os.environ["AGENTLAB_MODEL_URL"]
            if self.token is None:
                self.token = os.environ["AGENTLAB_MODEL_TOKEN"]

            return HuggingFaceURLChatModel(
                model_name=self.model_name,
                model_url=self.model_url,
                token=self.token,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                n_retry_server=self.n_retry_server,
            )
        else:
            raise ValueError(f"Backend {self.backend} is not supported")


@dataclass
class ChatModelArgs(BaseModelArgs):
    """Object added for backward compatibility with the old ChatModelArgs."""

    model_path: str = None
    model_url: str = None
    model_size: str = None
    training_total_tokens: int = None
    hf_hosted: bool = False
    is_model_operational: str = False
    sliding_window: bool = False
    n_retry_server: int = 4
    infer_tokens_length: bool = False
    vision_support: bool = False
    shard_support: bool = True
    extra_tgi_args: dict = None
    tgi_image: str = None
    info: dict = None

    def __post_init__(self):
        import warnings

        warnings.simplefilter("always", DeprecationWarning)
        warnings.warn(
            "ChatModelArgs is deprecated and used only for xray. Use one of the specific model args classes instead.",
            DeprecationWarning,
        )
        warnings.simplefilter("default", DeprecationWarning)

    def make_model(self):
        pass
