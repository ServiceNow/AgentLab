import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain.schema import AIMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from agentlab.llm.langchain_utils import HuggingFaceChatModel, RekaChatModel

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

    @property
    def short_model_name(self):
        if "/" in self.model_name:
            return self.model_name.split("/", 1)[1]
        else:
            return self.model_name

    @property
    def base_model_name(self):
        return self.model_name.split("/")[0]


@dataclass
class APIModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an API."""

    def make_model(self):
        base_name = self.model_name.split("/")[0]
        if base_name == "openai":
            model_name = self.model_name.split("/")[-1]

            return ChatOpenAI(
                model_name=model_name,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
        elif base_name == "azure":
            model_name = self.model_name.split("/")[1]
            deployment_name = self.model_name.split("/")[2]
            return AzureChatOpenAI(
                model_name=model_name,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                deployment_name=deployment_name,
            )
        elif base_name == "reka":
            model_name = self.model_name.split("/")[-1]
            return RekaChatModel(
                model_name=model_name,
            )
        elif base_name == "huggingface":
            return HuggingFaceChatModel(
                model_name=self.model_name.split("/", 1)[1],
                hf_hosted=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                max_total_tokens=self.max_total_tokens,
                max_input_tokens=self.max_input_tokens,
                model_url=None,
                n_retry_server=4,
            )
        else:
            raise ValueError(f"Backend {base_name} is not supported")


@dataclass
class OpenAiModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenAI
    model."""

    def make_model(self):
        if self.base_model_name != "openai":
            raise ValueError(f"Model {self.model_name} is not an OpenAI model")

        return ChatOpenAI(
            model_name=self.short_model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )


class HuggingFaceModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with a HuggingFace model."""

    def make_model(self):
        return HuggingFaceChatModel(
            model_name=self.model_name,
            hf_hosted=True,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            max_total_tokens=self.max_total_tokens,
            max_input_tokens=self.max_input_tokens,
            model_url=None,
            n_retry_server=4,
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
            # client = InferenceClient(model=self.model_url, token=self.token)
            # llm = HuggingFaceEndpoint(
            #     name=self.model_name,
            #     endpoint_url=self.model_url,
            #     max_new_tokens=self.max_new_tokens,
            #     temperature=self.temperature,
            #     client=client,
            # )

            # return ChatHuggingFace(model_id=self.model_name, llm=llm)

            return HuggingFaceChatModel(
                model_name=self.model_name,
                hf_hosted=False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                max_total_tokens=self.max_total_tokens,
                max_input_tokens=self.max_input_tokens,
                model_url=self.model_url,
                n_retry_server=self.n_retry_server,
            )
        else:
            raise ValueError(f"Backend {self.backend} is not supported")
