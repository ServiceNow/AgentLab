from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import re

from agentlab.llm.toolkit_servers import (
    auto_launch_server,
    kill_server,
    check_server_status,
)
from langchain.schema import AIMessage
import time

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from dataclasses import dataclass

from langchain.schema import AIMessage

from agentlab.llm.langchain_utils import HuggingFaceChatModel, RekaChatModel


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

    @abstractmethod
    def make_model(self):
        pass

    def prepare_server(self):
        pass

    def close_server(self):
        pass


@dataclass
class APIModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an API."""

    vision_support: bool = False

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
        if base_name == "reka":
            model_name = self.model_name.split("/")[-1]
            return RekaChatModel(
                model_name=model_name,
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
                raise ValueError("model_url must be specified for huggingface backend")
            if self.token is None:
                raise ValueError("token must be specified for huggingface backend")
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
