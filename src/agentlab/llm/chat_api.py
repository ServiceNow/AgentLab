import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import openai
from openai import AzureOpenAI, OpenAI

import agentlab.llm.tracking as tracking
from agentlab.llm.huggingface_utils import HuggingFaceURLChatModel
from agentlab.llm.llm_utils import _extract_wait_time


class CheatMiniWoBLLM:
    """For unit-testing purposes only. It only work with miniwob.click-test task."""

    def invoke(self, messages) -> str:
        prompt = messages[-1].get("content", "")
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
        return {"role": "assistant", "content": answer}

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
    def make_model(self) -> "ChatModel":
        pass

    def prepare_server(self):
        pass

    def close_server(self):
        pass


@dataclass
class OpenRouterModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenAI
    model."""

    def make_model(self):
        return OpenRouterChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )


@dataclass
class OpenAIModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenAI
    model."""

    def make_model(self):
        return OpenAIChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )


@dataclass
class AzureModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an Azure model."""

    deployment_name: str = None

    def make_model(self):
        return AzureChatModel(
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


class ChatModel(ABC):

    @abstractmethod
    def __init__(self, model_name, api_key=None, temperature=0.5, max_tokens=100, max_retry=1):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retry = max_retry

        self.client = OpenAI()

        self.input_cost = 0.0
        self.output_cost = 0.0

    def __call__(self, messages: list[dict]) -> dict:
        completion = None
        for itr in range(self.max_retry):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                break
            except openai.OpenAIError as e:
                logging.warning(
                    f"Failed to get a response from the API: \n{e}\n"
                    f"Retrying... ({itr+1}/{self.max_retry})"
                )
                wait_time = _extract_wait_time(e)
                logging.info(f"Waiting for {wait_time} seconds")
                time.sleep(wait_time)
                # TODO: add total delay limit ?

        if not completion:
            raise Exception("Failed to get a response from the API")

        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = input_tokens * self.input_cost + output_tokens * self.output_cost

        if hasattr(tracking.TRACKER, "instance") and isinstance(
            tracking.TRACKER.instance, tracking.LLMTracker
        ):
            tracking.TRACKER.instance(input_tokens, output_tokens, cost)

        return dict(role="assistant", content=completion.choices[0].message.content)

    def invoke(self, messages: list[dict]) -> dict:
        return self(messages)


class OpenRouterChatModel(ChatModel):
    def __init__(self, model_name, api_key=None, temperature=0.5, max_tokens=100, max_retry=1):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retry = max_retry

        api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        pricings = tracking.get_pricing_openrouter()

        self.input_cost = pricings[model_name]["prompt"]
        self.output_cost = pricings[model_name]["completion"]

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )


class OpenAIChatModel(ChatModel):
    def __init__(self, model_name, api_key=None, temperature=0.5, max_tokens=100, max_retry=1):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retry = max_retry

        api_key = api_key or os.getenv("OPENAI_API_KEY")

        pricings = tracking.get_pricing_openai()

        self.input_cost = float(pricings[model_name]["prompt"])
        self.output_cost = float(pricings[model_name]["completion"])

        self.client = OpenAI(
            api_key=api_key,
        )


class AzureChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        deployment_name=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=1,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retry = max_retry

        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")

        # AZURE_OPENAI_ENDPOINT has to be defined in the environment
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        assert endpoint, "AZURE_OPENAI_ENDPOINT has to be defined in the environment"

        pricings = tracking.get_pricing_openai()

        self.input_cost = float(pricings[model_name]["prompt"])
        self.output_cost = float(pricings[model_name]["completion"])

        self.client = AzureOpenAI(
            api_key=api_key,
            azure_deployment=deployment_name,
            azure_endpoint=endpoint,
            api_version="2024-02-01",
        )
