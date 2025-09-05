import logging
import os
import re
import time
from dataclasses import dataclass
from functools import partial
from typing import Optional

import anthropic
import openai
from openai import NOT_GIVEN, OpenAI

import agentlab.llm.tracking as tracking
from agentlab.llm.base_api import AbstractChatModel, BaseModelArgs
from agentlab.llm.llm_utils import AIMessage, Discussion


def make_system_message(content: str) -> dict:
    return dict(role="system", content=content)


def make_user_message(content: str) -> dict:
    return dict(role="user", content=content)


def make_assistant_message(content: str) -> dict:
    return dict(role="assistant", content=content)


class CheatMiniWoBLLM(AbstractChatModel):
    """For unit-testing purposes only. It only work with miniwob.click-test task."""

    def __init__(self, wait_time=0) -> None:
        self.wait_time = wait_time

    def __call__(self, messages) -> str:
        if self.wait_time > 0:
            print(f"Waiting for {self.wait_time} seconds")
            time.sleep(self.wait_time)

        if isinstance(messages, Discussion):
            prompt = messages.to_string()
        else:
            prompt = messages[1].get("content", "")
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
        return make_assistant_message(answer)


@dataclass
class CheatMiniWoBLLMArgs:
    model_name = "test/cheat_miniwob_click_test"
    max_total_tokens = 10240
    max_input_tokens = 8000
    max_new_tokens = 128
    wait_time: int = 0

    def make_model(self):
        return CheatMiniWoBLLM(self.wait_time)

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
            log_probs=self.log_probs,
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
            log_probs=self.log_probs,
        )


@dataclass
class AzureModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an Azure model."""

    deployment_name: str = (
        None  # NOTE: deployment_name is deprecated for Azure OpenAI and won't be used.
    )

    def make_model(self):
        return AzureChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            log_probs=self.log_probs,
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
            # Lazy import to avoid importing HF utilities on non-HF paths
            from agentlab.llm.huggingface_utils import HuggingFaceURLChatModel

            return HuggingFaceURLChatModel(
                model_name=self.model_name,
                model_url=self.model_url,
                token=self.token,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                n_retry_server=self.n_retry_server,
                log_probs=self.log_probs,
            )
        elif self.backend == "vllm":
            return VLLMChatModel(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
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


def _extract_wait_time(error_message, min_retry_wait_time=60):
    """Extract the wait time from an OpenAI RateLimitError message."""
    match = re.search(r"try again in (\d+(\.\d+)?)s", error_message)
    if match:
        return max(min_retry_wait_time, float(match.group(1)))
    return min_retry_wait_time


class RetryError(Exception):
    pass


def handle_error(error, itr, min_retry_wait_time, max_retry):
    if not isinstance(error, openai.OpenAIError):
        raise error
    logging.warning(
        f"Failed to get a response from the API: \n{error}\n" f"Retrying... ({itr+1}/{max_retry})"
    )
    wait_time = _extract_wait_time(
        error.args[0],
        min_retry_wait_time=min_retry_wait_time,
    )
    logging.info(f"Waiting for {wait_time} seconds")
    time.sleep(wait_time)
    error_type = error.args[0]
    return error_type


class OpenRouterError(openai.OpenAIError):
    pass


class ChatModel(AbstractChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=60,
        api_key_env_var=None,
        client_class=OpenAI,
        client_args=None,
        pricing_func=None,
        log_probs=False,
    ):
        assert max_retry > 0, "max_retry should be greater than 0"

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retry = max_retry
        self.min_retry_wait_time = min_retry_wait_time
        self.log_probs = log_probs

        # Get the API key from the environment variable if not provided
        if api_key_env_var:
            api_key = api_key or os.getenv(api_key_env_var)
        self.api_key = api_key

        # Get pricing information
        if pricing_func:
            pricings = pricing_func()
            try:
                self.input_cost = float(pricings[model_name]["prompt"])
                self.output_cost = float(pricings[model_name]["completion"])
            except KeyError:
                logging.warning(
                    f"Model {model_name} not found in the pricing information, prices are set to 0. Maybe try upgrading langchain_community."
                )
                self.input_cost = 0.0
                self.output_cost = 0.0
        else:
            self.input_cost = 0.0
            self.output_cost = 0.0

        client_args = client_args or {}
        self.client = client_class(
            api_key=api_key,
            **client_args,
        )

    def __call__(self, messages: list[dict], n_samples: int = 1, temperature: float = None) -> dict:
        # Initialize retry tracking attributes
        self.retries = 0
        self.success = False
        self.error_types = []

        completion = None
        e = None
        for itr in range(self.max_retry):
            self.retries += 1
            temperature = temperature if temperature is not None else self.temperature
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    n=n_samples,
                    temperature=temperature,
                    max_completion_tokens=self.max_tokens,
                    logprobs=self.log_probs,
                )

                if completion.usage is None:
                    raise OpenRouterError(
                        "The completion object does not contain usage information. This is likely a bug in the OpenRouter API."
                    )

                self.success = True
                break
            except openai.OpenAIError as e:
                error_type = handle_error(e, itr, self.min_retry_wait_time, self.max_retry)
                self.error_types.append(error_type)

        if not completion:
            raise RetryError(
                f"Failed to get a response from the API after {self.max_retry} retries\n"
                f"Last error: {error_type}"
            )

        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = input_tokens * self.input_cost + output_tokens * self.output_cost

        if hasattr(tracking.TRACKER, "instance") and isinstance(
            tracking.TRACKER.instance, tracking.LLMTracker
        ):
            tracking.TRACKER.instance(input_tokens, output_tokens, cost)

        if n_samples == 1:
            res = AIMessage(completion.choices[0].message.content)
            if self.log_probs:
                res["log_probs"] = completion.choices[0].log_probs
            return res
        else:
            return [AIMessage(c.message.content) for c in completion.choices]

    def get_stats(self):
        return {
            "n_retry_llm": self.retries,
            # "busted_retry_llm": int(not self.success), # not logged if it occurs anyways
        }


class OpenAIChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=60,
        log_probs=False,
    ):
        if max_tokens is None:
            max_tokens = NOT_GIVEN
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            api_key_env_var="OPENAI_API_KEY",
            client_class=OpenAI,
            pricing_func=partial(tracking.get_pricing_litellm, model_name=model_name),
            log_probs=log_probs,
        )


class OpenRouterChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=60,
        log_probs=False,
    ):
        client_args = {
            "base_url": "https://openrouter.ai/api/v1",
        }
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            api_key_env_var="OPENROUTER_API_KEY",
            client_class=OpenAI,
            client_args=client_args,
            pricing_func=tracking.get_pricing_openrouter,
            log_probs=log_probs,
        )


class AzureChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        deployment_name=None,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=60,
        log_probs=False,
    ):
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        assert (
            api_key
        ), "AZURE_OPENAI_API_KEY has to be defined in the environment when using AzureChatModel"
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        assert (
            endpoint
        ), "AZURE_OPENAI_ENDPOINT has to be defined in the environment when using AzureChatModel"

        if deployment_name is not None:
            logging.info(
                f"Deployment name is deprecated for Azure OpenAI and won't be used. Using model name: {model_name}."
            )

        client_args = {
            "base_url": endpoint,
            "default_query": {"api-version": "preview"},
        }
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            client_class=OpenAI,
            client_args=client_args,
            pricing_func=tracking.get_pricing_openai,
            log_probs=log_probs,
        )


def __getattr__(name: str):
    """Lazy re-export of optional classes to keep imports light.

    This lets users import HuggingFaceURLChatModel from agentlab.llm.chat_api
    without importing heavy dependencies unless actually used.

    Args:
        name: The name of the attribute to retrieve.

    Returns:
        The requested class or raises AttributeError if not found.

    Raises:
        AttributeError: If the requested attribute is not available.
    """
    if name == "HuggingFaceURLChatModel":
        from agentlab.llm.huggingface_utils import HuggingFaceURLChatModel

        return HuggingFaceURLChatModel
    raise AttributeError(name)


class VLLMChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        n_retry_server=4,
        min_retry_wait_time=60,
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=n_retry_server,
            min_retry_wait_time=min_retry_wait_time,
            api_key_env_var="VLLM_API_KEY",
            client_class=OpenAI,
            client_args={"base_url": "http://0.0.0.0:8000/v1"},
            pricing_func=None,
        )


class AnthropicChatModel(AbstractChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retry = max_retry

        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)

    def __call__(self, messages: list[dict], n_samples: int = 1, temperature: float = None) -> dict:
        # Convert OpenAI format to Anthropic format
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

        temperature = temperature if temperature is not None else self.temperature

        for attempt in range(self.max_retry):
            try:
                kwargs = {
                    "model": self.model_name,
                    "messages": anthropic_messages,
                    "max_tokens": self.max_tokens,
                    "temperature": temperature,
                }

                if system_message:
                    kwargs["system"] = system_message

                response = self.client.messages.create(**kwargs)

                # Track usage if available
                if hasattr(tracking.TRACKER, "instance"):
                    tracking.TRACKER.instance(
                        response.usage.input_tokens,
                        response.usage.output_tokens,
                        0,  # cost calculation would need pricing info
                    )

                return AIMessage(response.content[0].text)

            except Exception as e:
                if attempt == self.max_retry - 1:
                    raise e
                logging.warning(f"Anthropic API error (attempt {attempt + 1}): {e}")
                time.sleep(60)  # Simple retry delay


@dataclass
class AnthropicModelArgs(BaseModelArgs):
    def make_model(self):
        return AnthropicChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
