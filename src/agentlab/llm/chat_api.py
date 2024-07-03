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

from agentlab.llm.langchain_utils import HuggingFaceChatModel


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
    max_total_tokens = 1024
    max_input_tokens = 1024 - 128
    max_new_tokens = 128

    def make_chat_model(self):
        return CheatMiniWoBLLM()

    def prepare_server(self, registry):
        pass

    def close_server(self, registry):
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


@dataclass
class ServerModelArgs(BaseModelArgs):
    """Abstract class for server-based models, with methods for preparing and closing the server."""

    model_url: str = None

    def __post_init__(self):
        if self.max_total_tokens is None:
            self.max_total_tokens = 4096

        # TODO move this to tgi servers
        if self.max_new_tokens is None and self.max_input_tokens is not None:
            self.max_new_tokens = self.max_total_tokens - self.max_input_tokens
        elif self.max_new_tokens is not None and self.max_input_tokens is None:
            self.max_input_tokens = self.max_total_tokens - self.max_new_tokens
        elif self.max_new_tokens is None and self.max_input_tokens is None:
            raise ValueError("max_new_tokens or max_input_tokens must be specified")
        pass

    @abstractmethod
    def prepare_server(self, registry):
        pass

    @abstractmethod
    def close_server(self, registry):
        pass

    def key(self):
        return json.dumps(
            {
                "model_name": self.model_name,
                "max_total_tokens": self.max_total_tokens,
                "max_input_tokens": self.max_input_tokens,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
            }
        )


@dataclass
class APIChatModelArgs(BaseModelArgs):
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


@dataclass
class HostedModelArgs(BaseModelArgs):
    model_url: str = None
    token: str = None
    backend: str = "huggingface"
    n_retry_server: int = 4

    def make_model(self):
        if self.backend == "huggingface":
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


@dataclass
class ToolkitModelArgs(ServerModelArgs):
    """Serializable object for instantiating a generic chat model.

    Attributes
    ----------
    model_name : str
        The name or path of the model to use.
    model_path: str, optional
        Sometimes the model is stored locally. This is the path to the model.
    model_size: str, optional
        The size of the model to use. Relevant for TGI serving.
    model_url : str, optional
        The url of the model to use. If None, then model_name or model_name must
        be specified.
    tgi_token: str, optional
        The EAI token to use for authentication on Toolkit. Defaults to snow.optimass_account.cl4code's token.
    temperature : float
        The temperature to use for the model.
    max_new_tokens : int
        The maximum number of tokens to generate.
    max_total_tokens : int
        The maximum number of total tokens (input + output). Defaults to 4096.
    hf_hosted : bool
        Whether the model is hosted on HuggingFace Hub. Defaults to False.
    is_model_operational : bool
        Whether the model is operational or there are issues with it.
    sliding_window: bool
        Whether the model uses a sliding window during training. Defaults to False.
    n_retry_server: int, optional
        The number of times to retry the TGI server if it fails to respond. Defaults to 4.
    info : dict, optional
        Any other information about how the model was finetuned.
    """

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
        if self.model_url is not None and self.hf_hosted:
            raise ValueError("model_url cannot be specified when hf_hosted is True")

    def make_model(self):
        # TODO: eventually check if the path is either a valid repo_id or a valid local checkpoint on DGX
        self.model_name = self.model_path if self.model_path else self.model_name
        return HuggingFaceChatModel(
            model_name=self.model_name,
            hf_hosted=self.hf_hosted,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            max_total_tokens=self.max_total_tokens,
            max_input_tokens=self.max_input_tokens,
            model_url=self.model_url,
            n_retry_server=self.n_retry_server,
        )

    @property
    def model_short_name(self):
        if "/" in self.model_name:
            return self.model_name.split("/")[1]
        else:
            return self.model_name

    # def key(self):
    #     """Return a unique key for these arguments."""
    #     keys = asdict(self)
    #     # removing the model_url since it will be modified by LLM_servers
    #     keys.pop("model_url", None)
    #     return json.dumps(keys, sort_keys=True)

    def prepare_server(self, registry):
        if self.key() in registry:
            self.model_url = registry[self.key()]["model_url"]
        else:
            job_id, model_url = auto_launch_server(self)
            registry[self.key()] = {"job_id": job_id, "model_url": model_url, "is_ready": False}
            self.model_url = model_url

        self.wait_server(registry)
        return

    def close_server(self, registry):
        if self.key() in registry:
            job_id = registry[self.key()]["job_id"]
            kill_server(job_id)
            del registry[self.key()]

    def wait_server(self, registry):
        job_id = registry[self.key()]["job_id"]
        model_url = registry[self.key()]["model_url"]
        is_ready = registry[self.key()]["is_ready"]
        while not is_ready:
            is_ready = check_server_status(job_id, model_url)
            time.sleep(3)
        registry[self.key()]["is_ready"] = is_ready
