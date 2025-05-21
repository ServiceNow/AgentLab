from abc import ABC, abstractmethod
from accelerate.utils.modeling import load_checkpoint_in_model
from agentlab.llm.llm_utils import AIMessage
from dataclasses import dataclass
from openai import AsyncOpenAI, RateLimitError
from transformers import AutoProcessor, MllamaForConditionalGeneration
from typing import Optional
import asyncio
import backoff
import fnmatch
import os


class VLModel(ABC):
    @abstractmethod
    def __call__(self, messages: list[dict]) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> dict:
        raise NotImplementedError


class VLModelArgs(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def make_model(self) -> VLModel:
        raise NotImplementedError

    @abstractmethod
    def prepare(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    @abstractmethod
    def set_reproducibility_mode(self):
        raise NotImplementedError


class LlamaModel(VLModel):
    def __init__(
        self,
        model_path: str,
        torch_dtype: str,
        checkpoint_dir: str,
        max_length: int,
        max_new_tokens: int,
        reproducibility_config: dict,
    ):
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype
        )
        if checkpoint_dir is not None:
            checkpoint_file = None
            for item in os.listdir(checkpoint_dir):
                if fnmatch.fnmatch(item, "pytorch_model*.bin") or fnmatch.fnmatch(
                    item, "model*.safetensors"
                ):
                    checkpoint_file = os.path.join(checkpoint_dir, item)
                    break
            load_checkpoint_in_model(self.model, checkpoint_file)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.reproducibility_config = reproducibility_config

    def __call__(self, messages: list[dict]) -> dict:
        return {}

    def get_stats(self) -> dict:
        return {}


@dataclass
class LlamaModelArgs(VLModelArgs):
    model_path: str
    torch_dtype: str
    checkpoint_dir: Optional[str]
    max_length: int
    max_new_tokens: int
    reproducibility_config: dict

    @property
    def model_name(self) -> str:
        return self.model_path.split("/")[-1].replace("-", "_").replace(".", "")

    def make_model(self) -> LlamaModel:
        return LlamaModel(
            model_path=self.model_path,
            torch_dtype=self.torch_dtype,
            checkpoint_dir=self.checkpoint_dir,
            max_length=self.max_length,
            max_new_tokens=self.max_new_tokens,
            reproducibility_config=self.reproducibility_config,
        )

    def prepare(self):
        pass

    def close(self):
        pass

    def set_reproducibility_mode(self):
        self.reproducibility_config = {"do_sample": False}


class OpenRouterAPIModel(VLModel):
    def __init__(
        self,
        base_url: str,
        model_id: str,
        max_tokens: int,
        reproducibility_config: dict,
    ):
        self.client = AsyncOpenAI(base_url=base_url, api_key=os.getenv("OPENROUTER_API_KEY"))
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.reproducibility_config = reproducibility_config

    def __call__(self, messages: list[dict]) -> dict:
        @backoff.on_exception(backoff.expo, RateLimitError)
        async def get_response(messages: list[dict], max_tokens: int, **kwargs):
            completion = await self.client.chat.completions.create(
                model=self.model_id, messages=messages, max_tokens=max_tokens, **kwargs
            )
            try:
                response = AIMessage(completion.choices[0].message.content)
            except:
                response = AIMessage("")
            return response

        return asyncio.run(
            get_response(
                messages=messages, max_tokens=self.max_tokens, **self.reproducibility_config
            )
        )

    def get_stats(self) -> dict:
        return {}


@dataclass
class OpenRouterAPIModelArgs(VLModelArgs):
    model_id: str
    base_url: str
    max_tokens: int
    reproducibility_config: dict

    @property
    def model_name(self) -> str:
        return self.model_id.split("/")[-1].replace("-", "_").replace(".", "")

    def make_model(self) -> OpenRouterAPIModel:
        return OpenRouterAPIModel(
            model_id=self.model_id,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            reproducibility_config=self.reproducibility_config,
        )

    def prepare(self):
        pass

    def close(self):
        pass

    def set_reproducibility_mode(self):
        self.reproducibility_config = {"temperature": 0.0}
