from agentlab.llm.llm_utils import AIMessage, Discussion
from dataclasses import dataclass
from openai import AsyncOpenAI, RateLimitError
from .base import VLModel, VLModelArgs
import asyncio
import backoff
import os


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

    def __call__(self, messages: Discussion) -> AIMessage:
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
    base_url: str
    model_id: str
    max_tokens: int
    reproducibility_config: dict

    @property
    def model_name(self) -> str:
        return self.model_id.split("/")[-1].replace("-", "_").replace(".", "")

    def make_model(self) -> OpenRouterAPIModel:
        return OpenRouterAPIModel(
            base_url=self.base_url,
            model_id=self.model_id,
            max_tokens=self.max_tokens,
            reproducibility_config=self.reproducibility_config,
        )

    def prepare(self):
        pass

    def close(self):
        pass

    def set_reproducibility_mode(self):
        self.reproducibility_config = {"temperature": 0.0}
