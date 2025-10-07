import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Union

import torch
from openai import AzureOpenAI, OpenAI


class RetrieverArgs:
    def make_model(self):
        pass


class Retriever(ABC):
    @abstractmethod
    def encode(self, text: Union[str, list[str]]):
        pass

    @abstractmethod
    def retrieve(self, query: str, chunks: Union[str, list[str]]):
        pass


@dataclass
class OpenAIRetrieverArgs:
    client: str = "openai"  # or "azure"
    model_name: str = "text-embedding-small-3"
    top_k: int = 5
    chunk_size: int = 100
    overlap: int = 10
    measure: Literal["cosine", "dot"] = "cosine"
    normalize_embeddings: bool = True
    use_recursive_text_splitter: bool = False

    def make_model(self) -> Retriever:
        return OpenAIRetriever(self)


class OpenAIRetriever(Retriever):
    def __init__(self, args: OpenAIRetrieverArgs):
        self.args = args
        self.model_name = args.model_name

        if args.client == "openai":
            self.client = OpenAI()
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("OPENAI_API_KEY environment variable is not set.")
        elif args.client == "azure":
            self.client = AzureOpenAI()
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            api_base = os.getenv("AZURE_OPENAI_API_BASE")
            if api_key is None or api_base is None:
                raise ValueError(
                    "AZURE_OPENAI_API_KEY or AZURE_OPENAI_API_BASE environment variable is not set."
                )

    def encode(self, text: Union[str, list[str]]):
        return self.client.embeddings.create(input=[text], model=self.model_name).data[0].embedding

    def retrieve(self, query: str, chunks: Union[str, list[str]]):
        def _normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
            return embeddings / embeddings.norm(dim=-1, keepdim=True)

        logging.debug(f"Encoding {len(chunks)} chunks...")
        chunks_embeddings = [torch.tensor(self.encode(chunk)) for chunk in chunks]
        logging.debug(f"Encoding query: {query}")
        query_embeddings = self.encode(query)
        chunks_embeddings = torch.stack(chunks_embeddings)
        query_embeddings = torch.tensor(query_embeddings)

        if self.args.normalize_embeddings:
            query_embeddings = _normalize_embeddings(query_embeddings)
            chunks_embeddings = _normalize_embeddings(chunks_embeddings)

        similarity_scores = torch.nn.functional.cosine_similarity(
            query_embeddings, chunks_embeddings
        )
        scores, indices = torch.topk(similarity_scores, k=min(self.args.top_k, len(chunks)))
        return scores, indices
