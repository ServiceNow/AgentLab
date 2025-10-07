import re
from dataclasses import dataclass

try:
    import bm25s
except ImportError:
    raise ImportError("bm25s is not installed. Please install it using `pip agentlab[retrievers]`.")
import tiktoken  # Added import for tiktoken

from .utils import get_chunks_from_tokenizer


def count_tokens(text: str) -> int:
    """Count the number of tokens in the text using tiktoken for GPT-4."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(text)
    return len(tokens)


@dataclass
class BM25RetrieverArgs:
    chunk_size: int = 100
    overlap: int = 10
    top_k: int = 5
    use_recursive_text_splitter: bool = False


class BM25SRetriever:
    """Simple retriever using BM25S to retrieve the most relevant lines"""

    def __init__(
        self,
        tree: str,
        chunk_size: int,
        overlap: int,
        top_k: int,
        use_recursive_text_splitter: bool,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.use_recursive_text_splitter = use_recursive_text_splitter
        corpus = get_chunks_from_tokenizer(tree)
        self.retriever = bm25s.BM25(corpus=corpus)
        tokenized_corpus = bm25s.tokenize(corpus)
        self.retriever.index(tokenized_corpus)

    def retrieve(self, query):
        tokenized_query = bm25s.tokenize(query)
        if self.top_k > len(self.retriever.corpus):
            results, _ = self.retriever.retrieve(
                query_tokens=tokenized_query, k=len(self.retriever.corpus)
            )
        else:
            results, _ = self.retriever.retrieve(query_tokens=tokenized_query, k=self.top_k)
        return [str(res) for res in results[0]]

    def create_text_chunks(self, axtree, chunk_size=200, overlap=50):
        if self.use_recursive_text_splitter:
            try:
                from langchain.text_splitter import (
                    RecursiveCharacterTextSplitter,
                )
            except ImportError:
                raise ImportError(
                    "langchain is not installed. Please install it using `pip agentlab[retrievers]`."
                )

            text_splitter = RecursiveCharacterTextSplitter()
            return text_splitter.split_text(axtree)
        else:
            return get_chunks_from_tokenizer(axtree, self.chunk_size, self.overlap)
