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

    @staticmethod
    def extract_bid(line):
        """
        Extracts the bid from a line in the format '[bid] textarea ...'.

        Parameters:
            line (str): The input line containing the bid in square brackets.

        Returns:
            str: The extracted bid, or None if no bid is found.
        """
        match = re.search(r"\[([a-zA-Z0-9]+)\]", line)
        if match:
            return match.group(1)
        return None

    @classmethod
    def get_elements_around(cls, tree, element_id, n):
        """
        Get n elements around the given element_id from the AXTree while preserving its indentation structure.

        :param tree: String representing the AXTree with indentations.
        :param element_id: The element ID to center around (can include alphanumeric IDs like 'a203').
        :param n: The number of elements to include before and after.
        :return: String of the AXTree elements around the given element ID, preserving indentation.
        """
        # Split the tree into lines
        lines = tree.splitlines()

        # Extract the line indices and content containing element IDs
        id_lines = [(i, line) for i, line in enumerate(lines) if "[" in line and "]" in line]

        # Parse the IDs from the lines
        parsed_ids = []
        for idx, line in id_lines:
            try:
                element_id_in_line = line.split("[")[1].split("]")[0]
                parsed_ids.append((idx, element_id_in_line, line))
            except IndexError:
                continue

        # Find the index of the element with the given ID
        target_idx = next(
            (i for i, (_, eid, _) in enumerate(parsed_ids) if eid == element_id), None
        )

        if target_idx is None:
            raise ValueError(f"Element ID {element_id} not found in the tree.")

        # Calculate the range of elements to include
        start_idx = max(0, target_idx - n)
        end_idx = min(len(parsed_ids), target_idx + n + 1)

        # Collect the lines to return
        result_lines = []
        for idx in range(start_idx, end_idx):
            line_idx = parsed_ids[idx][0]
            result_lines.append(lines[line_idx])

        return "\n".join(result_lines)
