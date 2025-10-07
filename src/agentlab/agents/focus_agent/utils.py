import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o")


def get_nb_tokens(text):
    return len(encoder.encode(text))


def add_line_numbers_to_tree(axtree_txt: str) -> str:
    """
    Adds line numbers to the tree text.
    """
    lines = axtree_txt.strip().splitlines()
    numbered_lines = [f"{i+1:>4} {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered_lines)


def remove_no_bid_lines(axtree_txt: str) -> str:
    """
    Cleans the AXTree text by removing elements with no bid
    """
    lines = axtree_txt.splitlines()[1:]  # skip the first line
    cleaned_lines = [line for line in lines if "[" and "]" in line]
    cleaned_lines.insert(0, axtree_txt.splitlines()[0])  # add back the first line
    return "\n".join(cleaned_lines)
