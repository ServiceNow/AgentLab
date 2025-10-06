import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o")
tokenizer = tiktoken.get_encoding(encoder.name)


def get_chunks_from_tokenizer(axtree, chunk_size=200, overlap=50):
    all_text = tokenizer.encode(axtree)
    chunks = []
    for i in range(0, len(all_text), chunk_size - overlap):
        tokens = all_text[i : i + chunk_size]
        chunk = tokenizer.decode(tokens)
        chunks.append(chunk)
    return chunks
