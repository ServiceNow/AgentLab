from datasets import load_dataset
import requests

# Hugging Face dataset name for the index
INDEX_DATASET = "your_username/agent_traces_index"

# Function to query traces based on LLM and benchmark
def query_traces(llm=None, benchmark=None):
    dataset = load_dataset(INDEX_DATASET, split="train")
    df = dataset.to_pandas()

    if llm:
        df = df[df["llm"] == llm]
    if benchmark:
        df = df[df["benchmark"] == benchmark]

    return df[["exp_id", "study_name", "trace_pointer"]].to_dict(orient="records")

# Function to download a trace based on exp_id
def download_trace(exp_id: str, save_path: str):
    dataset = load_dataset(INDEX_DATASET, split="train")
    df = dataset.to_pandas()
    trace_url = df[df["exp_id"] == exp_id]["trace_pointer"].values[0]

    response = requests.get(trace_url)
    with open(save_path, "wb") as f:
        f.write(response.content)
    print(f"Downloaded trace {exp_id} to {save_path}")

