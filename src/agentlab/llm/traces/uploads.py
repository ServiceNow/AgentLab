from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
import pandas as pd

# Hugging Face dataset names
INDEX_DATASET = "/agent_traces_index"
TRACE_DATASET = "/agent_traces_data"

# Hugging Face API instance
api = HfApi()

def upload_index_data(index_df: pd.DataFrame):
    dataset = Dataset.from_pandas(index_df)
    dataset.push_to_hub(INDEX_DATASET, split="train")

def upload_trace(trace_file: str, exp_id: str):
    api.upload_file(
        path_or_fileobj=trace_file,
        path_in_repo=f"{exp_id}.zip",
        repo_id=TRACE_DATASET,
        repo_type="dataset",
    )

def add_study(exp_id: str, study_name: str, llm: str, benchmark: str, trace_file: str):
    # Check if the benchmark is whitelisted
    WHITELISTED_BENCHMARKS = ["benchmark1", "benchmark2"]
    if benchmark not in WHITELISTED_BENCHMARKS:
        raise ValueError("Benchmark not whitelisted")

    # Assign a license based on LLM and benchmark
    LICENSES = {
        ("GPT-4", "benchmark1"): "MIT",
        ("Llama2", "benchmark2"): "Apache-2.0",
    }
    license_type = LICENSES.get((llm, benchmark), "Unknown")

    # Upload trace file
    upload_trace(trace_file, exp_id)

    # Create metadata entry
    index_entry = {
        "exp_id": exp_id,
        "study_name": study_name,
        "llm": llm,
        "benchmark": benchmark,
        "license": license_type,
        "trace_pointer": f"https://huggingface.co/datasets/{TRACE_DATASET}/resolve/main/{exp_id}.zip",
    }

    # Load the existing index dataset and add new entry
    dataset = load_dataset(INDEX_DATASET, split="train")
    df = dataset.to_pandas()
    df = df.append(index_entry, ignore_index=True)
    upload_index_data(df)

    print(f"Study {exp_id} added successfully!")

