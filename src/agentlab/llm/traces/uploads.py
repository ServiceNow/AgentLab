from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
import pandas as pd
import zipfile
import os
from pathlib import Path

# Hugging Face dataset names
INDEX_DATASET = "/agent_traces_index"
TRACE_DATASET = "/agent_traces_data"

# Hugging Face API instance
api = HfApi()

def upload_index_data(index_df: pd.DataFrame):
    dataset = Dataset.from_pandas(index_df)
    dataset.push_to_hub(INDEX_DATASET, split="train")

def upload_trace(trace_file: str, exp_id: str):
    # Define the target zip file path
    zip_file_path = f"{exp_id}.zip"
    
    # Check if the file is already zipped
    if not trace_file.lower().endswith(".zip"):
        # Compress the file into a zip archive
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            trace_file_name = Path(trace_file).name
            zipf.write(trace_file, trace_file_name)
        print(f"Compressed {trace_file} to {zip_file_path}")
    else:
        zip_file_path = trace_file
        print(f"File {trace_file} is already a zip archive.")
    
    # Upload the (possibly newly created) zip file
    api.upload_file(
        path_or_fileobj=zip_file_path,
        path_in_repo=f"{exp_id}.zip",
        repo_id=TRACE_DATASET,
        repo_type="dataset",
    )

    # Optionally remove the temporary zip file if it was created during this process
    if zip_file_path != trace_file:
        os.remove(zip_file_path)
        print(f"Temporary zip file {zip_file_path} removed.")

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

