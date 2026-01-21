from datasets import load_dataset
from typing import List
import os

HF_TOKEN = os.getenv('HF_TOKEN')
HF_REPO_NAME = os.getenv('HF_REPO_NAME')


def query_traces_by_llm_and_benchmark(llm: str, benchmark: str) -> List[dict]:
    """
    Query traces based on the provided LLM and benchmark.
    :param llm: The name of the LLM (e.g., 'GPT-4').
    :param benchmark: The benchmark name (e.g., 'benchmark1').
    :return: A list of trace metadata dictionaries.
    """
    INDEX_DATASET = f"{HF_REPO_NAME}/agent_traces_index"

    try:
        dataset = load_dataset(INDEX_DATASET, use_auth_token=HF_TOKEN, split='train')
        results = [
            {
                'exp_id': row['exp_id'],
                'study_id': row['study_id'],
                'llm': row['llm'],
                'benchmark': row['benchmark'],
                'trace_pointer': row['trace_pointer']
            }
            for row in dataset
            if row['llm'] == llm and row['benchmark'] == benchmark
        ]
        return results
    except Exception as e:
        print(f"Error querying traces for LLM '{llm}' and benchmark '{benchmark}': {e}")
        return []


def download_trace_by_experiment_id(exp_id: str, output_dir: str) -> None:
    """
    Download the trace file based on the experiment ID.
    :param exp_id: The ID of the experiment whose trace file needs to be downloaded.
    :param output_dir: The directory where the trace file will be saved.
    """
    TRACE_DATASET = f"{HF_REPO_NAME}/agent_traces_data"

    try:
        dataset = load_dataset(TRACE_DATASET, use_auth_token=HF_TOKEN, split='train')
        for row in dataset:
            if row['exp_id'] == exp_id:
                trace_file = row['zip_file']
                output_path = os.path.join(output_dir, trace_file)
                dataset.download_and_prepare()
                dataset.to_csv(output_path)
                print(f"Trace file for experiment '{exp_id}' downloaded to {output_path}.")
                return
        print(f"Experiment ID '{exp_id}' not found in the dataset.")
    except Exception as e:
        print(f"Error downloading trace file for experiment '{exp_id}': {e}")

