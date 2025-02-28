import os
from datasets import Dataset, load_dataset, concatenate_datasets
import zipfile

# Retrieve environment variables
hf_repo_name = os.getenv('HF_REPO_NAME')
hf_token = os.getenv('HF_TOKEN')

def upload_study(study_details: dict):
    """
    Uploads study details to the Hugging Face `STUDY_DATASET`.

    Args:
        study_details (dict): A dictionary containing study_id, study_name, and description.
    """
    
    if not hf_repo_name or not hf_token:
        raise ValueError("HF_REPO_NAME and HF_TOKEN environment variables must be set.")
    
    study_dataset = f"{hf_repo_name}/agent_traces_study"
    
    # Load existing dataset or create a new one
    try:
        dataset = load_dataset(study_dataset, split='train', token=hf_token)
        existing_data = dataset.to_dict()
    except Exception as e:
        print(f"Could not load existing dataset: {e}. Creating a new dataset.")
        existing_data = None
    
    # Create a new dataset with the new study details
    new_data = Dataset.from_dict({
        'study_id': [study_details.get('study_id')],
        'study_name': [study_details.get('study_name')],
        'description': [study_details.get('description')]
    })
    
    # Concatenate with existing data if available
    if existing_data:
        existing_dataset = Dataset.from_dict(existing_data)
        combined_data = concatenate_datasets([existing_dataset, new_data])
    else:
        combined_data = new_data
    
    # Push updated dataset to the Hugging Face Hub
    try:
        combined_data.push_to_hub(study_dataset, token=hf_token)
        print("Study details uploaded successfully!")
    except Exception as e:
        print(f"Failed to upload study details: {e}")

def upload_trace(exp_id: str, directory: str, benchmark: str) -> str:
    """
    Compresses a directory into a zip file, uploads it to the TRACE_DATASET on Hugging Face,
    and returns the URL of the uploaded file.

    Args:
        exp_id (str): The experiment ID associated with this trace.
        directory (str): The path to the directory to compress.
        benchmark (str): The benchmark name, which must be whitelisted.

    Returns:
        str: The URL of the uploaded zip file in the dataset.
    """
    # Check if the benchmark is whitelisted
    WHITELISTED_BENCHMARKS = ["benchmark1", "benchmark2"]
    if benchmark not in WHITELISTED_BENCHMARKS:
        raise ValueError("Benchmark not whitelisted")

    if not hf_repo_name or not hf_token:
        raise ValueError("HF_REPO_NAME and HF_TOKEN environment variables must be set.")
    
    trace_dataset = f"{hf_repo_name}/agent_traces_data"
    
    # Create a zip file from the directory
    zip_filename = f"{exp_id}.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, directory))
    
    print(f"Directory '{directory}' compressed into '{zip_filename}'.")
    
    # Load existing dataset or create a new one
    try:
        dataset = load_dataset(trace_dataset, use_auth_token=hf_token, split='train')
        existing_data = {
            'exp_id': dataset['exp_id'],
            'zip_file': dataset['zip_file']
        }
    except Exception as e:
        print(f"Could not load existing dataset: {e}. Creating a new dataset.")
        existing_data = None
    
    # Create a new dataset with the new experiment trace
    new_data = Dataset.from_dict({
        'exp_id': [exp_id],
        'zip_file': [zip_filename]
    })
    
    # Concatenate with existing data if available
    if existing_data:
        existing_dataset = Dataset.from_dict(existing_data)
        combined_data = concatenate_datasets([existing_dataset, new_data])
    else:
        combined_data = new_data
    
    # Push updated dataset to the Hugging Face Hub
    combined_data.push_to_hub(trace_dataset, token=hf_token)
    print("Experiment trace uploaded successfully!")
    
    # Clean up the local zip file
    os.remove(zip_filename)
    print(f"Temporary zip file '{zip_filename}' removed.")
    
    # Construct and return the file URL on the Hugging Face Hub
    file_url = f"https://huggingface.co/datasets/{trace_dataset}/resolve/main/{zip_filename}"
    print(f"File URL: {file_url}")
    
    return file_url

def update_index(exp_id: str, study_id: str, llm: str, benchmark: str, license: str, trace_pointer: str):
    """
    Adds a record to the INDEX_DATASET on Hugging Face with the given experiment details.

    Args:
        exp_id (str): The experiment ID.
        study_id (str): The study ID.
        llm (str): The name of the large language model used.
        benchmark (str): The benchmark used for evaluation.
        license (str): The license type for the experiment.
        trace_pointer (str): The URL pointer to the trace data (must be a valid HTTPS URL).
    """
        
    if not hf_repo_name or not hf_token:
        raise ValueError("HF_REPO_NAME and HF_TOKEN environment variables must be set.")
    
    index_dataset = f"{hf_repo_name}/agent_traces_index"
    
    # Load existing dataset or create a new one
    try:
        dataset = load_dataset(index_dataset, use_auth_token=hf_token, split='train')
        existing_data = {
            'exp_id': dataset['exp_id'],
            'study_id': dataset['study_id'],
            'llm': dataset['llm'],
            'benchmark': dataset['benchmark'],
            'license': dataset['license'],
            'trace_pointer': dataset['trace_pointer']
        }
    except Exception as e:
        print(f"Could not load existing dataset: {e}. Creating a new dataset.")
        existing_data = None
    
    # Create a new dataset with the provided index details
    new_data = Dataset.from_dict({
        'exp_id': [exp_id],
        'study_id': [study_id],
        'llm': [llm],
        'benchmark': [benchmark],
        'license': [license],
        'trace_pointer': [trace_pointer]
    })
    
    # Concatenate with existing data if available
    if existing_data:
        existing_dataset = Dataset.from_dict(existing_data)
        combined_data = concatenate_datasets([existing_dataset, new_data])
    else:
        combined_data = new_data
    
    # Push updated dataset to the Hugging Face Hub
    combined_data.push_to_hub(index_dataset, token=hf_token)
    print("Index dataset updated successfully!")

