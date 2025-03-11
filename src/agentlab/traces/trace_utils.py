import os
import zipfile

from bgym import ExpArgs, ExpResult
from datasets import Dataset, concatenate_datasets, load_dataset

from agentlab.experiments.study import Study

# Retrieve environment variables
hf_user_name = os.getenv("HF_REPO_NAME")
hf_token = os.getenv("HF_TOKEN")


def upload_study(study: Study):
    """
    Uploads study details to the Hugging Face `STUDY_DATASET`.

    Args:
        study (Study): The study object containing the experiment details.
    """

    if not hf_user_name or not hf_token:
        raise ValueError("HF_REPO_NAME and HF_TOKEN environment variables must be set.")

    study_dataset = f"{hf_user_name}/agent_traces_study"

    # Load existing dataset or create a new one
    try:
        dataset = load_dataset(study_dataset, split="train", token=hf_token)
        existing_data = dataset.to_dict()
        headers = dataset.column_names
    except Exception as e:
        print(f"Could not load existing dataset: {e}. Creating a new dataset.")
        existing_data = None
        headers = None

    # Create a new dataset with the new study details
    entries = study.get_journal_entries(
        strict_reproducibility=False, headers=headers
    )  # type: list[list]
    if headers is None:
        headers = entries[0]

    entries = entries[1:]
    entries = list(zip(*entries))
    new_data = Dataset.from_dict({header: entries[i] for i, header in enumerate(headers)})

    # Concatenate with existing data if available
    if existing_data:
        existing_dataset = Dataset.from_dict(existing_data)
        combined_data = concatenate_datasets([existing_dataset, new_data])
    else:
        combined_data = new_data

    # Push updated dataset to the Hugging Face Hub
    try:
        combined_data.push_to_hub(study_dataset, token=hf_token, create_pr=True)
        print("Study details uploaded successfully!")
    except Exception as e:
        print(f"Failed to upload study details: {e}")


def upload_trace(exp_args: ExpArgs) -> str:
    """
    Compresses a directory into a zip file, uploads it to the TRACE_DATASET on Hugging Face,
    and returns the URL of the uploaded file.

    Args:
        exp_args (ExpArgs): The experiment arguments.

    Returns:
        str: The URL of the uploaded zip file in the dataset.
    """
    # # Check if the benchmark is whitelisted
    # WHITELISTED_BENCHMARKS = ["benchmark1", "benchmark2"]
    # if benchmark not in WHITELISTED_BENCHMARKS:
    #     raise ValueError("Benchmark not whitelisted")

    if not hf_user_name or not hf_token:
        raise ValueError("HF_REPO_NAME and HF_TOKEN environment variables must be set.")

    trace_dataset = f"{hf_user_name}/agent_traces_data"

    # Create a zip file from the directory
    zip_filename = f"{exp_args.exp_id}.zip"
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(exp_args.exp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, exp_args.exp_dir))

    print(f"Directory '{exp_args}' compressed into '{zip_filename}'.")

    # Load existing dataset or create a new one
    try:
        dataset = load_dataset(trace_dataset, use_auth_token=hf_token, split="train")
        existing_data = {"exp_id": dataset["exp_id"], "zip_file": dataset["zip_file"]}
    except Exception as e:
        print(f"Could not load existing dataset: {e}. Creating a new dataset.")
        existing_data = None

    # Create a new dataset with the new experiment trace
    new_data = Dataset.from_dict({"exp_id": [exp_args.exp_id], "zip_file": [zip_filename]})

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


def update_index(
    exp_results: list[ExpResult], study_id: str, license: str, trace_pointers: list[str]
):
    """
    Adds a record to the INDEX_DATASET on Hugging Face with the given experiment details.

    Args:
        exp_args (ExpArgs): The experiment arguments.
        study_id (str): The study ID associated with this experiment.
        license (str): The license type for the experiment trace.
        trace_pointer (str): The URL of the experiment trace file.
    """

    if not hf_user_name or not hf_token:
        raise ValueError("HF_REPO_NAME and HF_TOKEN environment variables must be set.")

    index_dataset = f"{hf_user_name}/agent_traces_index"

    # Load existing dataset or create a new one
    try:
        dataset = load_dataset(index_dataset, use_auth_token=hf_token, split="train")
        existing_data = dataset.to_dict()
    except Exception as e:
        print(f"Could not load existing dataset: {e}. Creating a new dataset.")
        existing_data = None

    # Create a new dataset with the provided index details

    dataset = [exp_res.get_exp_record() for exp_res in exp_results]
    for el in dataset:
        el.pop("exp_dir")

    # list[dict] -> dict[list]
    new_data = {key: [d[key] for d in dataset] for key in dataset[0].keys()}
    new_data["study_id"] = [study_id.hex] * len(exp_results)
    new_data["license"] = [license] * len(exp_results)
    new_data["trace_pointer"] = trace_pointers

    new_data = Dataset.from_dict(new_data)

    # Concatenate with existing data if available
    if existing_data:
        existing_dataset = Dataset.from_dict(existing_data)
        combined_data = concatenate_datasets([existing_dataset, new_data])
    else:
        combined_data = new_data

    # Push updated dataset to the Hugging Face Hub
    combined_data.push_to_hub(index_dataset, token=hf_token, create_pr=True)
    print("Index dataset updated successfully!")


if __name__ == "__main__":
    import os
    import pathlib

    from agentlab.experiments.study import Study
    from agentlab.traces.trace_utils import update_index, upload_study

    path = pathlib.Path("/path/to/study")

    study = Study.load(path)
    study.load_exp_args_list()

    upload_study(study)
    update_index(study.exp_args_list, study.uuid, "open", ["w/e"] * len(study.exp_args_list))
