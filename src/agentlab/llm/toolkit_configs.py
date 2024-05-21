# TODO: move me once we have a better place for configs

import logging
import os
import subprocess

import yaml

## team-wide Tooklit variables

# these three could be different for you if you've modified you ~/.research-interactive-env
UI_COPILOT_DATA_PATH = "/mnt/ui_copilot/data"
UI_COPILOT_DATA_RW_PATH = "/mnt/ui_copilot/data_rw"
RESULTS_PATH = "/mnt/ui_copilot/results/"

TOOLKIT_DATA_STORE = "snow.research_ui_copilot.data"
TGI_IMAGE_LLMD = "ghcr.io/huggingface/text-generation-inference:sha-630800e"
TGI_IMAGE_OFFICIAL = "ghcr.io/huggingface/text-generation-inference:sha-59b3ffe"
TGI_IMAGE_LATEST = "ghcr.io/huggingface/text-generation-inference:latest"
UI_COPILOT_DATA_VOLUME = "snow.research_ui_copilot.data"
RESULTS_VOLUME = "snow.research_ui_copilot.results"
BASE_MODELS_PATH = f"{UI_COPILOT_DATA_PATH}/huggingface_hub_cache"
BASE_MODELS_RW_PATH = f"{UI_COPILOT_DATA_RW_PATH}/huggingface_hub_cache"
FINETUNING_PATH = f"{UI_COPILOT_DATA_PATH}/finetuning"
FINETUNING_RW_PATH = f"{UI_COPILOT_DATA_RW_PATH}/finetuning"
FINETUNING_CKPT_PATH = f"{FINETUNING_PATH}/checkpoints"
FINETUNING_CKPT_RW_PATH = f"{FINETUNING_RW_PATH}/checkpoints"
FINETUNING_DATASETS_PATH = f"{FINETUNING_PATH}/datasets"
FINETUNING_DATASETS_RW_PATH = f"{FINETUNING_RW_PATH}/datasets"
FINETUNING_TRAINING_DATASETS_PATH = f"{FINETUNING_DATASETS_PATH}/training_datasets"
FINETUNING_TRAINING_DATASETS_RW_PATH = f"{FINETUNING_DATASETS_RW_PATH}/training_datasets"

# Create a dictionary with the variables and their values
variables = {
    "TOOLKIT_DATA_STORE": TOOLKIT_DATA_STORE,
    "TGI_IMAGE_LLMD": TGI_IMAGE_LLMD,
    "TGI_IMAGE_OFFICIAL": TGI_IMAGE_OFFICIAL,
    "UI_COPILOT_DATA_VOLUME": UI_COPILOT_DATA_VOLUME,
    "UI_COPILOT_DATA_PATH": UI_COPILOT_DATA_PATH,
    "UI_COPILOT_DATA_RW_PATH": UI_COPILOT_DATA_RW_PATH,
    "RESULTS_VOLUME": RESULTS_VOLUME,
    "RESULTS_PATH": RESULTS_PATH,
    "BASE_MODELS_PATH": BASE_MODELS_PATH,
    "BASE_MODELS_RW_PATH": BASE_MODELS_RW_PATH,
    "FINETUNING_PATH": FINETUNING_PATH,
    "FINETUNING_RW_PATH": FINETUNING_RW_PATH,
    "FINETUNING_CKPT_PATH": FINETUNING_CKPT_PATH,
    "FINETUNING_CKPT_RW_PATH": FINETUNING_CKPT_RW_PATH,
    "FINETUNING_DATASETS_PATH": FINETUNING_DATASETS_PATH,
    "FINETUNING_DATASETS_RW_PATH": FINETUNING_DATASETS_RW_PATH,
    "FINETUNING_TRAINING_DATASETS_PATH": FINETUNING_TRAINING_DATASETS_PATH,
    "FINETUNING_TRAINING_DATASETS_RW_PATH": FINETUNING_TRAINING_DATASETS_RW_PATH,
}


# Add variables to bashrc file
# TODO: edge case: if the variable is set to a different value, we should update it
def add_to_bashrc(var, value):
    rc_file = os.path.expanduser("~/.bashrc")
    if not os.path.exists(rc_file):
        rc_file = os.path.expanduser("~/.zshrc")
    if os.path.exists(rc_file):
        with open(rc_file, "r") as file:
            if f"export {var}={value}" not in file.read():
                command = f'echo "export {var}={value}" >> {rc_file}'
                subprocess.run(command, shell=True)
    else:
        print(
            "Error: Neither ~/.bashrc nor ~/.zshrc file exists. Your environment variables will not be saved."
        )


for var, value in variables.items():
    if not os.environ.get(var):
        add_to_bashrc(var, value)


## user specific variables you need to set if you want to use TGI servers
print("Automatically setting your eai account")
command = "eai account get --format json"
try:
    result = subprocess.run(command, shell=True, capture_output=True)
    ACCOUNT_NAME = yaml.safe_load(result.stdout)["fullName"]
except:
    print(
        "Error: Could not get your eai account. You won't be able to automatically lauch OSS LLMs"
    )


def set_env_var_if_unset(var_name, value):
    """
    Set an environment variable to a given value if it is not already set.

    Parameters:
    var_name (str): The name of the environment variable.
    value (str): The value to set the environment variable to if it is not already set.
    """
    if os.getenv(var_name) is None:
        os.environ[var_name] = value
        print(
            f"Warning: The environment variable '{var_name}' was not set and has been set to '{value}'."
        )
    else:
        print(f"The environment variable '{var_name}' is already set.")


## WorkArena
SNOW_INSTANCE_URL = "https://dev260540.service-now.com/"
SNOW_INSTANCE_UNAME = "admin"
# NOTE use a raw string to avoid escaping special characters
SNOW_INSTANCE_PWD = r"xx^DUF7Pu!z9"

## WebArena
SERVER_HOSTNAME = "ec2-3-21-46-179.us-east-2.compute.amazonaws.com"

config = {
    # WorkArena
    "SNOW_INSTANCE_URL": SNOW_INSTANCE_URL,
    "SNOW_INSTANCE_UNAME": SNOW_INSTANCE_UNAME,
    "SNOW_INSTANCE_PWD": SNOW_INSTANCE_PWD,
    # WebArena w/ Toolkit ports
    "BASE_URL": f"http://{SERVER_HOSTNAME}",
    "SHOPPING": f"{SERVER_HOSTNAME}:3306/",
    "SHOPPING_ADMIN": f"{SERVER_HOSTNAME}:7565/admin",
    "REDDIT": f"{SERVER_HOSTNAME}:9001",
    "GITLAB": f"{SERVER_HOSTNAME}:8080",
    "WIKIPEDIA": f"{SERVER_HOSTNAME}:80/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
    "MAP": f"{SERVER_HOSTNAME}:22",
    "HOMEPAGE": f"{SERVER_HOSTNAME}:42022",
}

for key, value in config.items():
    set_env_var_if_unset(key, value)
