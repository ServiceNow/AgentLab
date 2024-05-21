import os
from huggingface_hub import InferenceClient
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

# TODO: where should we set this? can we all use the same?
# NOTE: don't forget to grant permission to the job
# TGI_TOKEN = "42"
# os.environ["TGI_TOKEN"] = TGI_TOKEN


def tgi_check_servers(model_url=None):
    """
    Test models from Hugging Face Hub using a specified prompt.

    Parameters:
        model_url (str, optional): Specific model URL to use for inference; overrides the default loop.
    """
    # Determine the model(s) to use
    if model_url:
        models = {"custom_model": model_url}
    else:
        models = {
            k: v.model_url for k, v in CHAT_MODEL_ARGS_DICT.items() if v.model_url is not None
        }

    prompt = "What is your favorite programming language and is it Visual Basic?"

    print(f"\nPrompt: {prompt}\n")

    # Iterate through each model and make inference requests
    for model_name, model_endpoint in models.items():
        # Creating the client instance
        client = InferenceClient(model=model_endpoint, token=os.environ["TGI_TOKEN"])

        # Making the inference request
        result = client.text_generation(prompt=prompt, max_new_tokens=20)

        # Format and display the result
        result = result.replace("\n", "")
        print(f"{model_name}: {result}\n")


# dunder main
if __name__ == "__main__":

    specific_model_url = "https://94a486f7-3917-40e0-b97c-3edc07afcb8f.job.console.elementai.com"
    # specific_model_url = None

    tgi_check_servers(model_url=specific_model_url)
