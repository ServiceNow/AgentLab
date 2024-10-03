import logging
import os
import time
from functools import partial
from typing import Any, List, Optional

from huggingface_hub import InferenceClient
from pydantic import Field
from transformers import AutoTokenizer, GPT2TokenizerFast

from agentlab.llm.prompt_templates import PromptTemplate, get_prompt_template


class HFBaseChatModel:
    """
    Custom LLM Chatbot that can interface with HuggingFace models.

    This class allows for the creation of a custom chatbot using models hosted
    on HuggingFace Hub or a local checkpoint. It provides flexibility in defining
    the temperature for response sampling and the maximum number of new tokens
    in the response.

    Attributes:
        llm (Any): The HuggingFaceHub model instance.
        prompt_template (Any): Template for the prompt to be used for the model's input sequence.
    """

    llm: Any = Field(description="The HuggingFaceHub model instance")
    tokenizer: Any = Field(
        default=None,
        description="The tokenizer to use for the model",
    )
    prompt_template: Optional[PromptTemplate] = Field(
        default=None,
        description="Template for the prompt to be used for the model's input sequence",
    )
    n_retry_server: int = Field(
        default=4,
        description="The number of times to retry the server if it fails to respond",
    )

    def __init__(self, model_name, n_retry_server):
        super().__init__()
        self.n_retry_server = n_retry_server

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if isinstance(self.tokenizer, GPT2TokenizerFast):
            logging.warning(
                f"No chat template is defined for {model_name}. Resolving to the hard-coded templates."
            )
            self.tokenizer = None
            self.prompt_template = get_prompt_template(model_name)

    def __call__(
        self,
        messages: list[dict],
    ) -> dict:

        # NOTE: The `stop`, `run_manager`, and `kwargs` arguments are ignored in this implementation.

        if self.tokenizer:
            # messages_formated = _convert_messages_to_dict(messages) ## ?
            try:
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception as e:
                if "Conversation roles must alternate" in str(e):
                    logging.warning(
                        f"Failed to apply the chat template. Maybe because it doesn't support the 'system' role"
                        "Retrying with the 'system' role appended to the 'user' role."
                    )
                    messages = _prepend_system_to_first_user(messages)
                    prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
                else:
                    raise e

        elif self.prompt_template:
            prompt = self.prompt_template.construct_prompt(messages)

        itr = 0
        while True:
            try:
                response = self.llm(prompt)
                return response
            except Exception as e:
                if itr == self.n_retry_server - 1:
                    raise e
                logging.warning(
                    f"Failed to get a response from the server: \n{e}\n"
                    f"Retrying... ({itr+1}/{self.n_retry_server})"
                )
                time.sleep(5)
                itr += 1

    def _llm_type(self):
        return "huggingface"

    def invoke(self, messages: list[dict]) -> dict:
        return self(messages)

    def get_stats(self):
        return {}


class HuggingFaceURLChatModel(HFBaseChatModel):
    def __init__(
        self,
        model_name: str,
        model_url: str,
        token: Optional[str] = None,
        temperature: Optional[int] = 1e-1,
        max_new_tokens: Optional[int] = 512,
        n_retry_server: Optional[int] = 4,
    ):
        super().__init__(model_name, n_retry_server)
        if temperature < 1e-3:
            logging.warning("Models might behave weirdly when temperature is too low.")

        if token is None:
            token = os.environ["TGI_TOKEN"]

        client = InferenceClient(model=model_url, token=token)
        self.llm = partial(
            client.text_generation, temperature=temperature, max_new_tokens=max_new_tokens
        )


# def _convert_messages_to_dict(messages, column_remap={}):
#     """
#     Converts a list of message objects into a list of dictionaries, categorizing each message by its role.

#     Each message is expected to be an instance of one of the following types: SystemMessage, HumanMessage, AIMessage.
#     The function maps each message to its corresponding role ('system', 'user', 'assistant') and formats it into a dictionary.

#     Args:
#         messages (list): A list of message objects.
#         column_remap (dict): A dictionary that maps the column names to the desired output format.

#     Returns:
#         list: A list of dictionaries where each dictionary represents a message and contains 'role' and 'content' keys.

#     Raises:
#         ValueError: If an unsupported message type is encountered.

#     Example:
#         >>> messages = [SystemMessage("System initializing..."), HumanMessage("Hello!"), AIMessage("How can I assist?")]
#         >>> _convert_messages_to_dict(messages)
#         [
#             {"role": "system", "content": "System initializing..."},
#             {"role": "user", "content": "Hello!"},
#             {"role": "assistant", "content": "How can I assist?"}
#         ]
#     """

#     human_key = column_remap.get("HumanMessage", "user")
#     ai_message_key = column_remap.get("AIMessage", "assistant")
#     role_key = column_remap.get("role", "role")
#     text_key = column_remap.get("text", "content")
#     image_key = column_remap.get("image", "media_url")

#     # Mapping of message types to roles
#     message_type_to_role = {
#         SystemMessage: "system",
#         HumanMessage: human_key,
#         AIMessage: ai_message_key,
#     }

#     def convert_format_vision(message_content, role, text_key, image_key):
#         result = {}
#         result["type"] = role
#         for item in message_content:
#             if item["type"] == "text":
#                 result[text_key] = item["text"]
#             elif item["type"] == "image_url":
#                 result[image_key] = item["image_url"]
#         return result

#     chat = []
#     for message in messages:
#         message_role = message_type_to_role.get(type(message))
#         if message_role:
#             if isinstance(message.content, str):
#                 chat.append({role_key: message_role, text_key: message.content})
#             else:
#                 chat.append(
#                     convert_format_vision(message.content, message_role, text_key, image_key)
#                 )
#         else:
#             raise ValueError(f"Message type {type(message)} not supported")

#     return chat


def _prepend_system_to_first_user(messages, column_remap={}):
    # Initialize an index for the system message
    system_index = None

    human_key = column_remap.get("HumanMessage", "user")
    role_key = column_remap.get("role", "role")
    text_key = column_remap.get("text", "content")

    # Find the system content and its index
    for i, msg in enumerate(messages):
        if msg[role_key] == "system":
            system_index = i
            system_content = msg[text_key]
            break  # Stop after finding the first system message

    # If a system message was found, modify the first user message and remove the system message
    if system_index is not None:
        for msg in messages:
            if msg[role_key] == human_key:
                # Prepend system content to the first user content
                msg[text_key] = system_content + "\n" + msg[text_key]
                # Remove the original system message
                del messages[system_index]
                break  # Ensures that only the first user message is modified

    return messages
