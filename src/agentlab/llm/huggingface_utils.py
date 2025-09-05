import logging
import os
import time
from functools import partial
from typing import Any, List, Optional, Union

from pydantic import Field

from agentlab.llm.base_api import AbstractChatModel
from agentlab.llm.llm_utils import AIMessage, Discussion
from agentlab.llm.prompt_templates import PromptTemplate, get_prompt_template


class HFBaseChatModel(AbstractChatModel):
    """
    Custom LLM Chatbot that can interface with HuggingFace models with support for multiple samples.

    This class allows for the creation of a custom chatbot using models hosted
    on HuggingFace Hub or a local checkpoint. It provides flexibility in defining
    the temperature for response sampling and the maximum number of new tokens
    in the response.

    Attributes:
        llm (Any): The HuggingFaceHub model instance.
        prompt_template (Any): Template for the prompt to be used for the model's input sequence.
        tokenizer (Any): The tokenizer to use for the model.
        n_retry_server (int): Number of times to retry on server failure.
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

    def __init__(self, model_name, base_model_name, n_retry_server, log_probs):
        super().__init__()
        self.n_retry_server = n_retry_server
        self.log_probs = log_probs

        # Lazy import to avoid heavy transformers import when unused
        try:
            from transformers import AutoTokenizer, GPT2TokenizerFast  # type: ignore
        except Exception as e:  # pragma: no cover - surfaced only when transformers missing
            raise ImportError(
                "The 'transformers' package is required for HuggingFace models. Install it to use HF backends."
            ) from e

        if base_model_name is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if isinstance(self.tokenizer, GPT2TokenizerFast):
            logging.warning(
                f"No chat template is defined for {base_model_name}. Resolving to the hard-coded templates."
            )
            self.tokenizer = None
            self.prompt_template = get_prompt_template(model_name)

    def __call__(
        self,
        messages: list[dict],
        n_samples: int = 1,
        temperature: Optional[float] = None,
    ) -> Union[AIMessage, List[AIMessage]]:
        """
        Generate one or more responses for the given messages.

        Args:
            messages: List of message dictionaries containing the conversation history.
            n_samples: Number of independent responses to generate. Defaults to 1.
            temperature: The temperature for response sampling. Defaults to None.

        Returns:
            If n_samples=1, returns a single AIMessage.
            If n_samples>1, returns a list of AIMessages.

        Raises:
            Exception: If the server fails to respond after n_retry_server attempts or if the chat template fails.
        """
        if self.tokenizer:
            try:
                if isinstance(messages, Discussion):
                    messages.merge()
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception as e:
                if "Conversation roles must alternate" in str(e):
                    logging.warning(
                        "Failed to apply the chat template. Maybe because it doesn't support the 'system' role. "
                        "Retrying with the 'system' role appended to the 'user' role."
                    )
                    messages = _prepend_system_to_first_user(messages)
                    prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
                else:
                    raise e
        elif self.prompt_template:
            prompt = self.prompt_template.construct_prompt(messages)

        responses = []
        for _ in range(n_samples):
            itr = 0
            while True:
                try:
                    temperature = (
                        temperature
                        if temperature is not None
                        else getattr(self, "temperature", 0.1)
                    )
                    answer = self.llm(prompt, temperature=temperature)
                    response = AIMessage(answer)
                    if self.log_probs:
                        response["content"] = answer.generated_text
                        response["log_probs"] = answer.details
                    responses.append(response)
                    break
                except Exception as e:
                    if itr == self.n_retry_server - 1:
                        raise e
                    logging.warning(
                        f"Failed to get a response from the server: \n{e}\n"
                        f"Retrying... ({itr+1}/{self.n_retry_server})"
                    )
                    time.sleep(5)
                    itr += 1

        return responses[0] if n_samples == 1 else responses

    def _llm_type(self):
        return "huggingface"


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
                msg[text_key] = str(system_content) + "\n" + str(msg[text_key])
                # Remove the original system message
                del messages[system_index]
                break  # Ensures that only the first user message is modified

    return messages


class HuggingFaceURLChatModel(HFBaseChatModel):
    """HF backend using a Text Generation Inference (TGI) HTTP endpoint.

    This class is placed here to keep all heavy HF imports optional and only
    loaded when a HF backend is explicitly requested.
    """

    def __init__(
        self,
        model_name: str,
        model_url: str,
        base_model_name: Optional[str] = None,
        token: Optional[str] = None,
        temperature: Optional[float] = 1e-1,
        max_new_tokens: Optional[int] = 512,
        n_retry_server: Optional[int] = 4,
        log_probs: Optional[bool] = False,
    ):
        super().__init__(model_name, base_model_name, n_retry_server, log_probs)
        if temperature is not None and temperature < 1e-3:
            logging.warning("Models might behave weirdly when temperature is too low.")
        self.temperature = temperature

        if token is None:
            # support both env var names used elsewhere
            token = os.environ.get("TGI_TOKEN") or os.environ.get("AGENTLAB_MODEL_TOKEN")

        # Lazy import huggingface_hub here to avoid import on non-HF paths
        try:
            from huggingface_hub import InferenceClient  # type: ignore
        except Exception as e:  # pragma: no cover - surfaced only when package missing
            raise ImportError(
                "The 'huggingface_hub' package is required for HuggingFace URL backends."
            ) from e

        client = InferenceClient(model=model_url, token=token)
        self.llm = partial(
            client.text_generation,
            max_new_tokens=max_new_tokens,
            details=log_probs,
        )
