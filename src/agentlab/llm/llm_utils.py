import base64
import collections
import importlib
import io
import json
import logging
import os
import re
import time
from copy import deepcopy
from functools import cache
from typing import TYPE_CHECKING, Any, Union
from warnings import warn

import anthropic
import numpy as np
import openai
import tiktoken
import yaml
from PIL import Image

langchain_community = importlib.util.find_spec("langchain_community")
if langchain_community is not None:
    from langchain.schema import BaseMessage as LangchainBaseMessage
    from langchain_community.adapters.openai import convert_message_to_dict
else:
    LangchainBaseMessage = None
    convert_message_to_dict = None

if TYPE_CHECKING:
    from agentlab.llm.chat_api import ChatModel


def messages_to_dict(messages: list[dict] | list[LangchainBaseMessage]) -> dict:
    new_messages = Discussion()
    for m in messages:
        if isinstance(m, dict):
            new_messages.add_message(m)
        elif isinstance(m, str):
            new_messages.add_message({"role": "<unknown role>", "content": m})
        elif LangchainBaseMessage is not None and isinstance(m, LangchainBaseMessage):
            new_messages.add_message(convert_message_to_dict(m))
        else:
            raise ValueError(f"Unknown message type: {type(m)}")
    return new_messages


class RetryError(ValueError):
    pass


def retry(
    chat: "ChatModel",
    messages: "Discussion",
    n_retry: int,
    parser: callable,
    log: bool = True,
):
    """Retry querying the chat models with the response from the parser until it
    returns a valid value.

    If the answer is not valid, it will retry and append to the chat the  retry
    message.  It will stop after `n_retry`.

    Note, each retry has to resend the whole prompt to the API. This can be slow
    and expensive.

    Args:
        chat (ChatModel): a ChatModel object taking a list of messages and
            returning a list of answers, all in OpenAI format.
        messages (list): the list of messages so far. This list will be modified with
            the new messages and the retry messages.
        n_retry (int): the maximum number of sequential retries.
        parser (callable): a function taking a message and retruning a parsed value,
            or raising a ParseError
        log (bool): whether to log the retry messages.

    Returns:
        dict: the parsed value, with a string at key "action".

    Raises:
        ParseError: if the parser could not parse the response after n_retry retries.
    """
    tries = 0
    while tries < n_retry:
        answer = chat(messages)
        # TODO: could we change this to not use inplace modifications ?
        messages.append(answer)
        try:
            return parser(answer["content"])
        except ParseError as parsing_error:
            tries += 1
            if log:
                msg = f"Query failed. Retrying {tries}/{n_retry}.\n[LLM]:\n{answer['content']}\n[User]:\n{str(parsing_error)}"
                logging.info(msg)
            messages.append(dict(role="user", content=str(parsing_error)))

    raise ParseError(f"Could not parse a valid value after {n_retry} retries.")


def generic_call_api_with_retries(
    client_function,
    api_params,
    is_response_valid_fn,
    rate_limit_exceptions,
    api_error_exceptions,
    get_status_code_fn=None,
    max_retries=10,
    initial_retry_delay_seconds=20,
    max_retry_delay_seconds=60 * 5,
):
    """
    Makes an API call with retries for transient failures, rate limiting,
    and responses deemed invalid by a custom validation function.
    (Refactored for improved readability with helper functions)

    Args:
        client_function: The API client function to call.
        api_params: Parameters to pass to the client function.
        is_response_valid_fn: Function to validate if the response is valid.
        rate_limit_exceptions: Tuple of exception types for rate limiting.
        api_error_exceptions: Tuple of exception types for API errors.
        get_status_code_fn: Optional function to extract status code from exceptions.
        max_retries: Maximum number of retry attempts.
        initial_retry_delay_seconds: Initial delay between retries in seconds.
        max_retry_delay_seconds: Maximum delay between retries in seconds.

    Returns:
        The API response if successful.

    Raises:
        Exception: For unexpected errors that are immediately re-raised.
        RuntimeError: If API call fails after maximum retries.
    """

    def _calculate_delay(
        current_attempt, initial_delay, max_delay, is_first_attempt_for_type=False
    ):
        """Calculates exponential backoff delay."""
        # For invalid response content (not an exception), the first "attempt" at retrying this specific issue
        # might use a slightly different delay calculation if desired (e.g. attempt-1 for the exponent).
        # For exceptions, the attempt number directly applies.
        # Here, we use 'current_attempt' for exception-driven retries,
        # and 'current_attempt -1' for the first retry due to invalid content (is_first_attempt_for_type).
        if is_first_attempt_for_type:  # First retry due to invalid content
            # The first retry after an invalid response (attempt 1 for this *type* of failure)
            effective_attempt = current_attempt - 1  # Use 0 for the first exponent
        else:  # Retries due to exceptions or subsequent invalid content retries
            effective_attempt = current_attempt  # Use current_attempt for exponent

        # Ensure effective_attempt for exponent is at least 0
        exponent_attempt = max(
            0, effective_attempt if not is_first_attempt_for_type else current_attempt - 1
        )

        return min(initial_delay * (2**exponent_attempt), max_delay)

    def _handle_invalid_response_content(attempt):
        logging.warning(
            f"[Attempt {attempt}/{max_retries}] API response deemed invalid by validation function. Retrying after delay..."
        )
        if attempt < max_retries:
            # For the first retry due to invalid content, use attempt-1 for exponent
            delay = _calculate_delay(
                attempt,
                initial_retry_delay_seconds,
                max_retry_delay_seconds,
                is_first_attempt_for_type=True,
            )
            logging.debug(f"Sleeping for {delay:.2f} seconds due to invalid response content.")
            time.sleep(delay)
            return True  # Indicate retry
        return False  # Max retries reached for this path

    def _handle_rate_limit_error(e, attempt):
        logging.warning(
            f"[Attempt {attempt}/{max_retries}] Rate limit error: {e}. Retrying after delay..."
        )
        if attempt < max_retries:
            delay = _calculate_delay(attempt, initial_retry_delay_seconds, max_retry_delay_seconds)
            logging.debug(f"Sleeping for {delay:.2f} seconds due to rate limit.")
            time.sleep(delay)
            return True  # Indicate retry
        return False  # Max retries reached for this path

    def _handle_api_error(e, attempt):
        logging.error(f"[Attempt {attempt}/{max_retries}] APIError: {e}")
        status_code = None
        if get_status_code_fn:
            try:
                status_code = get_status_code_fn(e)
            except Exception as ex_status_fn:
                logging.warning(
                    f"Could not get status code from exception {type(e)} using get_status_code_fn: {ex_status_fn}"
                )

        if status_code == 429 or (status_code and status_code >= 500):
            log_msg = "Rate limit (429)" if status_code == 429 else f"Server error ({status_code})"
            logging.warning(f"{log_msg} indicated by status code. Retrying after delay...")
            if attempt < max_retries:
                delay = _calculate_delay(
                    attempt, initial_retry_delay_seconds, max_retry_delay_seconds
                )
                logging.debug(
                    f"Sleeping for {delay:.2f} seconds due to API error status {status_code}."
                )
                time.sleep(delay)
                return True  # Indicate retry
            return False  # Max retries reached for this path
        else:
            logging.error(
                f"Non-retriable or unrecognized API error occurred (status: {status_code}). Raising."
            )
            raise e  # Re-raise non-retriable error

    # Main retry loop
    for attempt in range(1, max_retries + 1):
        try:
            response = client_function(**api_params)

            if is_response_valid_fn(response):
                logging.info(f"[Attempt {attempt}/{max_retries}] API call succeeded.")
                return response
            else:
                if _handle_invalid_response_content(attempt):
                    continue
                else:  # Max retries reached after invalid content
                    break

        except rate_limit_exceptions as e:
            if _handle_rate_limit_error(e, attempt):
                continue
            else:  # Max retries reached after rate limit
                break

        except api_error_exceptions as e:
            # _handle_api_error will raise if non-retriable, or return True to continue
            if _handle_api_error(e, attempt):
                continue
            else:  # Max retries reached for retriable API error
                break

        except Exception as e:  # Catch-all for truly unexpected errors
            logging.exception(
                f"[Attempt {attempt}/{max_retries}] Unexpected exception: {e}. Raising."
            )
            raise e  # Re-raise unexpected errors immediately

    logging.error(f"Exceeded maximum {max_retries} retry attempts. API call failed.")
    raise RuntimeError(f"API call failed after {max_retries} retries.")


def call_openai_api_with_retries(client_function, api_params, max_retries=10):
    """
    Makes an OpenAI API call with retries for transient failures,
    rate limiting, and invalid or error-containing responses.
    (This is now a wrapper around generic_call_api_with_retries for OpenAI)

    Args:
        client_function: The OpenAI API client function to call.
        api_params: Parameters to pass to the client function.
        max_retries: Maximum number of retry attempts.

    Returns:
        The OpenAI API response if successful.
    """

    def is_openai_response_valid(response):
        # Check for explicit error field in response object first
        if getattr(response, "error", None):
            logging.warning(f"OpenAI API response contains an error attribute: {response.error}")
            return False  # Treat as invalid for retry purposes
        if hasattr(response, "choices") and response.choices:  # Chat Completion API
            return True
        if hasattr(response, "output") and response.output:  # Response API
            return True
        logging.warning("OpenAI API response is missing 'choices' or 'output' is empty.")
        return False

    def get_openai_status_code(exception):
        return getattr(exception, "http_status", None)

    return generic_call_api_with_retries(
        client_function=client_function,
        api_params=api_params,
        is_response_valid_fn=is_openai_response_valid,
        rate_limit_exceptions=(openai.RateLimitError,),
        api_error_exceptions=(openai.APIError,),  # openai.RateLimitError is caught first
        get_status_code_fn=get_openai_status_code,
        max_retries=max_retries,
        # You can also pass initial_retry_delay_seconds and max_retry_delay_seconds
        # if you want to customize them from their defaults in the generic function.
    )


def call_anthropic_api_with_retries(client_function, api_params, max_retries=10):
    """
    Makes an Anthropic API call with retries for transient failures,
    rate limiting, and invalid responses.
    (This is a wrapper around generic_call_api_with_retries for Anthropic)

    Args:
        client_function: The Anthropic API client function to call.
        api_params: Parameters to pass to the client function.
        max_retries: Maximum number of retry attempts.

    Returns:
        The Anthropic API response if successful.
    """

    def is_anthropic_response_valid(response):
        """Checks if the Anthropic response is valid."""
        # A successful Anthropic message response typically has:
        # - a 'type' attribute equal to 'message' (for message creation)
        # - a 'content' attribute which is a list of blocks
        # - no 'error' attribute at the top level of the response object itself
        #   (errors are usually raised as exceptions by the client)

        if not response:
            logging.warning("Anthropic API response is None or empty.")
            return False

        # Check for explicit error type if the API might return it in a 200 OK
        # For anthropic.types.Message, an error would typically be an exception.
        # However, if the client_function could return a dict with an 'error' key:
        if isinstance(response, dict) and response.get("type") == "error":
            logging.warning(f"Anthropic API response indicates an error: {response.get('error')}")
            return False

        # For anthropic.types.Message objects from client.messages.create
        if hasattr(response, "type") and response.type == "message":
            if hasattr(response, "content") and isinstance(response.content, list):
                # Optionally, check if content is not empty, though an empty content list
                # might be valid for some assistant stop reasons.
                return True
            else:
                logging.warning(
                    "Anthropic API response is of type 'message' but missing valid 'content'."
                )
                return False

        logging.warning(
            f"Anthropic API response does not appear to be a valid message object. Type: {getattr(response, 'type', 'N/A')}"
        )
        return False

    def get_anthropic_status_code(exception):
        """Extracts HTTP status code from an Anthropic exception."""
        # anthropic.APIStatusError has a 'status_code' attribute
        return getattr(exception, "status_code", None)

    # Define Anthropic specific exceptions.
    # anthropic.RateLimitError for specific rate limit errors.
    # anthropic.APIError is a base class for many errors.
    # anthropic.APIStatusError provides status_code.
    # anthropic.APIConnectionError for network issues.
    # Order can matter if there's inheritance; specific ones first.

    # Ensure these are the correct exception types from your installed anthropic library version.
    anthropic_rate_limit_exception = anthropic.RateLimitError
    # Broader API errors, APIStatusError is more specific for HTTP status related issues.
    # APIConnectionError for network problems. APIError as a general catch-all.
    anthropic_api_error_exceptions = (
        anthropic.APIStatusError,  # Catches errors with a status_code
        anthropic.APIConnectionError,  # Catches network-related issues
        anthropic.APIError,  # General base class for other Anthropic API errors
    )

    return generic_call_api_with_retries(
        client_function=client_function,
        api_params=api_params,
        is_response_valid_fn=is_anthropic_response_valid,
        rate_limit_exceptions=(anthropic_rate_limit_exception,),
        api_error_exceptions=anthropic_api_error_exceptions,
        get_status_code_fn=get_anthropic_status_code,
        max_retries=max_retries,
        # You can also pass initial_retry_delay_seconds and max_retry_delay_seconds
        # if you want to customize them from their defaults in the generic function.
    )


def supports_tool_calling_for_openrouter(
    model_name: str,
) -> bool:
    """
    Check if the openrouter model supports tool calling.

    Args:
        model_name (str): The name of the model.

    Returns:
        bool: True if the model supports tool calling, False otherwise.
    """
    import os

    import openai

    client = openai.Client(
        api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1"
    )
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Call the test tool"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "dummy_tool",
                        "description": "Just a test tool",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                }
            ],
            tool_choice="required",
        )
        response = response.to_dict()
        return "tool_calls" in response["choices"][0]["message"]
    except Exception as e:
        print(f"Skipping tool callign support check in openrouter for {model_name}: {e}")
        return True


def retry_multiple(
    chat: "ChatModel",
    messages: "Discussion",
    n_retry: int,
    parser: callable,
    log: bool = True,
    num_samples: int = 1,
):
    """Retry querying the chat models with the response from the parser until it
    returns a valid value.

    If the answer is not valid, it will retry and append to the chat the  retry
    message.  It will stop after `n_retry`.

    Note, each retry has to resend the whole prompt to the API. This can be slow
    and expensive.

    Args:
        chat (ChatModel): a ChatModel object taking a list of messages and
            returning a list of answers, all in OpenAI format.
        messages (list): the list of messages so far. This list will be modified with
            the new messages and the retry messages.
        n_retry (int): the maximum number of sequential retries.
        parser (callable): a function taking a message and retruning a parsed value,
            or raising a ParseError
        log (bool): whether to log the retry messages.
        num_samples (int): the number of samples to generate from the model.

    Returns:
        list[dict]: the parsed value, with a string at key "action".

    Raises:
        ParseError: if the parser could not parse the response after n_retry retries.
    """
    tries = 0
    while tries < n_retry:
        answer_list = chat(messages, n_samples=num_samples)
        # TODO: could we change this to not use inplace modifications ?
        if not isinstance(answer_list, list):
            answer_list = [answer_list]

        # TODO taking the 1st hides the other generated answers in AgentXRay
        messages.append(answer_list[0])
        parsed_answers = []
        errors = []
        for answer in answer_list:
            try:
                parsed_answers.append(parser(answer["content"]))
            except ParseError as parsing_error:
                errors.append(str(parsing_error))
        # if we have a valid answer, return it
        if parsed_answers:
            return parsed_answers, tries
        else:
            tries += 1
            if log:
                msg = f"Query failed. Retrying {tries}/{n_retry}.\n[LLM]:\n{answer['content']}\n[User]:\n{str(errors)}"
                logging.info(msg)
            messages.append(dict(role="user", content=str(errors)))

    raise ParseError(f"Could not parse a valid value after {n_retry} retries.")


def truncate_tokens(text, max_tokens=8000, start=0, model_name="gpt-4"):
    """Use tiktoken to truncate a text to a maximum number of tokens."""
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)
    if len(tokens) - start > max_tokens:
        return enc.decode(tokens[start : (start + max_tokens)])
    else:
        return text


@cache
def get_tokenizer_old(model_name="openai/gpt-4"):
    if model_name.startswith("test"):
        return tiktoken.encoding_for_model("gpt-4")
    if model_name.startswith("openai"):
        return tiktoken.encoding_for_model(model_name.split("/")[-1])
    if model_name.startswith("azure"):
        return tiktoken.encoding_for_model(model_name.split("/")[1])
    if model_name.startswith("reka"):
        logging.warning(
            "Reka models don't have a tokenizer implemented yet. Using the default one."
        )
        return tiktoken.encoding_for_model("gpt-4")
    else:
        # Lazy import of transformers only when needed
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as e:
            raise ImportError(
                "The 'transformers' package is required to use non-OpenAI/Azure tokenizers."
            ) from e
        return AutoTokenizer.from_pretrained(model_name)


@cache
def get_tokenizer(model_name="gpt-4"):
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        logging.info(f"Could not find a tokenizer for model {model_name}. Trying HuggingFace.")
    try:
        from transformers import AutoTokenizer  # type: ignore

        return AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logging.info(f"Could not find a tokenizer for model {model_name}: {e} Defaulting to gpt-4.")
    return tiktoken.encoding_for_model("gpt-4")


def count_tokens(text, model="openai/gpt-4"):
    enc = get_tokenizer(model)
    return len(enc.encode(text))


def json_parser(message):
    """Parse a json message for the retry function."""

    try:
        value = json.loads(message)
        valid = True
        retry_message = ""
    except json.JSONDecodeError as e:
        warn(e)
        value = {}
        valid = False
        retry_message = "Your response is not a valid json. Please try again and be careful to the format. Don't add any apology or comment, just the answer."
    return value, valid, retry_message


def yaml_parser(message):
    """Parse a yaml message for the retry function."""

    # saves gpt-3.5 from some yaml parsing errors
    message = re.sub(r":\s*\n(?=\S|\n)", ": ", message)

    try:
        value = yaml.safe_load(message)
        valid = True
        retry_message = ""
    except yaml.YAMLError as e:
        warn(str(e))
        value = {}
        valid = False
        retry_message = "Your response is not a valid yaml. Please try again and be careful to the format. Don't add any apology or comment, just the answer."
    return value, valid, retry_message


def _compress_chunks(text, identifier, skip_list, split_regex="\n\n+"):
    """Compress a string by replacing redundant chunks by identifiers. Chunks are defined by the split_regex."""
    text_list = re.split(split_regex, text)
    text_list = [chunk.strip() for chunk in text_list]
    counter = collections.Counter(text_list)
    def_dict = {}
    id = 0

    # Store items that occur more than once in a dictionary
    for item, count in counter.items():
        if count > 1 and item not in skip_list and len(item) > 10:
            def_dict[f"{identifier}-{id}"] = item
            id += 1

    # Replace redundant items with their identifiers in the text
    compressed_text = "\n".join(text_list)
    for key, value in def_dict.items():
        compressed_text = compressed_text.replace(value, key)

    return def_dict, compressed_text


def compress_string(text):
    """Compress a string by replacing redundant paragraphs and lines with identifiers."""

    # Perform paragraph-level compression
    def_dict, compressed_text = _compress_chunks(
        text, identifier="§", skip_list=[], split_regex="\n\n+"
    )

    # Perform line-level compression, skipping any paragraph identifiers
    line_dict, compressed_text = _compress_chunks(
        compressed_text, "¶", list(def_dict.keys()), split_regex="\n+"
    )
    def_dict.update(line_dict)

    # Create a definitions section
    def_lines = ["<definitions>"]
    for key, value in def_dict.items():
        def_lines.append(f"{key}:\n{value}")
    def_lines.append("</definitions>")
    definitions = "\n".join(def_lines)

    return definitions + "\n" + compressed_text


def extract_html_tags(text, keys):
    """Extract the content within HTML tags for a list of keys.

    All text and keys will be converted to lowercase before matching.

    Args:
        text (str): The input string containing the HTML tags.
        keys (list[str]): The HTML tags to extract the content from.

    Returns:
        dict: A dictionary mapping each key to a list of subset in `text` that match the key.
    """
    content_dict = {}
    # text = text.lower()
    # keys = set([k.lower() for k in keys])
    for key in keys:
        pattern = f"<{key}>(.*?)</{key}>"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            content_dict[key] = [match.strip() for match in matches]
    return content_dict


class ParseError(Exception):
    pass


def extract_code_blocks(text) -> list[tuple[str, str]]:
    pattern = re.compile(r"```(\w*\n)?(.*?)```", re.DOTALL)

    matches = pattern.findall(text)
    return [(match[0].strip(), match[1].strip()) for match in matches]


def parse_html_tags_raise(text, keys=(), optional_keys=(), merge_multiple=False):
    """A version of parse_html_tags that raises an exception if the parsing is not successful."""
    content_dict, valid, retry_message = parse_html_tags(
        text, keys, optional_keys, merge_multiple=merge_multiple
    )
    if not valid:
        raise ParseError(retry_message)
    return content_dict


def parse_html_tags(text, keys=(), optional_keys=(), merge_multiple=False):
    """Satisfy the parse api, extracts 1 match per key and validates that all keys are present

    Args:
        text (str): The input string containing the HTML tags.
        keys (list[str]): The HTML tags to extract the content from.
        optional_keys (list[str]): The HTML tags to extract the content from, but are optional.
        merge_multiple (bool): Whether to merge multiple instances of the same key.

    Returns:
        dict: A dictionary mapping each key to a subset of `text` that match the key.
        bool: Whether the parsing was successful.
        str: A message to be displayed to the agent if the parsing was not successful.

    """
    all_keys = tuple(keys) + tuple(optional_keys)
    content_dict = extract_html_tags(text, all_keys)
    retry_messages = []

    for key in all_keys:
        if key not in content_dict:
            if key not in optional_keys:
                retry_messages.append(f"Missing the key <{key}> in the answer.")
        else:
            val = content_dict[key]
            content_dict[key] = val[0]
            if len(val) > 1:
                if not merge_multiple:
                    retry_messages.append(
                        f"Found multiple instances of the key {key}. You should have only one of them."
                    )
                else:
                    # merge the multiple instances
                    content_dict[key] = "\n".join(val)

    valid = len(retry_messages) == 0
    retry_message = "\n".join(retry_messages)
    return content_dict, valid, retry_message


def download_and_save_model(model_name: str, save_dir: str = "."):
    # Lazy import of transformers only when explicitly downloading a model
    try:
        from transformers import AutoModel  # type: ignore
    except Exception as e:
        raise ImportError(
            "The 'transformers' package is required to download and save models."
        ) from e
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(save_dir)
    print(f"Model downloaded and saved to {save_dir}")


def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """Convert a numpy array to a base64 encoded image url."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")

    image_base64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{image_base64}"


def image_to_png_base64_url(image: np.ndarray | Image.Image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, "PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{image_base64}"


def img_to_base_64(image: Image.Image | np.ndarray) -> str:
    """Converts a PIL Image or NumPy array to a base64-encoded string."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return b64_str


class BaseMessage(dict):
    def __init__(self, role: str, content: Union[str, list[dict]], **kwargs):
        allowed_attrs = {"log_probs"}
        invalid_attrs = set(kwargs.keys()) - allowed_attrs
        if invalid_attrs:
            raise ValueError(f"Invalid attributes: {invalid_attrs}")
        self["role"] = role
        self["content"] = deepcopy(content)
        self.update(kwargs)

    def __str__(self, warn_if_image=False) -> str:
        if isinstance(self["content"], str):
            return self["content"]
        if not all(elem["type"] == "text" for elem in self["content"]):
            msg = "The content of the message has images, which are not displayed in the string representation."
            if warn_if_image:
                logging.warning(msg)
            else:
                logging.info(msg)

        return "\n".join(
            [
                elem["text"]
                for elem in self["content"]
                if elem["type"] == "text" or elem["type"] == "input_text"
            ]
        )

    def add_content(self, type: str, content: Any):
        if isinstance(self["content"], str):
            text = self["content"]
            self["content"] = []
            self["content"].append({"type": "text", "text": text})
        self["content"].append({"type": type, type: content})

    def add_text(self, text: str):
        self.add_content("text", text)

    def add_image(self, image: np.ndarray | Image.Image | str, detail: str = None):
        if not isinstance(image, str):
            image_url = image_to_jpg_base64_url(image)
        else:
            image_url = image
        if detail:
            self.add_content("image_url", {"url": image_url, "detail": detail})
        else:
            self.add_content("image_url", {"url": image_url})

    def to_markdown(self):
        if isinstance(self["content"], str):
            return f"\n```\n{self['content']}\n```\n"
        res = []
        for elem in self["content"]:
            # add texts between ticks and images
            if elem["type"] == "text":
                res.append(f"\n```\n{elem['text']}\n```\n")
            elif elem["type"] == "image_url":
                img_str = (
                    elem["image_url"]
                    if isinstance(elem["image_url"], str)
                    else elem["image_url"]["url"]
                )
                res.append(f"![image]({img_str})")
        return "\n".join(res)

    def merge(self):
        """Merges content elements of type 'text' if they are adjacent."""
        if isinstance(self["content"], str):
            return
        new_content = []
        for elem in self["content"]:
            if elem["type"] == "text":
                if new_content and new_content[-1]["type"] == "text":
                    new_content[-1]["text"] += "\n" + elem["text"]
                else:
                    new_content.append(elem)
            else:
                new_content.append(elem)
        self["content"] = new_content
        if len(self["content"]) == 1:
            self["content"] = self["content"][0]["text"]


class SystemMessage(BaseMessage):
    def __init__(self, content: Union[str, list[dict]]):
        super().__init__("system", content)


class HumanMessage(BaseMessage):
    def __init__(self, content: Union[str, list[dict]]):
        super().__init__("user", content)


class AIMessage(BaseMessage):
    def __init__(self, content: Union[str, list[dict]], log_probs=None):
        super().__init__("assistant", content, log_probs=log_probs)


class Discussion:
    def __init__(self, messages: Union[list[BaseMessage], BaseMessage] = None):
        if isinstance(messages, BaseMessage):
            messages = [messages]
        elif messages is None:
            messages = []
        self.messages = messages

    @property
    def last_message(self):
        return self.messages[-1]

    def merge(self):
        for m in self.messages:
            m.merge()

    def __str__(self) -> str:
        return "\n".join(str(m) for m in self.messages)

    def to_string(self):
        self.merge()
        return str(self)

    def to_openai(self):
        self.merge()
        return self.messages

    def add_message(
        self,
        message: BaseMessage | dict = None,
        role: str = None,
        content: Union[str, list[dict]] = None,
    ):
        if message is None:
            message = BaseMessage(role, content)
        else:
            if isinstance(message, dict):
                message = BaseMessage(**message)
        self.messages.append(message)

    def append(self, message: BaseMessage | dict):
        self.add_message(message)

    def add_content(self, type: str, content: Any):
        """Add content to the last message."""
        self.last_message.add_content(type, content)

    def add_text(self, text: str):
        """Add text to the last message."""
        self.last_message.add_text(text)

    def add_image(self, image: np.ndarray | Image.Image | str, detail: str = None):
        """Add an image to the last message."""
        self.last_message.add_image(image, detail)

    def __iter__(self):
        return iter(self.messages)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, key):
        return self.messages[key]

    def to_markdown(self):
        self.merge()
        return "\n".join([f"Message {i}\n{m.to_markdown()}\n" for i, m in enumerate(self.messages)])


if __name__ == "__main__":
    # model_to_download = "THUDM/agentlm-70b"
    model_to_download = "databricks/dbrx-instruct"
    save_dir = "/mnt/ui_copilot/data_rw/base_models/"
    # set the following env variable to enable the transfer of the model
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    download_and_save_model(model_to_download, save_dir=save_dir)
