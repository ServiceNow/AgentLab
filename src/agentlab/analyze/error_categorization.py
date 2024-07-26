import re


def is_critical_server_error(err_msg: str, stack_trace: str, return_error_type=False):
    if stack_trace is None:
        return False
    server_error_conditions = [
        "502 Server Error: Bad Gateway",  # url is not valid or server is not ready yet
        "500 Server Error: Internal Server Error",  # When the server is up, but somehow unresponsive
        "424 Client Error: Failed Dependency",  # Request failed during generation: Server error: CUDA error: invalid argument
        "403 Client Error: Forbidden",
        "401 Client Error: Unauthorized",  # the server was auth-token protected
        "400 Client Error: Bad Request",  # there's no server. (the server probably failed) sometimes only visible in stack trace?
    ]
    if not return_error_type:
        return any(condition in stack_trace for condition in server_error_conditions)
    else:
        for condition in server_error_conditions:
            if condition in stack_trace:
                return condition
        return None


def is_minor_server_error(err_msg: str, stack_trace: str, return_error_type=False):
    if stack_trace is None:
        return False
    server_error_conditions = [
        "504 Server Error: Gateway Time-out",
        "Server error: Out of available cache blocks",  # TODO(look into)
        "openai.APIConnectionError: Connection error",
    ]
    if not return_error_type:
        return any(condition in stack_trace for condition in server_error_conditions)
    else:
        for condition in server_error_conditions:
            if condition in stack_trace:
                return condition
        return None


def is_retry_error(err_msg: str, stack_trace: str):
    """Use regex on the stack trace to detect retry errors.

    The pattern is "ValueError: Could not parse a valid value after d+
    retries."

    Args:
        err_msg (str): The error message
        stack_trace (str): The stack trace

    Returns:
        bool: True if the error is a retry error, False otherwise
    """
    if stack_trace is None:
        return False
    pattern = r"ValueError: Could not parse a valid value after \d+ retries."
    return re.search(pattern, stack_trace) is not None


def is_input_length_error(err_msg: str, stack_trace: str):
    """Use regex on the stack trace to detect input length errors.

    The patterns are "422 Client Error: Unprocessable Entity"
    and also "Input validation error: `inputs` tokens + `max_new_tokens` must be <="

    Args:
        err_msg (str): The error message
        stack_trace (str): The stack trace

    Returns:
        bool: True if the error is an input length error, False otherwise
    """
    if stack_trace is None:
        return False
    patterns = [
        r"422 Client Error: Unprocessable Entity",
        r"Input validation error: `inputs` tokens \+ `max_new_tokens` must be <=",
    ]
    return any(
        re.search(pattern, err_msg) or re.search(pattern, stack_trace) for pattern in patterns
    )


ERR_CLASS_MAP = {
    "critical_server_error": is_critical_server_error,
    "minor_server_error": is_minor_server_error,
    "retry_error": is_retry_error,
    "input_length_error": is_input_length_error,
}
