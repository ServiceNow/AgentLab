import logging
import sys
from functools import lru_cache


@lru_cache(maxsize=None)
def setup_logging(level=logging.INFO):
    """Configure logging once and cache the result.

    Using lru_cache ensures this only runs once per process,
    even if imported and called multiple times.
    """
    # Remove any existing handlers to avoid duplicates
    root = logging.getLogger()
    for handler in root.handlers:
        root.removeHandler(handler)

    # Configure format and handler
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Set up root logger
    root.addHandler(console_handler)
    root.setLevel(level)

    return root


# Call it once when module is imported
logger = setup_logging()
