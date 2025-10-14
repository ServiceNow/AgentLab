try:
    from ._version import __version__
except ImportError:
    # Fallback for development without installed package
    __version__ = "unknown"
