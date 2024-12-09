# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import subprocess


# Automatically retrieve the project version from Git
def get_version():
    try:
        return subprocess.check_output(["git", "describe", "--tags"], encoding="utf-8").strip()
    except Exception:
        return "0.0.0"


project = "AgentLab"
author = "ServiceNow"
release = get_version()  # Full version string including tags
version = release  # Short version (e.g., 1.0)


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    # "myst_parser",  # Add this to enable Markdown parsing
    "sphinx.ext.napoleon",
]

intersphinx_mapping = {
    "rtd": ("https://docs.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

source_suffix = {
    ".rst": "restructuredtext",
}


templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": -1,
    "collapse_navigation": False,
    "display_version": True,
    "version_selector": True,
}
html_static_path = ["_static"]

import sys

sys.path.insert(0, os.path.abspath("../../src"))
