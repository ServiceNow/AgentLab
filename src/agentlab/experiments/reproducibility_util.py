from copy import deepcopy

from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from pathlib import Path
from git import Repo, InvalidGitRepositoryError
from importlib import metadata
from git.config import GitConfigParser
import os


def _get_repo(module):
    return Repo(Path(module.__file__).resolve().parent, search_parent_directories=True)


def _get_benchmark_version(benchmark_name):
    if benchmark_name.startswith("miniwob"):
        return metadata.distribution("browsergym.miniwob").version
    elif benchmark_name.startswith("workarena"):
        return metadata.distribution("browsergym.workarena").version
    elif benchmark_name.startswith("webarena"):
        return metadata.distribution("browsergym.webarena").version
    elif benchmark_name.startswith("visualwebarena"):
        return metadata.distribution("browsergym.visualwebarena").version
    else:
        raise ValueError(f"Unknown benchmark {benchmark_name}")


def get_git_username(repo: Repo) -> str:
    """
    Retrieves the first available Git username from various sources.

    This function checks multiple locations for the Git username in the following order:
    1. Repository-specific configuration
    2. GitHub API (if the remote is a GitHub repository)
    3. Global Git configuration
    4. System Git configuration
    5. Environment variables (GIT_AUTHOR_NAME and GIT_COMMITTER_NAME)

    Args:
        repo (git.Repo): A GitPython Repo object representing the Git repository.

    Returns:
        str: The first non-None username found, or None if no username is found.
    """
    # Repository-specific configuration
    username = repo.config_reader().get_value("user", "name", None)
    if username:
        return username

    # GitHub username
    remote_url = repo.remotes.origin.url
    if "github.com" in remote_url:
        import re
        import urllib.request
        import json

        match = re.search(r"github\.com[:/](.+)/(.+)\.git", remote_url)
        if match:
            owner, repo_name = match.groups()
            api_url = f"https://api.github.com/repos/{owner}/{repo_name}"
            with urllib.request.urlopen(api_url) as response:
                data = json.loads(response.read().decode())
                username = data["owner"]["login"]
                if username:
                    return username

    # Global configuration
    username = GitConfigParser(repo.git.config("--global", "--list"), read_only=True).get_value(
        "user", "name", None
    )
    if username:
        return username

    # System configuration
    username = GitConfigParser(repo.git.config("--system", "--list"), read_only=True).get_value(
        "user", "name", None
    )
    if username:
        return username

    # Environment variables
    return os.environ.get("GIT_AUTHOR_NAME") or os.environ.get("GIT_COMMITTER_NAME")


def get_git_info(module):
    """
    Retrieve comprehensive git information for the given module.

    This function attempts to find the git repository containing the specified
    module and returns the current commit hash and a comprehensive list of all
    files that contribute to the repository's state.

    Args:
        module: The Python module object to check for git information.

    Returns:
        tuple: A tuple containing two elements:
            - str or None: The current git commit hash, or None if not a git repo.
            - list of tuple: A list of (status, Path) tuples for all modified files.
              Empty list if not a git repo. Status can be 'M' (modified), 'A' (added),
              'D' (deleted), 'R' (renamed), 'C' (copied), 'U' (updated but unmerged),
              or '??' (untracked).
    """

    try:
        repo = _get_repo(module)

        git_hash = repo.head.object.hexsha

        modified_files = []

        # Staged changes
        staged_changes = repo.index.diff(repo.head.commit)
        for change in staged_changes:
            modified_files.append((change.change_type, Path(change.a_path)))

        # Unstaged changes
        unstaged_changes = repo.index.diff(None)
        for change in unstaged_changes:
            modified_files.append((change.change_type, Path(change.a_path)))

        # Untracked files
        untracked_files = repo.untracked_files
        for file in untracked_files:
            modified_files.append(("??", Path(file)))

        return git_hash, modified_files
    except InvalidGitRepositoryError:
        return None, []


def get_reproducibility_info(benchmark_name, ignore_changes=False):
    import agentlab
    from browsergym import core

    info = {
        "git_user": get_git_username(_get_repo(agentlab)),
        "benchmark": benchmark_name,
        "benchmark_version": _get_benchmark_version(benchmark_name),
    }

    def add_info(module_name, module):
        git_hash, modified_files = get_git_info(module)

        modified_files_str = "\n".join([f"{status} {file}" for status, file in modified_files])

        if len(modified_files) > 0 and not ignore_changes:
            raise ValueError(
                f"Module {module_name} has uncommitted changes."
                "Please commit or stash these changes before running the experiment or set ignore_changes=True."
                f"Modified files:  \n{modified_files_str}\n"
            )

        info[f"{module_name}_version"] = module.__version__
        info[f"{module_name}_git_hash"] = git_hash
        info[f"{module_name}__local_modifications"] = modified_files_str

    add_info("agentlab", agentlab)
    add_info("browsergym", core)
    return info


def set_temp(agent_args: GenericAgentArgs, temperature=0):
    """Set temperature to 0. Assumes a GenericAgent structure."""
    agent_args = deepcopy(agent_args)
    agent_args.chat_model_args.temperature = temperature
    return agent_args
