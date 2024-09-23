from copy import deepcopy
import csv
from datetime import datetime
import json
import logging
import platform

from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from pathlib import Path
from git import Repo, InvalidGitRepositoryError
from importlib import metadata
from git.config import GitConfigParser
import os
import agentlab


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


def _get_git_username(repo: Repo) -> str:
    """
    Retrieves the first available Git username from various sources.

    Note: overlycomplex designed by Claude and not fully tested.

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
    try:
        username = repo.config_reader().get_value("user", "name", None)
        if username:
            return username
    except Exception:
        pass

    try:
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
    except Exception:
        pass

    try:
        # Global configuration
        username = GitConfigParser(repo.git.config("--global", "--list"), read_only=True).get_value(
            "user", "name", None
        )
        if username:
            return username
    except Exception:
        pass

    try:
        # System configuration
        username = GitConfigParser(repo.git.config("--system", "--list"), read_only=True).get_value(
            "user", "name", None
        )
        if username:
            return username
    except Exception:
        pass

    # Environment variables
    return os.environ.get("GIT_AUTHOR_NAME") or os.environ.get("GIT_COMMITTER_NAME")


def _get_git_info(module, changes_white_list=()) -> tuple[str, list[tuple[str, Path]]]:
    """
    Retrieve comprehensive git information for the given module.

    This function attempts to find the git repository containing the specified
    module and returns the current commit hash and a comprehensive list of all
    files that contribute to the repository's state.

    Args:
        module: The Python module object to check for git information.
        changes_white_list: A list of file paths to ignore when checking for changes.

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

        # wildcard matching from white list
        modified_files_filtered = []
        for status, file in modified_files:
            if any(file.match(pattern) for pattern in changes_white_list):
                continue
            modified_files_filtered.append((status, file))

        return git_hash, modified_files_filtered
    except InvalidGitRepositoryError:
        return None, []


def get_reproducibility_info(
    agent_name,
    benchmark_name,
    comment=None,
    changes_white_list=(  # Files that are often modified during experiments but do not affect reproducibility
        "*/reproducibility_script.py",
        "*reproducibility_journal.csv",
        "*/launch_command.py",
    ),
    ignore_changes=False,
):
    """
    Retrieve a dict of information that could influence the reproducibility of an experiment.
    """
    import agentlab
    from browsergym import core

    info = {
        "git_user": _get_git_username(_get_repo(agentlab)),
        "agent_name": agent_name,
        "benchmark": benchmark_name,
        "comment": comment,
        "benchmark_version": _get_benchmark_version(benchmark_name),
        "date": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "os": f"{platform.system()} ({platform.version()})",
        "python_version": platform.python_version(),
        "playwright_version": metadata.distribution("playwright").version,
    }

    def add_git_info(module_name, module):
        git_hash, modified_files = _get_git_info(module, changes_white_list)

        modified_files_str = "\n".join([f"  {status}: {file}" for status, file in modified_files])

        if len(modified_files) > 0:
            msg = (
                f"Module {module_name} has uncommitted changes. "
                f"Modified files:  \n{modified_files_str}\n"
            )
            if ignore_changes:
                logging.warning(
                    msg + "Ignoring changes as requested and proceeding to experiments."
                )
            else:
                raise ValueError(
                    msg + "Please commit or stash your changes before running the experiment."
                )

        info[f"{module_name}_version"] = module.__version__
        info[f"{module_name}_git_hash"] = git_hash
        info[f"{module_name}__local_modifications"] = modified_files_str

    add_git_info("agentlab", agentlab)
    add_git_info("browsergym", core)
    return info


def _assert_compatible(info: dict, old_info: dict):
    """Make sure that the two info dicts are compatible."""
    # TODO may need to adapt if there are multiple agents, and the re-run on
    # error only has a subset of agents. Hence old_info.agent_name != info.agent_name
    for key in info.keys():
        if key in ("date", "avg_reward", "std_err", "n_completed", "n_err"):
            continue
        if info[key] != old_info[key]:
            raise ValueError(
                f"Reproducibility info already exist and is not compatible."
                f"Key {key} has changed from {old_info[key]} to {info[key]}."
            )


def write_reproducibility_info(
    study_dir, agent_name, benchmark_name, comment=None, ignore_changes=False
):
    info = get_reproducibility_info(
        agent_name, benchmark_name, comment, ignore_changes=ignore_changes
    )
    return save_reproducibility_info(study_dir, info)


def save_reproducibility_info(study_dir, info):
    """
    Save a JSON file containing reproducibility information to the specified directory.
    """

    info_path = Path(study_dir) / "reproducibility_info.json"

    if info_path.exists():
        with open(info_path, "r") as f:
            existing_info = json.load(f)
        _assert_compatible(info, existing_info)
        logging.info(
            "Reproducibility info already exists and is compatible. Overwriting the old one."
        )

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    info_str = json.dumps(info, indent=4)
    logging.info(f"Reproducibility info saved to {info_path}. Info: {info_str}")

    return info


def load_reproducibility_info(study_dir) -> dict[str]:
    """Retrieve the reproducibility info from the study directory."""
    info_path = Path(study_dir) / "reproducibility_info.json"
    with open(info_path, "r") as f:
        return json.load(f)


from agentlab.analyze import inspect_results


def add_reward(info, study_dir, ignore_incomplete=False):
    result_df = inspect_results.load_result_df(study_dir)
    report = inspect_results.summarize_study(result_df)

    if len(report) > 1:
        raise ValueError("Multi agent not implemented yet")

    assert isinstance(info["agent_name"], str)

    idx = report.index[0]
    n_err = report.loc[idx, "n_err"].item()
    n_completed, n_total = report.loc[idx, "n_completed"].split("/")
    if n_err > 0 and not ignore_incomplete:
        raise ValueError(
            f"Experiment has {n_err} errors. Please rerun the study and make sure all tasks are completed."
        )
    if n_completed != n_total and not ignore_incomplete:
        raise ValueError(
            f"Experiment has {n_completed} completed tasks out of {n_total}. "
            f"Please rerun the study and make sure all tasks are completed."
        )

    for key in ("avg_reward", "std_err", "n_err", "n_completed"):
        value = report.loc[idx, key]
        if hasattr(value, "item"):
            value = value.item()
        info[key] = value


def _get_csv_headers(file_path: str) -> list[str]:
    with open(file_path, "r", newline="") as file:
        reader = csv.reader(file)
        try:
            headers = next(reader)
        except StopIteration:
            headers = None
    return headers


def append_to_journal(info, journal_path=None):
    if journal_path is None:
        journal_path = Path(agentlab.__file__).parent.parent.parent / "reproducibility_journal.csv"

    rows = []
    headers = None
    if journal_path.exists():
        headers = _get_csv_headers(journal_path)

    if headers is None:
        headers = list(info.keys())
        rows.append(headers)

    if isinstance(info["agent_name"], (list, tuple)):
        # handle multiple agents
        assert len(info["agent_name"]) == len(info["reward"])
        assert len(info["agent_name"]) == len(info["std_err"])

        for i, agent_name in info["agent_name"]:
            sub_info = info.copy()
            sub_info["agent_name"] = agent_name
            sub_info["reward"] = info["reward"][i]
            sub_info["std_err"] = info["std_err"][i]
            rows.append([str(sub_info[key]) for key in headers])
    else:
        rows.append([str(info[key]) for key in headers])
    with open(journal_path, "a", newline="") as file:
        writer = csv.writer(file)
        for row in rows:
            writer.writerow(row)


def add_experiment_to_journal(study_dir, ignore_incomplete=False):
    info = load_reproducibility_info(study_dir)
    add_reward(info, study_dir, ignore_incomplete)
    save_reproducibility_info(study_dir, info)
    append_to_journal(info)


def set_temp(agent_args: GenericAgentArgs, temperature=0):
    """Set temperature to 0. Assumes a GenericAgent structure."""
    agent_args = deepcopy(agent_args)
    agent_args.chat_model_args.temperature = temperature
    return agent_args
