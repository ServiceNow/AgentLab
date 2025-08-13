"""Tiny benchmark setup helpers.

Currently supports MiniWob++: clones the repo at a pinned commit and writes
MINIWOB_URL to .env. Designed to be minimal and easy to maintain.
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Optional

logger = logging.getLogger(__name__)


def _ensure_repo(repo_url: str, clone_dir: pathlib.Path, commit: Optional[str] = None) -> None:
    """Clone repo if missing and optionally checkout a commit (minimal, shell-only).

    Args:
        repo_url: URL of the git repository to clone.
        clone_dir: Directory path where the repository should be cloned.
        commit: Optional commit hash to checkout after cloning.
    """
    clone_dir = clone_dir.resolve()
    if not clone_dir.exists():
        clone_dir.parent.mkdir(parents=True, exist_ok=True)
        os.system(f"git clone '{repo_url}' '{clone_dir}' >/dev/null 2>&1 || true")
    # If it's a git repo and a commit is provided, best-effort checkout
    if commit and (clone_dir / ".git").exists():
        os.system(f"git -C '{clone_dir}' fetch --all --tags >/dev/null 2>&1 || true")
        os.system(f"git -C '{clone_dir}' checkout {commit} >/dev/null 2>&1 || true")


def _write_env_kv(env_path: pathlib.Path, key: str, value: str) -> None:
    """Idempotently write/update KEY=VALUE in .env file."""
    env_path = env_path.resolve()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text().splitlines()
    key_prefix = f"{key}="
    updated = False
    new_line = f"{key}={value}"
    out_lines: list[str] = []
    for line in lines:
        if line.strip().startswith(key_prefix):
            if not updated:
                out_lines.append(new_line)
                updated = True
            # Skip any other occurrences
        else:
            out_lines.append(line)
    if not updated:
        out_lines.append(new_line)
    env_path.write_text("\n".join(out_lines) + "\n")


def setup_miniwob(project_root: pathlib.Path) -> str:
    """Set up MiniWob++ locally and configure MINIWOB_URL in .env.

    Steps:
    - Clone https://github.com/Farama-Foundation/miniwob-plusplus.git (if missing)
    - Checkout pinned commit for reproducibility
    - Compute file:// URL to the local miniwob HTML assets
    - Write MINIWOB_URL to <project_root>/.env

    Args:
        project_root: Project root directory path.

    Returns:
        The configured MINIWOB_URL string.
    """
    # Clone the upstream repo at a pinned commit and use local HTML assets
    repo_url = "https://github.com/Farama-Foundation/miniwob-plusplus.git"
    commit = "7fd85d71a4b60325c6585396ec4f48377d049838"
    clone_dir = project_root / "miniwob-plusplus"
    _ensure_repo(repo_url=repo_url, clone_dir=clone_dir, commit=commit)
    miniwob_dir = (clone_dir / "miniwob" / "html" / "miniwob").resolve()
    # We still set URL even if folder doesn't exist yet; but warn through return
    url = miniwob_dir.as_uri().rstrip("/") + "/"

    env_path = project_root / ".env"
    _write_env_kv(env_path, "MINIWOB_URL", url)
    os.environ["MINIWOB_URL"] = url  # make available in current process immediately
    logger.info("MINIWOB_URL set to %s (recorded in %s)", url, env_path)
    return url


def ensure_benchmark(benchmark: str, project_root: pathlib.Path) -> Optional[str]:
    """Run setup lazily if required for the given benchmark.

    Args:
        benchmark: Name of the benchmark to ensure setup for.
        project_root: Project root directory path.

    Returns:
        The URL when setup is performed, otherwise None.
    """
    key = benchmark.strip().lower()
    if key in {"miniwob", "miniwob++", "miniwob-plusplus"}:
        if not os.getenv("MINIWOB_URL"):
            return setup_miniwob(project_root)
        return None
    # No-op for unsupported/other benchmarks
    return None
