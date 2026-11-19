"""Code identity for run manifests and label provenance."""

from __future__ import annotations

import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*arguments: str, cwd: Path = REPO_ROOT) -> str | None:
    try:
        result = subprocess.run(["git", *arguments], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


@lru_cache(maxsize=8)
def code_identity(repo: Path = REPO_ROOT) -> dict:
    """Commit of the repository, plus a digest of uncommitted changes when the tree is dirty."""
    commit = _git("rev-parse", "HEAD", cwd=repo)
    status = _git("status", "--porcelain", cwd=repo)
    identity = {"repository": repo.name, "git_commit": commit.strip() if commit else None,
                "git_dirty": bool(status and status.strip())}
    if identity["git_dirty"]:
        diff = (_git("diff", "HEAD", cwd=repo) or "") + (status or "")
        identity["diff_sha256"] = hashlib.sha256(diff.encode("utf-8", "replace")).hexdigest()
    return identity
