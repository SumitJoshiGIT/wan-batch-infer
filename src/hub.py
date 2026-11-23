"""Hub abstraction for Hugging Face and local testing stand-in.
Handles commit operations, listing files, and downloads with retry on transient network errors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import time
from typing import List, Optional, Union


@dataclass
class CommitOperation:
    path_in_repo: str
    content_bytes: Optional[bytes] = None
    path_or_fileobj: Optional[Union[str, Path]] = None
    delete: bool = False


class Hub(ABC):
    @abstractmethod
    def create_commit(
        self,
        repo_id: str,
        operations: List[CommitOperation],
        commit_message: str,
        repo_type: str = "dataset",
    ) -> str:
        """Create a single commit containing multiple operations (upload/delete)."""
        pass

    @abstractmethod
    def list_files(
        self,
        repo_id: Optional[str] = None,
        prefix: Optional[str] = None,
        repo_type: str = "dataset",
    ) -> List[str]:
        """List files in repository, optionally matching a prefix."""
        pass

    @abstractmethod
    def download_file(
        self,
        repo_id: str,
        path_in_repo: str,
        local_path: Union[str, Path],
        repo_type: str = "dataset",
    ) -> Path:
        """Download a file from the repository to local_path."""
        pass


class LocalHub(Hub):
    """File system directory stand-in for Hub testing."""

    def __init__(self, root_dir: Union[str, Path]):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create_commit(
        self,
        repo_id: str,
        operations: List[CommitOperation],
        commit_message: str,
        repo_type: str = "dataset",
    ) -> str:
        repo_dir = self.root_dir / repo_id
        repo_dir.mkdir(parents=True, exist_ok=True)

        for op in operations:
            target = repo_dir / op.path_in_repo
            if op.delete:
                if target.exists():
                    target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                if op.content_bytes is not None:
                    target.write_bytes(op.content_bytes)
                elif op.path_or_fileobj is not None:
                    src = Path(op.path_or_fileobj)
                    shutil.copy2(src, target)
                else:
                    raise ValueError(f"CommitOperation for {op.path_in_repo} has no content")

        return "local_commit_sha"

    def list_files(
        self,
        repo_id: Optional[str] = None,
        prefix: Optional[str] = None,
        repo_type: str = "dataset",
    ) -> List[str]:
        if repo_id:
            search_dir = self.root_dir / repo_id
        else:
            search_dir = self.root_dir

        if not search_dir.exists():
            return []

        rel_paths = []
        for p in search_dir.rglob("*"):
            if p.is_file():
                rel = p.relative_to(search_dir).as_posix()
                if prefix:
                    clean_prefix = prefix.strip("/")
                    if rel == clean_prefix or rel.startswith(clean_prefix + "/"):
                        rel_paths.append(rel)
                else:
                    rel_paths.append(rel)
        return sorted(rel_paths)

    def download_file(
        self,
        repo_id: str,
        path_in_repo: str,
        local_path: Union[str, Path],
        repo_type: str = "dataset",
    ) -> Path:
        src = self.root_dir / repo_id / path_in_repo
        if not src.exists():
            raise FileNotFoundError(f"File {path_in_repo} not found in repo {repo_id}")
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return dest


class HFHub(Hub):
    """Hugging Face Hub implementation using huggingface_hub."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("HF_TOKEN")
        from huggingface_hub import HfApi
        self.api = HfApi(token=self.token)

    def create_commit(
        self,
        repo_id: str,
        operations: List[CommitOperation],
        commit_message: str,
        repo_type: str = "dataset",
    ) -> str:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        from huggingface_hub.utils import HfHubHTTPError

        hf_ops = []
        for op in operations:
            if op.delete:
                hf_ops.append(CommitOperationDelete(path_in_repo=op.path_in_repo))
            else:
                if op.content_bytes is not None:
                    hf_ops.append(
                        CommitOperationAdd(
                            path_in_repo=op.path_in_repo,
                            path_or_fileobj=op.content_bytes,
                        )
                    )
                elif op.path_or_fileobj is not None:
                    hf_ops.append(
                        CommitOperationAdd(
                            path_in_repo=op.path_in_repo,
                            path_or_fileobj=str(op.path_or_fileobj),
                        )
                    )

        # Retry loop for 429 and 5xx
        max_retries = 5
        base_delay = 2.0
        for attempt in range(max_retries):
            try:
                commit_info = self.api.create_commit(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    operations=hf_ops,
                    commit_message=commit_message,
                )
                return getattr(commit_info, "oid", str(commit_info))
            except HfHubHTTPError as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status and (status == 429 or status >= 500):
                    if attempt < max_retries - 1:
                        sleep_time = base_delay * (2 ** attempt)
                        time.sleep(sleep_time)
                        continue
                raise

        raise RuntimeError(f"create_commit failed after {max_retries} retries")

    def list_files(
        self,
        repo_id: Optional[str] = None,
        prefix: Optional[str] = None,
        repo_type: str = "dataset",
    ) -> List[str]:
        if not repo_id:
            raise ValueError("repo_id required for HFHub.list_files")
        all_files = self.api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
        if prefix:
            clean_prefix = prefix.strip("/")
            return [
                f for f in all_files
                if f == clean_prefix or f.startswith(clean_prefix + "/")
            ]
        return all_files

    def download_file(
        self,
        repo_id: str,
        path_in_repo: str,
        local_path: Union[str, Path],
        repo_type: str = "dataset",
    ) -> Path:
        from huggingface_hub import hf_hub_download
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type=repo_type,
            local_dir=str(dest.parent),
            token=self.token,
        )
        if Path(downloaded).resolve() != dest.resolve():
            shutil.copy2(downloaded, dest)
        return dest
