"""Tests for src/hub.py hub abstraction."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hub import Hub, LocalHub, HFHub, CommitOperation


def test_local_hub_commit_and_list_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        hub = LocalHub(root_dir=tmpdir)
        
        # Initial file list should be empty
        assert hub.list_files() == []
        
        # Perform a commit operation with file additions and deletions
        ops = [
            CommitOperation(path_in_repo="batches/b1/items.jsonl", content_bytes=b'{"id": "1"}\n'),
            CommitOperation(path_in_repo="clips/b1/1/clip.mp4", content_bytes=b'mp4data'),
            CommitOperation(path_in_repo="clips/b1/1/meta.json", content_bytes=b'{"vram": 100}'),
        ]
        hub.create_commit(repo_id="test-repo", operations=ops, commit_message="Add item 1")
        
        files = hub.list_files(repo_id="test-repo", prefix="clips/b1")
        assert sorted(files) == ["clips/b1/1/clip.mp4", "clips/b1/1/meta.json"]
        
        all_files = hub.list_files(repo_id="test-repo")
        assert "batches/b1/items.jsonl" in all_files
        assert len(all_files) == 3


def test_local_hub_download_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        hub = LocalHub(root_dir=tmpdir)
        ops = [
            CommitOperation(path_in_repo="test/hello.txt", content_bytes=b"hello world"),
        ]
        hub.create_commit(repo_id="test-repo", operations=ops, commit_message="Initial commit")
        
        download_target = Path(tmpdir) / "downloads" / "hello.txt"
        hub.download_file(repo_id="test-repo", path_in_repo="test/hello.txt", local_path=str(download_target))
        
        assert download_target.exists()
        assert download_target.read_bytes() == b"hello world"


def test_local_hub_delete_operation():
    with tempfile.TemporaryDirectory() as tmpdir:
        hub = LocalHub(root_dir=tmpdir)
        ops = [
            CommitOperation(path_in_repo="dir/file1.txt", content_bytes=b"1"),
            CommitOperation(path_in_repo="dir/file2.txt", content_bytes=b"2"),
        ]
        hub.create_commit(repo_id="test-repo", operations=ops, commit_message="Add two")
        assert len(hub.list_files(repo_id="test-repo")) == 2
        
        del_ops = [
            CommitOperation(path_in_repo="dir/file1.txt", delete=True),
        ]
        hub.create_commit(repo_id="test-repo", operations=del_ops, commit_message="Delete one")
        assert hub.list_files(repo_id="test-repo") == ["dir/file2.txt"]


def test_hf_hub_retry_on_500():
    from huggingface_hub.utils import HfHubHTTPError
    import requests

    with patch("huggingface_hub.HfApi") as mock_api_cls:
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api

        # Create a mock response with 500 error
        resp_500 = requests.Response()
        resp_500.status_code = 500
        err_500 = HfHubHTTPError("Server error", response=resp_500)

        # First call fails with 500, second call succeeds
        mock_api.create_commit.side_effect = [err_500, "commit_hash_123"]

        with patch("time.sleep") as mock_sleep:
            hub = HFHub(token="fake_token")
            ops = [CommitOperation(path_in_repo="test.txt", content_bytes=b"data")]
            res = hub.create_commit(repo_id="repo", operations=ops, commit_message="msg")

            assert res == "commit_hash_123"
            assert mock_api.create_commit.call_count == 2
            mock_sleep.assert_called_once()

