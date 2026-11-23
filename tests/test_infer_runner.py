"""Comprehensive tests for src/infer.py runner.

Covers:
- GPU check (T4 required, overridable by env)
- Colab weights puller with hash check
- Shard filtering
- ONE commit per item to LocalHub (clip, meta.json, item.json, progress.json)
- Local files deleted after confirmed commit
- Rerun skips finished ids
- Clean stop on STOP_REQUESTED / runtime budget between items
- Malformed items land in failed_ids.jsonl and loop continues
- Server killed (proc.poll()) triggers process group kill and restart
- Server returning error in /history triggers process group kill and restart
"""

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hub import LocalHub, CommitOperation
import infer


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def test_gpu_check_refuses_without_t4():
    """GPU check must refuse unless nvidia-smi reports T4."""
    with patch.dict(os.environ, {}, clear=True):
        if "INFER_SKIP_GPU_CHECK" in os.environ:
            del os.environ["INFER_SKIP_GPU_CHECK"]

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["nvidia-smi"], returncode=0, stdout="Tesla V100-SXM2-16GB\n"
            )
            with pytest.raises(SystemExit) as exc_info:
                infer.check_gpu()
            assert exc_info.value.code != 0

        # When T4 is present (even two T4s)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["nvidia-smi"], returncode=0, stdout="Tesla T4\nTesla T4\n"
            )
            infer.check_gpu()


def test_gpu_check_overridden_by_env():
    """GPU check can be overridden via INFER_SKIP_GPU_CHECK=1."""
    with patch.dict(os.environ, {"INFER_SKIP_GPU_CHECK": "1"}):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("No nvidia-smi")
            infer.check_gpu()


def test_colab_weights_download_and_hash():
    """Colab weight puller downloads files, verifies sha256, and measures time."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hub = LocalHub(root_dir=tmpdir)
        content = b"model weights binary data"
        sha256 = hashlib.sha256(content).hexdigest()

        ops = [
            CommitOperation(path_in_repo="weights.bin", content_bytes=content)
        ]
        hub.create_commit("weights-repo", ops, "add weights")

        dest_dir = Path(tmpdir) / "models"
        elapsed = infer.pull_colab_weights(
            repo_id="weights-repo",
            revision="main",
            dest_dir=dest_dir,
            expected_hashes={"weights.bin": sha256},
            hub=hub,
        )
        assert elapsed >= 0.0
        assert (dest_dir / "weights.bin").read_bytes() == content

        # Corrupted hash must fail loudly
        with pytest.raises(ValueError, match="Hash mismatch"):
            infer.pull_colab_weights(
                repo_id="weights-repo",
                revision="main",
                dest_dir=dest_dir,
                expected_hashes={"weights.bin": "wronghash"},
                hub=hub,
            )


def test_shard_filter():
    """Shard filter partitions items by index % num_shards."""
    items = [{"id": f"item_{i}"} for i in range(10)]
    shard_0 = infer.filter_items_for_shard(items, shard_index=0, num_shards=3)
    shard_1 = infer.filter_items_for_shard(items, shard_index=1, num_shards=3)
    shard_2 = infer.filter_items_for_shard(items, shard_index=2, num_shards=3)

    assert [x["id"] for x in shard_0] == ["item_0", "item_3", "item_6", "item_9"]
    assert [x["id"] for x in shard_1] == ["item_1", "item_4", "item_7"]
    assert [x["id"] for x in shard_2] == ["item_2", "item_5", "item_8"]


def _setup_test_dataset(hub: LocalHub, input_repo: str, batch_id: str, items: list):
    """Helper to populate input dataset with items.jsonl and dummy keyframe images."""
    items_bytes = "".join(json.dumps(it) + "\n" for it in items).encode("utf-8")
    ops = [
        CommitOperation(
            path_in_repo=f"batches/{batch_id}/items.jsonl",
            content_bytes=items_bytes,
        ),
        CommitOperation(
            path_in_repo=f"batches/{batch_id}/keyframes/start.jpg",
            content_bytes=b"fake_jpg_start",
        ),
        CommitOperation(
            path_in_repo=f"batches/{batch_id}/keyframes/end.jpg",
            content_bytes=b"fake_jpg_end",
        ),
    ]
    hub.create_commit(input_repo, ops, "setup input batch")


def test_infer_runner_one_commit_per_item_and_local_cleanup():
    """Runner commits clip.mp4, meta.json, item.json, progress.json in ONE commit per item and deletes local files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hub_root = Path(tmpdir) / "hub"
        hub = LocalHub(root_dir=hub_root)
        work_dir = Path(tmpdir) / "work"
        work_dir.mkdir()

        batch_id = "b1"
        items = [
            {"id": "item1", "prompt": "prompt 1", "image": "keyframes/start.jpg", "end_image": "keyframes/end.jpg"},
            {"id": "item2", "prompt": "prompt 2", "image": "keyframes/start.jpg", "end_image": "keyframes/end.jpg"},
        ]
        _setup_test_dataset(hub, "input-dataset", batch_id, items)

        # Start stub comfy server
        stub_script = str(Path(__file__).parent / "stub_comfy.py")
        port = get_free_port()
        server_proc = subprocess.Popen(
            [sys.executable, stub_script, "--port", str(port)],
            start_new_session=True,
        )
        time.sleep(0.5)

        try:
            commit_log = []
            orig_create_commit = hub.create_commit

            def tracking_create_commit(repo_id, operations, commit_message, repo_type="dataset"):
                commit_log.append({
                    "repo_id": repo_id,
                    "msg": commit_message,
                    "ops": [op.path_in_repo for op in operations],
                })
                return orig_create_commit(repo_id, operations, commit_message, repo_type)

            hub.create_commit = tracking_create_commit

            exit_code = infer.run_inference(
                batch_id=batch_id,
                shard_index=0,
                num_shards=1,
                run_id="run1",
                hub_dataset="output-dataset",
                input_dataset="input-dataset",
                comfy_url=f"http://127.0.0.1:{port}",
                hub=hub,
                work_dir=work_dir,
                skip_gpu_check=True,
            )

            assert exit_code == 0

            # Check commits
            # Should have setup heartbeat (at most one) + 2 item commits + 1 summary commit
            item_commits = [c for c in commit_log if "item" in c["msg"].lower()]
            assert len(item_commits) == 2

            # Each item commit must have clip.mp4, meta.json, item.json, progress.json
            for c in item_commits:
                paths = c["ops"]
                assert any(p.endswith("clip.mp4") for p in paths)
                assert any(p.endswith("meta.json") for p in paths)
                assert any(p.endswith("item.json") for p in paths)
                assert any(p.endswith("progress.json") for p in paths)

            # Local generated clips must be deleted after confirmed commit
            local_clips = list((work_dir / "clips").rglob("clip.mp4"))
            assert len(local_clips) == 0

            # Hub must have both items
            hub_files = hub.list_files("output-dataset")
            assert f"clips/{batch_id}/item1/clip.mp4" in hub_files
            assert f"clips/{batch_id}/item1/meta.json" in hub_files
            assert f"clips/{batch_id}/item1/item.json" in hub_files
            assert f"clips/{batch_id}/item2/clip.mp4" in hub_files
            assert "state/shard_0_run1/progress.json" in hub_files
            assert "state/shard_0_run1/summary.json" in hub_files

        finally:
            os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
            server_proc.wait()


def test_infer_runner_resume_skips_finished_ids():
    """Rerun skips item_ids already present under clips/<batch_id>/."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hub_root = Path(tmpdir) / "hub"
        hub = LocalHub(root_dir=hub_root)
        work_dir = Path(tmpdir) / "work"

        batch_id = "b2"
        items = [
            {"id": "item1", "prompt": "prompt 1", "image": "keyframes/start.jpg", "end_image": "keyframes/end.jpg"},
            {"id": "item2", "prompt": "prompt 2", "image": "keyframes/start.jpg", "end_image": "keyframes/end.jpg"},
        ]
        _setup_test_dataset(hub, "input-dataset", batch_id, items)

        # Pre-populate item1 on output dataset
        hub.create_commit("output-dataset", [
            CommitOperation(path_in_repo=f"clips/{batch_id}/item1/clip.mp4", content_bytes=b"dummy_mp4")
        ], "existing item1")

        stub_script = str(Path(__file__).parent / "stub_comfy.py")
        port = 8191
        server_proc = subprocess.Popen(
            [sys.executable, stub_script, "--port", str(port)],
            start_new_session=True,
        )
        time.sleep(0.5)

        try:
            processed_items = []

            exit_code = infer.run_inference(
                batch_id=batch_id,
                shard_index=0,
                num_shards=1,
                run_id="run2",
                hub_dataset="output-dataset",
                input_dataset="input-dataset",
                comfy_url=f"http://127.0.0.1:{port}",
                hub=hub,
                work_dir=work_dir,
                skip_gpu_check=True,
                item_callback=lambda item: processed_items.append(item["id"]),
            )

            assert exit_code == 0
            # Only item2 should have been processed, item1 was skipped
            assert processed_items == ["item2"]

        finally:
            os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
            server_proc.wait()


def test_infer_runner_clean_stop_on_runtime_budget():
    """Runtime budget exceeded stops cleanly between items and writes summary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hub_root = Path(tmpdir) / "hub"
        hub = LocalHub(root_dir=hub_root)
        work_dir = Path(tmpdir) / "work"

        batch_id = "b3"
        items = [
            {"id": "item1", "prompt": "prompt 1", "image": "keyframes/start.jpg", "end_image": "keyframes/end.jpg"},
            {"id": "item2", "prompt": "prompt 2", "image": "keyframes/start.jpg", "end_image": "keyframes/end.jpg"},
            {"id": "item3", "prompt": "prompt 3", "image": "keyframes/start.jpg", "end_image": "keyframes/end.jpg"},
        ]
        _setup_test_dataset(hub, "input-dataset", batch_id, items)

        stub_script = str(Path(__file__).parent / "stub_comfy.py")
        port = 8192
        server_proc = subprocess.Popen(
            [sys.executable, stub_script, "--port", str(port)],
            start_new_session=True,
        )
        time.sleep(0.5)

        try:
            # Set max_runtime_min very small (0.0001 min ~ 0.006s) so budget is exceeded after first item
            exit_code = infer.run_inference(
                batch_id=batch_id,
                shard_index=0,
                num_shards=1,
                run_id="run3",
                hub_dataset="output-dataset",
                input_dataset="input-dataset",
                max_runtime_min=0.0001,
                comfy_url=f"http://127.0.0.1:{port}",
                hub=hub,
                work_dir=work_dir,
                skip_gpu_check=True,
            )

            # Stopped cleanly
            hub_files = hub.list_files("output-dataset")
            assert "state/shard_0_run3/summary.json" in hub_files

            # Summary should record stopped/completed counts
            summary_bytes = (hub_root / "output-dataset" / "state/shard_0_run3/summary.json").read_bytes()
            summary = json.loads(summary_bytes.decode())
            assert summary["completed"] >= 1
            assert summary["stopped_early"] is True

        finally:
            os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
            server_proc.wait()


def test_infer_runner_malformed_item_lands_in_failed_ids():
    """Malformed item lands in failed_ids.jsonl, committed together with progress.json, and loop continues."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hub_root = Path(tmpdir) / "hub"
        hub = LocalHub(root_dir=hub_root)
        work_dir = Path(tmpdir) / "work"

        batch_id = "b4"
        items = [
            {"id": "good1", "prompt": "p1", "image": "keyframes/start.jpg", "end_image": "keyframes/end.jpg"},
            {"bad_json_no_id": True},  # malformed
            {"id": "good2", "prompt": "p2", "image": "keyframes/start.jpg", "end_image": "keyframes/end.jpg"},
        ]
        _setup_test_dataset(hub, "input-dataset", batch_id, items)

        stub_script = str(Path(__file__).parent / "stub_comfy.py")
        port = 8193
        server_proc = subprocess.Popen(
            [sys.executable, stub_script, "--port", str(port)],
            start_new_session=True,
        )
        time.sleep(0.5)

        try:
            exit_code = infer.run_inference(
                batch_id=batch_id,
                shard_index=0,
                num_shards=1,
                run_id="run4",
                hub_dataset="output-dataset",
                input_dataset="input-dataset",
                comfy_url=f"http://127.0.0.1:{port}",
                hub=hub,
                work_dir=work_dir,
                skip_gpu_check=True,
            )

            # Should exit non-zero because failed_ids > 0
            assert exit_code != 0

            hub_files = hub.list_files("output-dataset")
            assert f"clips/{batch_id}/good1/clip.mp4" in hub_files
            assert f"clips/{batch_id}/good2/clip.mp4" in hub_files
            assert "state/shard_0_run4/failed_ids.jsonl" in hub_files

            failed_content = (hub_root / "output-dataset" / "state/shard_0_run4/failed_ids.jsonl").read_text()
            assert "malformed" in failed_content.lower() or "missing" in failed_content.lower()

        finally:
            os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
            server_proc.wait()


def test_infer_runner_stub_error_triggers_group_kill_and_restart():
    """Stub error in /history status triggers process group kill, restart, appends failed_ids, and continues."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hub_root = Path(tmpdir) / "hub"
        hub = LocalHub(root_dir=hub_root)
        work_dir = Path(tmpdir) / "work"

        batch_id = "b5"
        items = [
            {"id": "fail_item", "prompt": "TRIGGER_ERROR", "image": "keyframes/start.jpg", "end_image": "keyframes/end.jpg"},
            {"id": "good_item", "prompt": "normal prompt", "image": "keyframes/start.jpg", "end_image": "keyframes/end.jpg"},
        ]
        _setup_test_dataset(hub, "input-dataset", batch_id, items)

        stub_script = str(Path(__file__).parent / "stub_comfy.py")
        port = 8194
        comfy_cmd = [sys.executable, stub_script, "--port", str(port)]

        exit_code = infer.run_inference(
            batch_id=batch_id,
            shard_index=0,
            num_shards=1,
            run_id="run5",
            hub_dataset="output-dataset",
            input_dataset="input-dataset",
            comfy_cmd=comfy_cmd,
            port=port,
            hub=hub,
            work_dir=work_dir,
            skip_gpu_check=True,
        )

        # Non-zero due to failure
        assert exit_code != 0

        hub_files = hub.list_files("output-dataset")
        # good_item succeeded after restart
        assert f"clips/{batch_id}/good_item/clip.mp4" in hub_files
        assert "state/shard_0_run5/failed_ids.jsonl" in hub_files

        failed_content = (hub_root / "output-dataset" / "state/shard_0_run5/failed_ids.jsonl").read_text()
        assert "fail_item" in failed_content


def test_infer_runner_stub_proc_killed_triggers_restart():
    """Server process termination (proc.poll()) triggers restart, logs failure, and continues."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hub_root = Path(tmpdir) / "hub"
        hub = LocalHub(root_dir=hub_root)
        work_dir = Path(tmpdir) / "work"

        batch_id = "b6"
        items = [
            {"id": "item1", "prompt": "p1", "image": "keyframes/start.jpg", "end_image": "keyframes/end.jpg"},
            {"id": "item2", "prompt": "p2", "image": "keyframes/start.jpg", "end_image": "keyframes/end.jpg"},
        ]
        _setup_test_dataset(hub, "input-dataset", batch_id, items)

        stub_script = str(Path(__file__).parent / "stub_comfy.py")
        port = 8195
        comfy_cmd = [sys.executable, stub_script, "--port", str(port)]

        # Hook into item execution to kill server on first item
        def kill_on_first(item, server_proc):
            if item["id"] == "item1" and server_proc and server_proc.poll() is None:
                os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)

        exit_code = infer.run_inference(
            batch_id=batch_id,
            shard_index=0,
            num_shards=1,
            run_id="run6",
            hub_dataset="output-dataset",
            input_dataset="input-dataset",
            comfy_cmd=comfy_cmd,
            port=port,
            hub=hub,
            work_dir=work_dir,
            skip_gpu_check=True,
            on_item_start_hook=kill_on_first,
        )

        assert exit_code != 0
        hub_files = hub.list_files("output-dataset")
        assert f"clips/{batch_id}/item2/clip.mp4" in hub_files
        assert "state/shard_0_run6/failed_ids.jsonl" in hub_files
        failed_content = (hub_root / "output-dataset" / "state/shard_0_run6/failed_ids.jsonl").read_text()
        assert "item1" in failed_content
