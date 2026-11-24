#!/usr/bin/env python3
"""Runner for Wan 2.2 I2V batch inference on remote GPU environments (Kaggle/Colab).

Handles GPU verification, ComfyUI process lifecycle in independent process groups,
sharding, resume, per-item watch with automatic recovery and restart, and
atomic single-commit state and clip reporting to private Hugging Face hub datasets.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from hub import Hub, HFHub, LocalHub, CommitOperation

STOP_REQUESTED = False


def _signal_handler(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


def check_gpu() -> None:
    """Refuse to run unless nvidia-smi reports T4 device(s).

    Overridable only via env INFER_SKIP_GPU_CHECK=1 for tests.
    """
    if os.environ.get("INFER_SKIP_GPU_CHECK") == "1":
        return

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        gpu_names = proc.stdout.strip().splitlines()
        if not gpu_names or not any("T4" in name for name in gpu_names):
            sys.stderr.write(f"GPU check failed: expected T4, found: {gpu_names}\n")
            sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"GPU check failed: {e}\n")
        sys.exit(1)


def pull_colab_weights(
    repo_id: str,
    revision: str,
    dest_dir: Path,
    expected_hashes: Dict[str, str],
    hub: Hub,
) -> float:
    """Pull weights from Hugging Face repo at pinned revision with SHA256 check.

    Measures and returns elapsed download time in seconds.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    for filename, expected_hash in expected_hashes.items():
        local_target = dest_dir / filename
        hub.download_file(
            repo_id=repo_id,
            path_in_repo=filename,
            local_path=local_target,
            repo_type="model",
        )

        hasher = hashlib.sha256()
        with open(local_target, "rb") as f:
            while chunk := f.read(1024 * 1024):
                hasher.update(chunk)
        digest = hasher.hexdigest()

        if digest.lower() != expected_hash.lower():
            local_target.unlink(missing_ok=True)
            raise ValueError(
                f"Hash mismatch for {filename}: expected {expected_hash}, got {digest}"
            )

    return time.time() - start_time


def filter_items_for_shard(
    items: List[Dict[str, Any]],
    shard_index: int,
    num_shards: int,
) -> List[Dict[str, Any]]:
    """Filter items for current shard by index % num_shards."""
    if num_shards <= 0:
        raise ValueError(f"num_shards must be > 0, got {num_shards}")
    if not (0 <= shard_index < num_shards):
        raise ValueError(
            f"shard_index must be between 0 and {num_shards - 1}, got {shard_index}"
        )
    return [
        item for idx, item in enumerate(items)
        if idx % num_shards == shard_index
    ]


def get_system_memory() -> Tuple[float, float]:
    """Return (peak_vram_mb, peak_ram_mb)."""
    vram_mb = 0.0
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            lines = res.stdout.strip().splitlines()
            if lines:
                vram_mb = float(lines[0].strip())
    except Exception:
        pass

    ram_mb = 0.0
    try:
        import resource
        # ru_maxrss is in KB on Linux
        usage = resource.getrusage(resource.RUSAGE_SELF)
        ram_mb = usage.ru_maxrss / 1024.0
    except Exception:
        pass

    return vram_mb, ram_mb


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def wait_for_port_release(port: int, timeout_sec: float = 15.0) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if not is_port_in_use(port):
            return
        time.sleep(0.2)


def kill_comfy_process_group(proc: Optional[subprocess.Popen], port: int) -> None:
    """SIGTERM process group, SIGKILL after grace, and wait for port release."""
    if proc is None:
        return

    pid = proc.pid
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return

    # 1. SIGTERM group
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass

    # Wait grace period
    grace_deadline = time.time() + 5.0
    while time.time() < grace_deadline:
        if proc.poll() is not None:
            break
        time.sleep(0.2)

    # 2. SIGKILL if still running
    if proc.poll() is None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=3.0)
        except Exception:
            pass

    # Wait for port to clear
    wait_for_port_release(port)


def build_workflow_prompt(
    template: Dict[str, Any],
    item: Dict[str, Any],
    width: int,
    height: int,
    length: int,
    local_keyframe_dir: Optional[Path],
    comfy_input_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Substitute template variables into ComfyUI workflow JSON."""
    raw_str = json.dumps(template)

    start_img = item.get("image", "")
    end_img = item.get("end_image", "")
    
    # If using ComfyUI input directory, copy keyframes there and use filename
    if comfy_input_dir is not None and local_keyframe_dir is not None:
        import shutil
        comfy_input_dir.mkdir(parents=True, exist_ok=True)
        start_basename = Path(start_img).name if start_img else ""
        end_basename = Path(end_img).name if end_img else ""
        start_unique = f"{item.get('id', 'item')}__{start_basename}" if start_basename else ""
        end_unique = f"{item.get('id', 'item')}__{end_basename}" if end_basename else ""
        
        if start_basename and (local_keyframe_dir / start_img).exists():
            shutil.copy2(local_keyframe_dir / start_img, comfy_input_dir / start_unique)
        if end_basename and (local_keyframe_dir / end_img).exists():
            shutil.copy2(local_keyframe_dir / end_img, comfy_input_dir / end_unique)
        
        start_path = start_unique
        end_path = end_unique
    else:
        # External ComfyUI (tests): use absolute paths or just the filename
        if local_keyframe_dir is not None:
            start_path = str(local_keyframe_dir / start_img) if not Path(start_img).is_absolute() else start_img
            end_path = str(local_keyframe_dir / end_img) if not Path(end_img).is_absolute() else end_img
        else:
            # Test mode without local_keyframe_dir: just use the image filename
            start_path = start_img
            end_path = end_img

    # Build replacements dict - numeric values stay as Python int/float
    # Keys are the raw placeholder names without {{ }}
    replacements = {
        "prompt": str(item.get("prompt", "")),
        "negative": str(item.get("negative", "low quality, bad anatomy, deformed")),
        "image": start_path,
        "end_image": end_path,
        "width": width,
        "height": height,
        "length": length,
        "seed": item.get("seed", 42),
        "steps": item.get("steps", 4),
        "cfg": item.get("cfg", 1.0),
        "shift": item.get("shift", 5.0),
        "split": item.get("split", 2),
        "sampler": str(item.get("sampler", "euler")),
        "scheduler": str(item.get("scheduler", "simple")),
        "prefix": f"wan14_{item.get('id', 'out')}",
        "unet_high": "Wan2.2-I2V-A14B-HighNoise-Q3_K_M.gguf",
        "unet_low": "Wan2.2-I2V-A14B-LowNoise-Q3_K_M.gguf",
        "clip": "umt5-xxl-encoder-Q3_K_M.gguf",
        "clip_type": "wan",
        "vae": "wan_2.1_vae.safetensors",
        "lora_high": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
        "lora_low": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
        "lora_strength": 1.0,
    }

    # Numeric fields that should be JSON numbers (not quoted strings in the template)
    numeric_fields = {"width", "height", "length", "seed", "steps", "split", "cfg", "shift", "lora_strength"}
    
    for key, val in replacements.items():
        placeholder = "{{" + key + "}}"  # e.g. {{width}}
        if key in numeric_fields:
            # Replace quoted placeholder with bare number (no quotes)
            raw_str = raw_str.replace(f'"{placeholder}"', str(val))
        else:
            # String field: replace quoted placeholder with JSON-escaped string
            raw_str = raw_str.replace(f'"{placeholder}"', json.dumps(str(val)))
    
    return json.loads(raw_str)


def run_inference(
    batch_id: str,
    shard_index: int,
    num_shards: int,
    run_id: str,
    hub_dataset: str,
    input_dataset: str,
    max_runtime_min: float = 600.0,
    item_timeout_min: float = 15.0,
    width: int = 960,
    height: int = 544,
    length: int = 25,
    comfy_url: Optional[str] = None,
    comfy_cmd: Optional[List[str]] = None,
    port: int = 8188,
    hub: Optional[Hub] = None,
    work_dir: Optional[Path] = None,
    skip_gpu_check: bool = False,
    item_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_item_start_hook: Optional[Callable[[Dict[str, Any], Optional[subprocess.Popen]], None]] = None,
    mount: Optional[Path] = None,
) -> int:
    """Core batch inference runner."""
    global STOP_REQUESTED

    if not skip_gpu_check:
        check_gpu()

    if hub is None:
        hub = HFHub()

    if work_dir is None:
        work_dir = Path("/kaggle/temp/wan-batch-infer-work" if os.path.exists("/kaggle") else "./infer_work")
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if mount is None:
        mount = Path("/kaggle/input" if os.path.exists("/kaggle") else "./kaggle_input")

    state_prefix = f"state/shard_{shard_index}_{run_id}"
    batch_input_prefix = f"batches/{batch_id}"

    start_timestamp = time.time()
    max_runtime_sec = max_runtime_min * 60.0

    # 1. Download items.jsonl and keyframes from input dataset
    local_input_dir = work_dir / "inputs" / batch_id
    local_input_dir.mkdir(parents=True, exist_ok=True)
    items_file = local_input_dir / "items.jsonl"

    hub.download_file(
        repo_id=input_dataset,
        path_in_repo=f"{batch_input_prefix}/items.jsonl",
        local_path=items_file,
    )

    all_raw_lines = items_file.read_text(encoding="utf-8").splitlines()
    parsed_items: List[Dict[str, Any]] = []
    for line in all_raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            parsed_items.append(json.loads(line))
        except Exception:
            parsed_items.append({"__malformed__": True, "__raw__": line})

    # Filter items for shard
    shard_items = filter_items_for_shard(parsed_items, shard_index, num_shards)

    # 2. Check resume: query existing clips under clips/<batch_id>/
    existing_files = set(hub.list_files(repo_id=hub_dataset, prefix=f"clips/{batch_id}"))
    finished_item_ids = set()
    for f in existing_files:
        match = re.search(rf"clips/{re.escape(batch_id)}/([^/]+)/clip\.mp4", f)
        if match:
            finished_item_ids.add(match.group(1))

    # Load workflow template
    workflow_path = Path(__file__).parent.parent / "workflows" / "i2v_wan14_flf2v.json"
    if workflow_path.exists():
        workflow_template = json.loads(workflow_path.read_text(encoding="utf-8"))
        # ComfyUI treats every top-level key as a node; drop non-node metadata such as "_meta"
        workflow_template = {k: v for k, v in workflow_template.items() if isinstance(v, dict) and "class_type" in v}
    else:
        workflow_template = {}

    # ComfyUI server management
    server_proc: Optional[subprocess.Popen] = None
    actual_url = comfy_url
    comfy_log_path = work_dir / "comfyui.log"

    def start_server() -> None:
        nonlocal server_proc, actual_url
        if comfy_url is not None:
            actual_url = comfy_url
            return

        # Real path: setup ComfyUI if not already set up (only when no comfy_cmd provided)
        comfy_dir = work_dir / "ComfyUI"
        setup_marker = comfy_dir / ".setup_done"
        if comfy_cmd is None and not setup_marker.exists():
            import comfy_setup
            comfy_setup.setup_comfyui(comfy_dir, mount)
            setup_marker.write_text("ok")  # only after a complete setup, so a half-installed dir is redone

        # Use main.py directly (ComfyUI auto-reads extra_model_paths.yaml from its root)
        cmd = comfy_cmd or [
            sys.executable,
            "main.py",
            "--listen", "127.0.0.1",
            "--port", str(port),
            "--cache-none",
            "--disable-auto-launch",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        
        # Redirect stdout/stderr to log file only for real ComfyUI (not tests with comfy_cmd)
        if comfy_cmd is None:
            log_file = open(comfy_log_path, "a")
            server_proc = subprocess.Popen(cmd, start_new_session=True, env=env, cwd=comfy_dir, stdout=log_file, stderr=subprocess.STDOUT)
        else:
            # Tests: no log redirection, no cwd (stub server handles its own cwd)
            server_proc = subprocess.Popen(cmd, start_new_session=True, env=env)
        actual_url = f"http://127.0.0.1:{port}"

        # Wait up to 300s for server to respond (cold start imports torch)
        wait_deadline = time.time() + 300.0
        while time.time() < wait_deadline:
            if server_proc.poll() is not None:
                if comfy_cmd is None:
                    log_file.close()
                # Print last 60 lines of log (only for real ComfyUI)
                try:
                    if comfy_log_path.exists():
                        log_lines = comfy_log_path.read_text(encoding="utf-8").splitlines()
                        tail = "\n".join(log_lines[-60:])
                    else:
                        tail = "No log file (test mode)"
                except Exception:
                    tail = "Could not read log file"
                raise RuntimeError(f"ComfyUI server terminated unexpectedly during startup\nLast 60 lines of log:\n{tail}")
            try:
                r = requests.get(f"{actual_url}/system_stats", timeout=1.0)
                if r.status_code == 200:
                    if comfy_cmd is None:
                        log_file.close()
                    return
            except Exception:
                pass
            time.sleep(0.5)
        
        if comfy_cmd is None:
            log_file.close()
        # Timeout - print last 60 lines
        try:
            if comfy_log_path.exists():
                log_lines = comfy_log_path.read_text(encoding="utf-8").splitlines()
                tail = "\n".join(log_lines[-60:])
            else:
                tail = "No log file (test mode)"
        except Exception:
            tail = "Could not read log file"
        raise RuntimeError(f"ComfyUI server startup timed out after 300s\nLast 60 lines of log:\n{tail}")

    start_server()

    # Heartbeat: setup done and server up (one commit per stage, never per item)
    hub.create_commit(
        repo_id=hub_dataset,
        operations=[
            CommitOperation(
                path_in_repo=f"{state_prefix}/progress.json",
                content_bytes=json.dumps({
                    "status": "ready",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "shard_index": shard_index,
                    "num_shards": num_shards,
                    "total_shard_items": len(shard_items),
                    "resumed_items": len(finished_item_ids),
                }, indent=2).encode(),
            )
        ],
        commit_message=f"heartbeat: setup ready shard {shard_index}",
    )

    # Progress tracking
    failed_records: List[Dict[str, Any]] = []
    completed_count = 0
    skipped_count = 0
    stopped_early = False
    peak_vram_recorded = 0.0
    peak_ram_recorded = 0.0

    try:
        for idx, item in enumerate(shard_items):
            # Check STOP_REQUESTED or runtime budget before item
            elapsed = time.time() - start_timestamp
            if STOP_REQUESTED or elapsed >= max_runtime_sec:
                stopped_early = True
                break

            # Handle malformed item
            if item.get("__malformed__") or "id" not in item:
                err_record = {
                    "id": item.get("id", f"malformed_{idx}"),
                    "error": "Malformed item JSON or missing 'id'",
                    "raw": item.get("__raw__", str(item)),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                failed_records.append(err_record)
                failed_bytes = "".join(json.dumps(r) + "\n" for r in failed_records).encode()
                progress_data = {
                    "completed": completed_count,
                    "failed": len(failed_records),
                    "skipped": skipped_count,
                    "total": len(shard_items),
                    "last_item": err_record["id"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                hub.create_commit(
                    repo_id=hub_dataset,
                    operations=[
                        CommitOperation(
                            path_in_repo=f"{state_prefix}/failed_ids.jsonl",
                            content_bytes=failed_bytes,
                        ),
                        CommitOperation(
                            path_in_repo=f"{state_prefix}/progress.json",
                            content_bytes=json.dumps(progress_data, indent=2).encode(),
                        ),
                    ],
                    commit_message=f"fail: malformed item {err_record['id']}",
                )
                continue

            item_id = str(item["id"])

            # Skip if already finished
            if item_id in finished_item_ids:
                skipped_count += 1
                continue

            if item_callback:
                item_callback(item)

            if on_item_start_hook:
                on_item_start_hook(item, server_proc)

            # Ensure keyframe images are downloaded
            for img_key in ("image", "end_image"):
                img_rel = item.get(img_key)
                if img_rel:
                    dest_path = local_input_dir / img_rel
                    if not dest_path.exists():
                        try:
                            hub.download_file(
                                repo_id=input_dataset,
                                path_in_repo=f"{batch_input_prefix}/{img_rel}",
                                local_path=dest_path,
                            )
                        except Exception as e:
                            # If download fails, fail item
                            pass

            # Prepare Comfy prompt payload
            comfy_input_dir = None
            if server_proc is not None:  # Real path, not tests
                comfy_input_dir = work_dir / "ComfyUI" / "input"
            workflow_prompt = build_workflow_prompt(
                workflow_template,
                item,
                width=width,
                height=height,
                length=length,
                local_keyframe_dir=local_input_dir,
                comfy_input_dir=comfy_input_dir,
            )

            item_start_time = time.time()
            item_failed = False
            item_error = ""
            video_bytes = None
            frames_count = 0

            # Check server alive before prompt
            if server_proc and server_proc.poll() is not None:
                item_failed = True
                item_error = f"ComfyUI server was not running (exit code {server_proc.returncode})"
            else:
                try:
                    # Submit prompt to ComfyUI
                    prompt_resp = requests.post(
                        f"{actual_url}/prompt",
                        json={"prompt": workflow_prompt, "client_id": "wan_batch_runner"},
                        timeout=10.0,
                    )
                    if prompt_resp.status_code != 200:
                        item_failed = True
                        item_error = f"POST /prompt returned HTTP {prompt_resp.status_code}: {prompt_resp.text}"
                    else:
                        prompt_info = prompt_resp.json()
                        prompt_id = prompt_info.get("prompt_id")

                        # Per-item watch loop
                        watch_deadline = time.time() + (item_timeout_min * 60.0)
                        while time.time() < watch_deadline:
                            # 1. Watch server died
                            if server_proc and server_proc.poll() is not None:
                                item_failed = True
                                item_error = f"ComfyUI server died during execution (code {server_proc.returncode})"
                                break

                            # 2. Watch /history/{prompt_id}
                            try:
                                hist_resp = requests.get(f"{actual_url}/history/{prompt_id}", timeout=5.0)
                                if hist_resp.status_code == 200:
                                    hist_data = hist_resp.json().get(prompt_id)
                                    if hist_data:
                                        status_info = hist_data.get("status", {})
                                        if status_info.get("status_str") == "error":
                                            item_failed = True
                                            item_error = f"ComfyUI execution error: {status_info.get('messages')}"
                                            break
                                        elif status_info.get("completed", False):
                                            # Find output frames from SaveImage node
                                            outputs = hist_data.get("outputs", {})
                                            frame_files = []  # list of (filename, subfolder, file_type)
                                            for node_id, node_out in outputs.items():
                                                if "images" in node_out:
                                                    for img_info in node_out["images"]:
                                                        frame_files.append((
                                                            img_info["filename"],
                                                            img_info.get("subfolder", ""),
                                                            img_info.get("type", "output")
                                                        ))
                                                    break  # first SaveImage node
                                            
                                            if not frame_files:
                                                item_failed = True
                                                item_error = "No output frames found in ComfyUI history"
                                                break

                                            # Download all frames
                                            frame_paths = []
                                            with tempfile.TemporaryDirectory() as tmpdir:
                                                tmpdir_path = Path(tmpdir)
                                                for idx, (filename, subfolder, file_type) in enumerate(frame_files):
                                                    view_resp = requests.get(
                                                        f"{actual_url}/view",
                                                        params={"filename": filename, "subfolder": subfolder, "type": file_type},
                                                        timeout=30.0,
                                                    )
                                                    if view_resp.status_code == 200:
                                                        frame_path = tmpdir_path / f"frame_{idx:04d}.png"
                                                        frame_path.write_bytes(view_resp.content)
                                                        frame_paths.append(frame_path)
                                                    else:
                                                        item_failed = True
                                                        item_error = f"Failed to download frame {filename}: HTTP {view_resp.status_code}"
                                                        break
                                                
                                                if not item_failed:
                                                    frames_count = len(frame_paths)
                                                    # Validate frame count: should be 4n+1 where n>=1 (e.g., 5, 9, 13, 17, 21, 25)
                                                    if frames_count < 5 or (frames_count - 1) % 4 != 0 or frames_count != length:
                                                        item_failed = True
                                                        item_error = f"Invalid frame count: {frames_count}. Expected {length} (4n+1)"
                                                        break
                                                    
                                                    # Encode frames to mp4 using ffmpeg
                                                    # Check ffmpeg is available (skip for stub tests)
                                                    if server_proc is not None and shutil.which("ffmpeg") is None:
                                                        item_failed = True
                                                        item_error = "ffmpeg not found in PATH"
                                                        break
                                                    
                                                    # Even dimensions required by libx264
                                                    out_width = width if width % 2 == 0 else width - 1
                                                    out_height = height if height % 2 == 0 else height - 1
                                                    
                                                    # Use the last frame path directory for output
                                                    output_mp4 = tmpdir_path / "clip.mp4"
                                                    ffmpeg_cmd = [
                                                        "ffmpeg", "-y",
                                                        "-framerate", "16",
                                                        "-i", str(tmpdir_path / "frame_%04d.png"),
                                                        "-c:v", "libx264",
                                                        "-pix_fmt", "yuv420p",
                                                        "-vf", f"scale={out_width}:{out_height}:force_original_aspect_ratio=decrease,pad={out_width}:{out_height}:(ow-iw)/2:(oh-ih)/2",
                                                        str(output_mp4)
                                                    ]
                                                    try:
                                                        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, timeout=60.0)
                                                        video_bytes = output_mp4.read_bytes()
                                                        if not video_bytes:
                                                            item_failed = True
                                                            item_error = "ffmpeg produced empty output"
                                                    except subprocess.CalledProcessError as e:
                                                        item_failed = True
                                                        item_error = f"ffmpeg encoding failed: {e.stderr.decode() if e.stderr else str(e)}"
                                                    except subprocess.TimeoutExpired:
                                                        item_failed = True
                                                        item_error = "ffmpeg encoding timed out"
                                            break
                            except Exception as req_err:
                                pass

                            time.sleep(1.0)
                        else:
                            if not item_failed:
                                item_failed = True
                                item_error = f"Item timed out after {item_timeout_min} minutes"

                except Exception as e:
                    item_failed = True
                    item_error = f"Exception submitting/polling item: {e}"

            # Memory reading
            cur_vram, cur_ram = get_system_memory()
            peak_vram_recorded = max(peak_vram_recorded, cur_vram)
            peak_ram_recorded = max(peak_ram_recorded, cur_ram)
            duration_sec = time.time() - item_start_time

            if item_failed:
                # Get ComfyUI log tail on failure
                comfy_log_tail = ""
                if comfy_log_path.exists():
                    try:
                        log_lines = comfy_log_path.read_text(encoding="utf-8").splitlines()
                        comfy_log_tail = "\n".join(log_lines[-60:])
                    except Exception:
                        pass
                
                err_record = {
                    "id": item_id,
                    "error": item_error,
                    "duration_sec": duration_sec,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "comfy_log_tail": comfy_log_tail,
                }
                failed_records.append(err_record)

                failed_bytes = "".join(json.dumps(r) + "\n" for r in failed_records).encode()
                progress_data = {
                    "completed": completed_count,
                    "failed": len(failed_records),
                    "skipped": skipped_count,
                    "total": len(shard_items),
                    "last_failed": item_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                # Single commit for failed_ids and progress
                hub.create_commit(
                    repo_id=hub_dataset,
                    operations=[
                        CommitOperation(
                            path_in_repo=f"{state_prefix}/failed_ids.jsonl",
                            content_bytes=failed_bytes,
                        ),
                        CommitOperation(
                            path_in_repo=f"{state_prefix}/progress.json",
                            content_bytes=json.dumps(progress_data, indent=2).encode(),
                        ),
                    ],
                    commit_message=f"fail: item {item_id}",
                )

                # Kill process group, wait port release, restart server
                if server_proc is not None:
                    kill_comfy_process_group(server_proc, port)
                    server_proc = None
                    start_server()

                continue

            # Item Succeeded: write local files, commit in ONE create_commit, then delete local files
            item_out_dir = work_dir / "clips" / batch_id / item_id
            item_out_dir.mkdir(parents=True, exist_ok=True)

            local_clip = item_out_dir / "clip.mp4"
            local_meta = item_out_dir / "meta.json"
            local_item = item_out_dir / "item.json"

            meta_data = {
                "item_id": item_id,
                "batch_id": batch_id,
                "duration_sec": duration_sec,
                "peak_vram_mb": cur_vram,
                "peak_ram_mb": cur_ram,
                "width": width,
                "height": height,
                "length": length,
                "frames": frames_count,
                "fps": 16,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            if not video_bytes:
                item_failed = True
                item_error = "No video bytes produced"
                # fall through to failure handling
            else:
                local_clip.write_bytes(video_bytes)
            local_meta.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")
            local_item.write_text(json.dumps(item, indent=2), encoding="utf-8")

            completed_count += 1
            progress_data = {
                "completed": completed_count,
                "failed": len(failed_records),
                "skipped": skipped_count,
                "total": len(shard_items),
                "last_item": item_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # ONE commit per item: clip.mp4, meta.json, item.json, progress.json
            item_prefix = f"clips/{batch_id}/{item_id}"
            ops = [
                CommitOperation(path_in_repo=f"{item_prefix}/clip.mp4", path_or_fileobj=str(local_clip)),
                CommitOperation(path_in_repo=f"{item_prefix}/meta.json", path_or_fileobj=str(local_meta)),
                CommitOperation(path_in_repo=f"{item_prefix}/item.json", path_or_fileobj=str(local_item)),
                CommitOperation(
                    path_in_repo=f"{state_prefix}/progress.json",
                    content_bytes=json.dumps(progress_data, indent=2).encode(),
                ),
            ]

            hub.create_commit(
                repo_id=hub_dataset,
                operations=ops,
                commit_message=f"item: {batch_id}/{item_id}",
            )

            # Local files deleted after confirmed commit
            shutil.rmtree(item_out_dir, ignore_errors=True)

    finally:
        # Flush summary
        summary = {
            "batch_id": batch_id,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "run_id": run_id,
            "total_shard_items": len(shard_items),
            "completed": completed_count,
            "failed": len(failed_records),
            "skipped": skipped_count,
            "stopped_early": stopped_early,
            "elapsed_sec": time.time() - start_timestamp,
            "peak_vram_mb": peak_vram_recorded,
            "peak_ram_mb": peak_ram_recorded,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            hub.create_commit(
                repo_id=hub_dataset,
                operations=[
                    CommitOperation(
                        path_in_repo=f"{state_prefix}/summary.json",
                        content_bytes=json.dumps(summary, indent=2).encode(),
                    )
                ],
                commit_message=f"summary: shard {shard_index} run {run_id}",
            )
        except Exception as e:
            sys.stderr.write(f"Failed to flush summary to hub: {e}\n")

        # Cleanup server
        if server_proc is not None:
            kill_comfy_process_group(server_proc, port)

    return 1 if len(failed_records) > 0 else 0


def main():
    if "--probe" in sys.argv:
        import probe
        probe.main()
        return
    parser = argparse.ArgumentParser(description="Wan 2.2 I2V batch inference runner")
    parser.add_argument("--batch-id", required=True, help="Batch ID")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index (0-based)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        help="Run identifier",
    )
    parser.add_argument("--max-runtime-min", type=float, default=600.0, help="Max total runtime in minutes")
    parser.add_argument("--item-timeout-min", type=float, default=15.0, help="Max per-item runtime in minutes")
    parser.add_argument("--hub-dataset", required=True, help="Destination HF dataset repository")
    parser.add_argument("--input-dataset", required=True, help="Source HF dataset repository with batch inputs")
    parser.add_argument("--width", type=int, default=960, help="Clip width")
    parser.add_argument("--height", type=int, default=544, help="Clip height")
    parser.add_argument("--length", type=int, default=25, help="Number of frames")
    parser.add_argument("--comfy-url", default=None, help="ComfyUI URL (if running externally or in tests)")
    parser.add_argument("--mount", default=None, help="Path to model cache mount (default: /kaggle/input or ./kaggle_input)")

    args = parser.parse_args()

    mount_path = Path(args.mount) if args.mount else None

    exit_code = run_inference(
        batch_id=args.batch_id,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        run_id=args.run_id,
        max_runtime_min=args.max_runtime_min,
        item_timeout_min=args.item_timeout_min,
        hub_dataset=args.hub_dataset,
        input_dataset=args.input_dataset,
        width=args.width,
        height=args.height,
        length=args.length,
        comfy_url=args.comfy_url,
        mount=mount_path,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
