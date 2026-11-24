"""Submission-1 probe: no model run. Verifies GPU, mount, constraints, secrets, one hub commit.

Writes result.json next to run.log and exits non-zero if any check failed.
"""
import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comfy_setup  # noqa: E402
from hub import CommitOperation, HFHub  # noqa: E402


def _check(results, name, fn):
    t0 = time.time()
    try:
        results[name] = {"ok": True, "detail": fn(), "sec": round(time.time() - t0, 2)}
    except Exception as e:  # recorded, never hidden: probe exits non-zero at the end
        results[name] = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}


def gpu():
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()
    if not any("T4" in n for n in out):
        raise RuntimeError(f"no T4: {out}")
    return out


def versions():
    v = {}
    for p in ("torch", "numpy", "opencv-python", "opencv-python-headless", "pillow"):
        try:
            v[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            pass
    return v


def mount(mount_dir):
    root = Path(tempfile.mkdtemp(prefix="probe_comfy_"))
    yaml_path = comfy_setup.write_extra_model_paths(root, Path(mount_dir))
    found = {}
    for line in yaml_path.read_text().splitlines():
        pass
    for dirpath, _, names in os.walk(mount_dir):
        for n in names:
            if any(n in files for files in comfy_setup.REQUIRED_MODEL_FILES.values()):
                p = Path(dirpath) / n
                with open(p, "rb") as f:
                    f.read(1 << 20)
                found[n] = p.stat().st_size
    missing = [n for fs in comfy_setup.REQUIRED_MODEL_FILES.values() for n in fs if n not in found]
    if missing:
        raise RuntimeError(f"missing: {missing}")
    return {"yaml": yaml_path.read_text()[:800], "files": found}


def constraints():
    text = comfy_setup.build_constraints()
    lines = [l for l in text.splitlines() if l.strip()]
    return lines


def secrets():
    return {k: bool(os.environ.get(k)) for k in ("HF_TOKEN", "GITHUB_TOKEN")}


def hub_commit(dataset, run_id):
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN missing")
    payload = json.dumps({"probe": run_id, "time": time.time()}).encode()
    sha = HFHub().create_commit(dataset, [CommitOperation(path_in_repo=f"probe/{run_id}.json", content_bytes=payload)],
                                f"probe {run_id}")
    return str(sha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--hub-dataset", required=True)
    ap.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--mount", default="/kaggle/input")
    ap.add_argument("--result", default="/kaggle/working/result.json")
    a = ap.parse_args()
    res = {}
    _check(res, "gpu", gpu)
    _check(res, "versions_before", versions)
    _check(res, "constraints", constraints)
    _check(res, "mount", lambda: mount(a.mount))
    _check(res, "secrets", secrets)
    if not res["secrets"].get("detail", {}).get("HF_TOKEN"):
        res["secrets"]["ok"] = False
    if not res["secrets"].get("detail", {}).get("GITHUB_TOKEN"):
        res["secrets"]["ok"] = False
    _check(res, "hub_commit", lambda: hub_commit(a.hub_dataset, a.run_id))
    _check(res, "versions_after", versions)
    res["ok"] = all(v.get("ok") for v in res.values() if isinstance(v, dict))
    try:
        Path(a.result).parent.mkdir(parents=True, exist_ok=True)
        Path(a.result).write_text(json.dumps(res, indent=1))
    except OSError as e:
        print("could not write result:", e)
    print(json.dumps(res, indent=1))
    # also publish the result to the hub so success is visible there
    try:
        HFHub().create_commit(a.hub_dataset, [CommitOperation(path_in_repo=f"probe/{a.run_id}_result.json",
                              content_bytes=json.dumps(res, indent=1).encode())], f"probe result {a.run_id}")
    except Exception as e:
        print("result commit failed:", e)
        res["ok"] = False
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
