#!/usr/bin/env python3
"""Run existing hq tools unchanged on a cloud GPU host from a bundle published to the hub.

The bundle (built by cloudjob) holds: the tools' code, input files, env.json (ComfyUI pins, models,
pip packages, steps) and any local-only ComfyUI custom nodes. This script rebuilds that environment,
runs each step's own runner argv, and uploads every step's output dir plus a DONE.json to the hub.
Success is what lands on the hub; DONE.json records exit codes and file hashes.
"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, tarfile, time, traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
LOCAL_HQ = "/home/user/hq"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def pick_root() -> Path:
    for base in ("/kaggle/temp", "/content", "/tmp"):
        if Path(base).exists():
            return Path(base) / "hq"
    return Path("/tmp/hq")


def extract(bundle: Path, root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle) as t:
        for m in t.getmembers():  # refuse path escapes
            if m.name.startswith("/") or ".." in Path(m.name).parts:
                raise SystemExit(f"unsafe path in bundle: {m.name}")
        t.extractall(root)
    return json.loads((root / "env.json").read_text())


def setup_comfy(root: Path, env: dict) -> Path:
    import comfy_setup
    comfy = root / "components/videos/comfyui"
    comfy_setup.setup_comfyui(comfy, None, env["comfy_commit"], env.get("gguf_commit"))
    extra = root / "extra/custom_nodes"
    if extra.exists():
        for d in extra.iterdir():
            shutil.copytree(d, comfy / "custom_nodes" / d.name, dirs_exist_ok=True)
    # vidpipe.comfy.start() launches COMFY_DIR/.venv/bin/python: point it at the host python (has torch)
    py = comfy / ".venv/bin/python"
    py.parent.mkdir(parents=True, exist_ok=True)
    if py.exists() or py.is_symlink():
        py.unlink()
    py.symlink_to(sys.executable)
    return comfy


def fetch_models(root: Path, env: dict, token) -> list:
    from huggingface_hub import hf_hub_download, snapshot_download
    models = root / "components/videos/models"
    got = []
    for m in env.get("models", []):
        dest = models / m["dest"]
        t0 = time.time()
        if not dest.exists():
            p = hf_hub_download(m["repo"], m["file"], token=token, local_dir=str(root / "_dl"))
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(p, dest)
        got.append({"name": m["name"], "bytes": dest.stat().st_size, "sec": round(time.time() - t0, 1)})
        print(f"model {m['name']}: {dest.stat().st_size} bytes", flush=True)
    for c in env.get("snapshots", []):  # e.g. the CLIP used by qc.identity
        d = models / c["dest"]
        if not d.exists():
            snapshot_download(c["repo"], local_dir=str(d), token=token,
                              allow_patterns=c.get("allow_patterns"))
    shutil.rmtree(root / "_dl", ignore_errors=True)
    return got


def pip_install(env: dict, constraints: Path) -> None:
    pkgs = env.get("pip", [])
    if not pkgs:
        return
    cmd = (["uv", "pip", "install", "-q", "--system"] if shutil.which("uv") else [sys.executable, "-m", "pip", "install", "-q"])
    subprocess.run(cmd + ["-c", str(constraints)] + pkgs, check=True)


OUT_REF = re.compile(r"\$\{out:(\d+)\}")


def _out_ref_steps(value) -> set:
    """Indices K of every ${out:K} reference in a (possibly nested) step params value."""
    if isinstance(value, str):
        return {int(k) for k in OUT_REF.findall(value)}
    if isinstance(value, list):
        return set().union(*(_out_ref_steps(v) for v in value), set())
    if isinstance(value, dict):
        return set().union(*(_out_ref_steps(v) for v in value.values()), set())
    return set()


def failed_deps(step_params, results: list) -> list:
    """Sorted indices of the steps that step_params reference and that failed or were skipped.

    results[k] must be the record of step k (done["steps"] keeps one record per step, in order).
    """
    failed = []
    for k in sorted(_out_ref_steps(step_params)):
        if k >= len(results):
            raise ValueError(f"step refers to ${{out:{k}}} but step {k} has not run yet")
        r = results[k]
        if r.get("skipped") or r["rc"] != 0:
            failed.append(k)
    return failed


def dep_failed(step_params, results: list) -> bool:
    return bool(failed_deps(step_params, results))


def skipped_record(i: int, step: dict, dep: int) -> dict:
    return {"i": i, "tool": step["tool"], "rc": None, "skipped": True, "reason": f"dependency step {dep} failed"}


def resolve(value, root: Path, outs: list):
    if isinstance(value, str):
        def sub(m):
            k = int(m.group(1))
            if outs[k] is None:
                raise ValueError(f"${{out:{k}}} points at a skipped step")
            return str(outs[k])
        value = OUT_REF.sub(sub, value)
        # Tools run with cwd <root>/tools, so a relative path to an existing file or dir under root must be made absolute.
        if value and not os.path.isabs(value) and (root / value).exists():
            return str((root / value).resolve())
        return value
    if isinstance(value, list):
        return [resolve(v, root, outs) for v in value]
    if isinstance(value, dict):
        return {k: resolve(v, root, outs) for k, v in value.items()}
    return value


def run_step(i: int, step: dict, root: Path, outs: list) -> dict:
    out_dir = root / "runs" / f"step{i}_{step['tool'].replace('.', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    params = resolve(step["params"], root, outs)
    pj = out_dir.parent / f"step{i}_params.json"
    pj.write_text(json.dumps(params))
    runner = step["runner"]
    argv = [a.replace("{params_json}", str(pj)).replace("{out_dir}", str(out_dir)).replace(LOCAL_HQ, str(root)) for a in runner["argv"]]
    if argv[0] in ("python", "python3"):
        argv[0] = sys.executable
    cwd = Path(runner["cwd"].replace(LOCAL_HQ, str(root)))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONPATH=str(root / "components/videos/src"))
    log = out_dir / "step.log"
    t0 = time.time()
    with open(log, "w") as lf:
        p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line); lf.write(line)
        rc = p.wait()
    outs.append(out_dir)
    return {"i": i, "tool": step["tool"], "rc": rc, "sec": round(time.time() - t0, 1), "out_dir": str(out_dir)}


def upload_dir(hub, dataset: str, local: Path, remote: str, msg: str) -> list:
    from hub import CommitOperation
    ops, files = [], []
    for f in sorted(local.rglob("*")):
        if f.is_file():
            rel = f.relative_to(local).as_posix()
            ops.append(CommitOperation(path_in_repo=f"{remote}/{rel}", path_or_fileobj=str(f)))
            files.append({"path": rel, "bytes": f.stat().st_size, "sha256": sha256(f)})
    if ops:
        hub.create_commit(dataset, ops, msg)
    return files


def beat(hub, dataset: str, base: str, stage: str, **extra) -> None:
    """Small progress file on the hub so a run is visible even when the platform shows no logs."""
    from hub import CommitOperation
    body = json.dumps({"stage": stage, "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}).encode()
    try:
        hub.create_commit(dataset, [CommitOperation(path_in_repo=f"{base}/PROGRESS.json", content_bytes=body)], f"progress {stage}")
    except Exception as e:  # visibility only; never fail the run for it
        print(f"progress beat failed: {type(e).__name__}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="path of the bundle in the job dataset")
    ap.add_argument("--job-dataset", required=True)
    ap.add_argument("--hub-dataset", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--keep-going", action="store_true", help="run later steps even if one failed")
    a = ap.parse_args(argv)
    token = os.environ.get("HF_TOKEN")
    from hub import HFHub, CommitOperation
    from huggingface_hub import hf_hub_download
    hub = HFHub(token)
    base = f"tool_runs/{a.run_id}"
    root = pick_root()
    done = {"run_id": a.run_id, "ok": False, "steps": [], "setup": {}}
    t0 = time.time()
    try:
        beat(hub, a.hub_dataset, base, "started")
        bundle = hf_hub_download(a.job_dataset, a.bundle, repo_type="dataset", token=token, force_download=True)
        env = extract(Path(bundle), root)
        import comfy_setup
        setup_comfy(root, env)
        done["setup"]["comfy_sec"] = round(time.time() - t0, 1)
        beat(hub, a.hub_dataset, base, "comfy ready", sec=done["setup"]["comfy_sec"])
        pip_install(env, root / "components/videos/comfyui/constraints.txt")
        done["setup"]["models"] = fetch_models(root, env, token)
        done["setup"]["ready_sec"] = round(time.time() - t0, 1)
        beat(hub, a.hub_dataset, base, "models ready", sec=done["setup"]["ready_sec"])
        outs: list = []
        for i, step in enumerate(env["steps"]):
            deps = failed_deps(step["params"], done["steps"])
            if deps:  # never run a step on a failed or skipped input; keep going with independent steps
                done["steps"].append(skipped_record(i, step, deps[0]))
                outs.append(None)  # keeps outs indexed by step
                print(f"step {i} {step['tool']} skipped: dependency step {deps[0]} failed", flush=True)
                continue
            beat(hub, a.hub_dataset, base, f"step {i} running", tool=step["tool"])
            res = run_step(i, step, root, outs)
            res["files"] = upload_dir(hub, a.hub_dataset, Path(res["out_dir"]), f"{base}/step{i}", f"tool_run {a.run_id} step{i}")
            res.pop("out_dir")
            done["steps"].append(res)
            if res["rc"] != 0 and not a.keep_going:
                break
        # a skipped step has rc None, so it also makes ok False
        done["ok"] = len(done["steps"]) == len(env["steps"]) and all(s["rc"] == 0 for s in done["steps"])
    except BaseException as e:  # record the cause on the hub, then fail
        done["error"] = f"{type(e).__name__}: {str(e)[:500]}"
        done["traceback"] = traceback.format_exc()[-3000:]
        traceback.print_exc()
    finally:
        done["total_sec"] = round(time.time() - t0, 1)
        hub.create_commit(a.hub_dataset, [CommitOperation(path_in_repo=f"{base}/DONE.json",
                          content_bytes=json.dumps(done, indent=1).encode())], f"tool_run {a.run_id} done ok={done['ok']}")
    return 0 if done["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
