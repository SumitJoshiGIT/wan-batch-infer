#!/usr/bin/env python3
"""
Thin launcher for Wan 2.2 I2V batch inference on Kaggle/Colab.
Stdlib only - no external dependencies.
"""

import os
import sys
import subprocess
import json
import base64
import traceback
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any

# Sentinel - staging replaces this with a serialized dict
EMBEDDED_RUN_CONFIG = None

# Secret names we handle (env -> kaggle_secrets -> .env)
SECRET_NAMES = (
    "GITHUB_TOKEN",   # read-only PAT for cloning
    "HF_TOKEN",       # Hugging Face token
)

def detect_platform() -> str:
    """Detect if running on Kaggle, Colab, or local."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if os.path.exists("/content"):
        return "colab"
    return "local"

def get_repo_dir(platform: str) -> Path:
    """Get the repository directory for the platform."""
    if platform == "kaggle":
        return Path("/kaggle/temp/wan-batch-infer")
    elif platform == "colab":
        return Path("/content/wan-batch-infer")
    else:
        return Path.cwd()

def load_secrets() -> Dict[str, str]:
    """Load secrets from env, kaggle_secrets, or .env file."""
    secrets = {}
    
    # 1. Environment variables (highest priority)
    for name in SECRET_NAMES:
        if name in os.environ:
            secrets[name] = os.environ[name]
    
    # 2. Kaggle secrets (if on Kaggle)
    if detect_platform() == "kaggle":
        try:
            import importlib
            ks = importlib.import_module("kaggle_secrets")
            client = ks.UserSecretsClient()
            for name in SECRET_NAMES:
                if name not in secrets:
                    try:
                        secrets[name] = client.get_secret(name)
                    except Exception as e:
                        print(f"secret {name}: {type(e).__name__}: {str(e)[:120]}")
        except Exception as e:
            print(f"kaggle_secrets unavailable: {type(e).__name__}: {str(e)[:120]}")
    
    # 3. .env file (gitignored)
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                if key in SECRET_NAMES and key not in secrets:
                    secrets[key] = val
    
    # Export to os.environ for child processes
    for k, v in secrets.items():
        os.environ[k] = v
    
    return secrets

def sanitize_error(msg: str, secret_token: Optional[str] = None) -> str:
    """Remove tokens, headers, and base64 from error messages."""
    import re
    if secret_token:
        msg = msg.replace(secret_token, '***')
        b64 = base64.b64encode(f"x-access-token:{secret_token}".encode()).decode()
        msg = msg.replace(b64, '***')
    # Remove Authorization header patterns
    msg = re.sub(r'Authorization:\s*Basic\s+\S+', 'Authorization: Basic ***', msg)
    msg = re.sub(r'Authorization:\s*Bearer\s+\S+', 'Authorization: Bearer ***', msg)
    # Remove x-access-token:TOKEN patterns
    msg = re.sub(r'x-access-token:[^\s]+', 'x-access-token:***', msg)
    # Remove GitHub token patterns (ghp_, gho_, ghu_, ghs_, ghr_, github_pat_)
    msg = re.sub(r'(gh[pousr]|github_pat)_[A-Za-z0-9_]+', '***', msg)
    # Remove HF token patterns
    msg = re.sub(r'hf_[A-Za-z0-9_]+', 'hf_***', msg)
    # Padded base64 blobs only: unpadded 16+ char strings are usually SHAs or paths
    msg = re.sub(r'[A-Za-z0-9+/]{16,}={1,2}', '***', msg)
    return msg

def clone_repo(repo_url: str, branch: str, target_dir: Path, github_token: Optional[str]) -> None:
    """Idempotent shallow clone with auth."""
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"  # fail fast instead of waiting for credentials
    auth_args: List[str] = []
    if github_token:
        auth_header = base64.b64encode(f"x-access-token:{github_token}".encode()).decode()
        auth_args = ["-c", f"http.extraHeader=Authorization: Basic {auth_header}"]

    if target_dir.exists():
        # Fetch + reset
        try:
            subprocess.run(
                ["git", *auth_args, "-C", str(target_dir), "fetch", "--depth", "1", "origin", branch],
                check=True, capture_output=True, text=True, env=env
            )
            subprocess.run(
                ["git", "-C", str(target_dir), "reset", "--hard", f"origin/{branch}"],
                check=True, capture_output=True, text=True, env=env
            )
            return
        except subprocess.CalledProcessError as e:
            # Fall through to fresh clone
            shutil.rmtree(target_dir, ignore_errors=True)
    
    # Fresh clone with auth header if token provided
    git_cmd = ["git", *auth_args, "clone", "--depth", "1", "--branch", branch, repo_url, str(target_dir)]
    
    try:
        subprocess.run(git_cmd, check=True, capture_output=True, text=True, env=env)
    except subprocess.CalledProcessError as e:
        err = sanitize_error(e.stderr or e.stdout or str(e), github_token)
        print(f"Clone failed: {err}", file=sys.stderr)
        sys.exit(1)

def install_requirements(req_path: Path) -> None:
    """Install requirements-cloud.txt with uv or pip, never reinstall torch."""
    if not req_path.exists():
        return
    
    # Check for uv
    has_uv = shutil.which("uv") is not None
    
    if has_uv:
        # Use uv, but exclude torch packages
        try:
            subprocess.run(
                ["uv", "pip", "install", "--system", "-q", "--no-cache-dir", "-r", str(req_path)],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"uv install failed: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Fallback to pip
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "-r", str(req_path)],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"pip install failed: {e}", file=sys.stderr)
            sys.exit(1)

def get_run_config() -> Dict[str, Any]:
    """Get run configuration from EMBEDDED_RUN_CONFIG or local sources."""
    platform = detect_platform()
    
    if EMBEDDED_RUN_CONFIG is not None:
        return EMBEDDED_RUN_CONFIG
    
    if platform == "colab":
        # Read args after '--' in argv
        if "--" in sys.argv:
            idx = sys.argv.index("--")
            args = sys.argv[idx + 1:]
            return {"repo": "", "branch": "main", "args": args}
        
        # Or local run_config.json
        config_path = Path("/content/run_config.json")
        if config_path.exists():
            return json.loads(config_path.read_text())
    
    if platform == "local":
        config_path = Path("run_config.json")
        if config_path.exists():
            return json.loads(config_path.read_text())
    
    raise SystemExit("No run configuration found. Set EMBEDDED_RUN_CONFIG or provide run_config.json / -- args")

def setup_logging(output_dir: Path) -> tuple:
    """Set up tee logging to run.log."""
    log_path = output_dir / "run.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open log file
    log_file = open(log_path, "w", buffering=1)  # line buffered
    
    # Create tee streams
    class TeeStream:
        def __init__(self, *streams):
            self.streams = streams
        
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        
        def flush(self):
            for s in self.streams:
                s.flush()
        
        def fileno(self):
            return self.streams[0].fileno()
    
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    
    return log_file, log_path

def main():
    platform = detect_platform()
    repo_dir = get_repo_dir(platform)
    
    # Set up logging early
    if platform == "kaggle":
        output_dir = Path("/kaggle/working/outputs")
    else:
        output_dir = repo_dir / "outputs"
    
    orig_out, orig_err = sys.stdout, sys.stderr
    log_file, log_path = setup_logging(output_dir)
    
    try:
        print(f"=== Wan Batch Inference Launcher ===")
        print(f"Platform: {platform}")
        print(f"Repo dir: {repo_dir}")
        print(f"Output dir: {output_dir}")
        print(f"Log: {log_path}")
        
        # Load secrets
        secrets = load_secrets()
        print(f"Loaded secrets: {list(secrets.keys())}")
        
        # Get run config
        config = get_run_config()
        repo_url = config.get("repo")
        branch = config.get("branch", "main")
        infer_args = config.get("args", [])
        
        if not repo_url:
            raise SystemExit("Run config missing 'repo' field")
        
        print(f"Repo: {repo_url}")
        print(f"Branch: {branch}")
        print(f"Infer args: {infer_args}")
        
        # Clone repo
        github_token = secrets.get("GITHUB_TOKEN")
        clone_repo(repo_url, branch, repo_dir, github_token)
        print(f"Cloned to {repo_dir}")
        
        # Install requirements
        req_path = repo_dir / "requirements-cloud.txt"
        if req_path.exists():
            install_requirements(req_path)
            print("Requirements installed")
        else:
            print("No requirements-cloud.txt found, skipping")
        
        # Run infer.py
        infer_script = repo_dir / "src" / "infer.py"
        if not infer_script.exists():
            raise SystemExit(f"infer.py not found at {infer_script}")
        
        cmd = [sys.executable, str(infer_script)] + infer_args
        print(f"Running: {' '.join(cmd)}")
        print(f"CWD: {repo_dir}")
        
        # Stream through our tee so infer.py output lands in run.log too
        proc = subprocess.Popen(cmd, cwd=repo_dir, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)
        print(f"Inference completed with exit code {rc}")
        
    except SystemExit as e:
        print(f"SystemExit: {e}", file=sys.stderr)
        raise
    except Exception as e:
        traceback.print_exc()
        raise
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err
        log_file.close()

if __name__ == "__main__":
    main()