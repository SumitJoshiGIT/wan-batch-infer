"""
Setup utilities for ComfyUI and custom nodes on remote GPU environments (Kaggle/Colab).
Handles pinned cloning, requirement filtering with host constraints,
and extra_model_paths.yaml generation from Kaggle dataset mirror mounts.
"""

import os
import sys
import re
import shutil
import subprocess
import importlib.metadata
from pathlib import Path
from typing import Dict, List, Optional, Set
import yaml

# Pinned commit constants (queried via git ls-remote of default branch HEAD)
# ComfyUI: default branch (master) HEAD as of 2026-11-23
COMFYUI_COMMIT = "1b7d0861e480310cc514c334a7c364b108892d5a"  # 2026-11-23T22:36:05Z
COMFYUI_REPO_URL = "https://github.com/Comfy-Org/ComfyUI.git"

# ComfyUI-GGUF: default branch (master) HEAD as of 2026-01-12
COMFYUI_GGUF_COMMIT = "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"  # 2026-01-12T12:55:28Z
COMFYUI_GGUF_REPO_URL = "https://github.com/city96/ComfyUI-GGUF.git"

# Wan nodes: WanFirstLastFrameToVideo and related Wan2.1/2.2 nodes are
# built directly into ComfyUI core in comfy_extras/nodes_wan.py.
# No third-party Wan node repo is needed.

# 6 mirror files provided by Kaggle dataset kagglekimg99k/wan14-model-cache
REQUIRED_MODEL_FILES: Dict[str, List[str]] = {
    "diffusion_models": [
        "Wan2.2-I2V-A14B-HighNoise-Q3_K_M.gguf",
        "Wan2.2-I2V-A14B-LowNoise-Q3_K_M.gguf",
    ],
    "text_encoders": [
        "umt5-xxl-encoder-Q3_K_M.gguf",
    ],
    "vae": [
        "wan_2.1_vae.safetensors",
    ],
    "loras": [
        "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
        "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
    ],
}

FORBIDDEN_PACKAGE_PREFIXES = (
    "torch",
    "torchvision",
    "torchaudio",
    "nvidia-",
)


def filter_requirements(lines: List[str]) -> List[str]:
    """Filter out lines for torch, torchvision, torchaudio, and nvidia-* packages.
    
    Preserves comments, blank lines, and non-torch dependencies.
    """
    filtered = []
    for line in lines:
        stripped = line.strip()
        # Keep comments and blank lines
        if not stripped or stripped.startswith("#"):
            filtered.append(line)
            continue
        
        # Parse package name (before version specifiers, extras, or environment markers)
        match = re.match(r"^([a-zA-Z0-9_\-\.]+)", stripped)
        if match:
            pkg_name = match.group(1).lower().replace("_", "-")
            # exact names only: a "torch" prefix would also drop torchsde, which ComfyUI needs
            if pkg_name in ("torch", "torchvision", "torchaudio") or pkg_name.startswith("nvidia-"):
                continue
        
        filtered.append(line)
    return filtered


def build_constraints(packages: Optional[List[str]] = None) -> str:
    """Generate constraints string (for -c constraints.txt) from host installed packages.
    
    Queries importlib.metadata for installed versions of torch, numpy, opencv, etc.
    """
    if packages is None:
        packages = ["torch", "torchvision", "torchaudio", "numpy", "opencv-python", "opencv-python-headless"]
    
    constraints = []
    for pkg in packages:
        try:
            ver = importlib.metadata.version(pkg)
            constraints.append(f"{pkg}=={ver}")
        except importlib.metadata.PackageNotFoundError:
            continue
    
    return "\n".join(constraints) + ("\n" if constraints else "")


def write_extra_model_paths(root: Path, mount: Path) -> Path:
    """Discover mirror files by walking the mount and write extra_model_paths.yaml.
    
    Points ComfyUI straight at /kaggle/input/... mirror folders without copying.
    Fails loudly (FileNotFoundError) if any required model file is missing.
    """
    mount = Path(mount)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    
    # Map each filename to its discovered parent directory
    file_to_dir: Dict[str, Path] = {}
    
    # Walk mount to locate files
    if mount.exists():
        for dirpath, _, filenames in os.walk(mount):
            dir_path = Path(dirpath)
            for fname in filenames:
                if fname not in file_to_dir:
                    file_to_dir[fname] = dir_path
    
    # Validate all required files are present
    category_dirs: Dict[str, Set[str]] = {}
    for category, fnames in REQUIRED_MODEL_FILES.items():
        category_dirs[category] = set()
        for fname in fnames:
            if fname not in file_to_dir:
                raise FileNotFoundError(
                    f"Required model file '{fname}' not found under mount path '{mount}'. "
                    f"Ensure dataset mirror kagglekimg99k/wan14-model-cache is mounted."
                )
            category_dirs[category].add(str(file_to_dir[fname].resolve()))
    
    # Construct ComfyUI extra_model_paths configuration
    config = {
        "kaggle_mirror": {
            category: "\n".join(sorted(dirs))
            for category, dirs in category_dirs.items()
        }
    }
    
    out_path = root / "extra_model_paths.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    
    return out_path


def clone_pinned_repo(repo_url: str, commit: str, dest_dir: Path) -> None:
    """Clone a repository shallowly and checkout a pinned commit."""
    dest_dir = Path(dest_dir)
    if (dest_dir / ".git").exists():
        subprocess.run(["git", "-C", str(dest_dir), "fetch", "--depth", "1", "origin", commit], check=True)
        subprocess.run(["git", "-C", str(dest_dir), "checkout", "-q", commit], check=True)
        return
    
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", repo_url, str(dest_dir)], check=True)
    subprocess.run(["git", "-C", str(dest_dir), "checkout", "-q", commit], check=True)


def setup_comfyui(comfy_dir: Path, mount_dir: Path) -> None:
    """Full setup sequence for ComfyUI on remote host."""
    comfy_dir = Path(comfy_dir)
    mount_dir = Path(mount_dir)
    
    # 1. Clone ComfyUI at pinned commit
    clone_pinned_repo(COMFYUI_REPO_URL, COMFYUI_COMMIT, comfy_dir)
    
    # 2. Clone ComfyUI-GGUF at pinned commit
    gguf_dir = comfy_dir / "custom_nodes" / "ComfyUI-GGUF"
    clone_pinned_repo(COMFYUI_GGUF_REPO_URL, COMFYUI_GGUF_COMMIT, gguf_dir)
    
    # 3. Create host constraints
    constraints_content = build_constraints()
    constraints_path = comfy_dir / "constraints.txt"
    constraints_path.write_text(constraints_content, encoding="utf-8")
    
    # 4. Install ComfyUI and node requirements with torch/nvidia filtered out
    req_files = [
        comfy_dir / "requirements.txt",
        gguf_dir / "requirements.txt",
    ]
    
    has_uv = shutil.which("uv") is not None
    for req_file in req_files:
        if not req_file.exists():
            continue
        original_lines = req_file.read_text(encoding="utf-8").splitlines(keepends=True)
        filtered_lines = filter_requirements(original_lines)
        temp_req = req_file.with_name("req_filtered.txt")
        temp_req.write_text("".join(filtered_lines), encoding="utf-8")
        
        install_cmd = [
            "uv", "pip", "install", "-q", "--no-cache-dir", "--system",
            "-c", str(constraints_path),
            "-r", str(temp_req),
        ] if has_uv else [
            sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
            "-c", str(constraints_path),
            "-r", str(temp_req),
        ]
        subprocess.run(install_cmd, check=True)
        temp_req.unlink(missing_ok=True)
    
    # 5. Write extra_model_paths.yaml pointing at mount
    write_extra_model_paths(comfy_dir, mount_dir)
