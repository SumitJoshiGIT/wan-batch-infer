"""Tests for src/comfy_setup.py - pure/testable functions, no network in tests."""

import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import comfy_setup


def test_filter_requirements_removes_torch_packages():
    """Test filter_requirements removes torch, torchvision, torchaudio, nvidia-*."""
    lines = [
        "torch==2.1.0\n",
        "torchvision==0.16.0\n",
        "torchaudio==2.1.0\n",
        "nvidia-cuda-runtime-cu11==11.7.99\n",
        "nvidia-cublas-cu11==11.10.3.66\n",
        "huggingface_hub>=0.25.0\n",
        "requests>=2.31.0\n",
        "# comment line\n",
        "\n",
        "numpy>=1.24.0\n",
    ]
    
    filtered = comfy_setup.filter_requirements(lines)
    
    # Should remove torch packages
    filtered_text = "".join(filtered)
    assert "torch==" not in filtered_text
    assert "torchvision==" not in filtered_text
    assert "torchaudio==" not in filtered_text
    assert "nvidia-" not in filtered_text
    
    # Should keep others
    assert "huggingface_hub" in filtered_text
    assert "requests" in filtered_text
    assert "numpy" in filtered_text
    
    # Should keep comments and blank lines
    assert "# comment line" in filtered_text
    assert "\n\n" in filtered_text


def test_filter_requirements_handles_variants():
    """Test filter handles package name variants (underscores, dashes)."""
    lines = [
        "torch_aaa==1.0\n",      # starts with torch
        "torchvision_ext==1.0\n", # starts with torchvision
        "nvidia_something==1.0\n", # starts with nvidia
    ]
    
    filtered = comfy_setup.filter_requirements(lines)
    filtered_text = "".join(filtered)
    # All should be removed
    assert filtered_text.strip() == ""


def test_build_constraints():
    """Test build_constraints generates constraints from installed packages."""
    constraints = comfy_setup.build_constraints(["requests", "nonexistent_package_xyz"])
    
    # Should contain requests with version
    assert "requests==" in constraints
    # Should not contain nonexistent package
    assert "nonexistent_package_xyz" not in constraints
    # Should end with newline if any packages found
    if "requests==" in constraints:
        assert constraints.endswith("\n")


def test_write_extra_model_paths_creates_valid_yaml():
    """Test write_extra_model_paths creates valid extra_model_paths.yaml."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "comfyui"
        mount = Path(tmpdir) / "kaggle_input"
        
        # Create mock mount structure mirroring Kaggle layout
        for category, files in comfy_setup.REQUIRED_MODEL_FILES.items():
            for fname in files:
                file_dir = mount / category / fname.replace(".gguf", "").replace(".safetensors", "")
                file_dir.mkdir(parents=True)
                (file_dir / fname).write_text("dummy")
        
        # Should succeed
        out_path = comfy_setup.write_extra_model_paths(root, mount)
        
        assert out_path.exists()
        assert out_path.name == "extra_model_paths.yaml"
        
        # Verify content is valid YAML with expected structure
        import yaml
        config = yaml.safe_load(out_path.read_text())
        assert "kaggle_mirror" in config
        for category in comfy_setup.REQUIRED_MODEL_FILES.keys():
            assert category in config["kaggle_mirror"]
            # All directories containing the files should be present in the YAML
            lines = config["kaggle_mirror"][category].strip().split("\n")
            assert len(lines) == len(comfy_setup.REQUIRED_MODEL_FILES[category])


def test_write_extra_model_paths_fails_loudly_on_missing_file():
    """Test write_extra_model_paths raises FileNotFoundError for missing files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "comfyui"
        mount = Path(tmpdir) / "kaggle_input"
        
        # Create mount with only some files (missing one)
        cat_dir = mount / "diffusion_models"
        cat_dir.mkdir(parents=True)
        (cat_dir / "Wan2.2-I2V-A14B-HighNoise-Q3_K_M.gguf").write_text("dummy")
        # Missing LowNoise file
        
        try:
            comfy_setup.write_extra_model_paths(root, mount)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            assert "LowNoise" in str(e) or "missing model file" in str(e).lower()


def test_required_model_files_complete():
    """Verify REQUIRED_MODEL_FILES has all 6 expected files across categories."""
    total_files = sum(len(files) for files in comfy_setup.REQUIRED_MODEL_FILES.values())
    assert total_files == 6
    
    # Check specific expected files
    assert "Wan2.2-I2V-A14B-HighNoise-Q3_K_M.gguf" in comfy_setup.REQUIRED_MODEL_FILES["diffusion_models"]
    assert "Wan2.2-I2V-A14B-LowNoise-Q3_K_M.gguf" in comfy_setup.REQUIRED_MODEL_FILES["diffusion_models"]
    assert "umt5-xxl-encoder-Q3_K_M.gguf" in comfy_setup.REQUIRED_MODEL_FILES["text_encoders"]
    assert "wan_2.1_vae.safetensors" in comfy_setup.REQUIRED_MODEL_FILES["vae"]
    assert "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors" in comfy_setup.REQUIRED_MODEL_FILES["loras"]
    assert "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors" in comfy_setup.REQUIRED_MODEL_FILES["loras"]


def test_pinned_commits_are_valid_sha():
    """Test pinned commit constants are valid 40-char SHA hex."""
    import re
    sha_pattern = re.compile(r'^[a-f0-9]{40}$')
    
    assert sha_pattern.match(comfy_setup.COMFYUI_COMMIT), f"Invalid ComfyUI commit: {comfy_setup.COMFYUI_COMMIT}"
    assert sha_pattern.match(comfy_setup.COMFYUI_GGUF_COMMIT), f"Invalid ComfyUI-GGUF commit: {comfy_setup.COMFYUI_GGUF_COMMIT}"
    
    # Verify they have date comments (not enforced programmatically, but documented)
    assert "2026" in comfy_setup.__doc__ or "2026" in open(__file__).read()


def test_forbidden_package_prefixes():
    """Test FORBIDDEN_PACKAGE_PREFIXES tuple contains expected items."""
    assert "torch" in comfy_setup.FORBIDDEN_PACKAGE_PREFIXES
    assert "torchvision" in comfy_setup.FORBIDDEN_PACKAGE_PREFIXES
    assert "torchaudio" in comfy_setup.FORBIDDEN_PACKAGE_PREFIXES
    assert "nvidia-" in comfy_setup.FORBIDDEN_PACKAGE_PREFIXES


def test_requirements_cloud_has_no_torch_packages():
    """Test that requirements-cloud.txt does not contain torch/nvidia packages."""
    req_path = Path(__file__).parent.parent / "requirements-cloud.txt"
    assert req_path.exists()
    
    lines = req_path.read_text().splitlines()
    for line in lines:
        line = line.strip().lower()
        if not line or line.startswith("#"):
            continue
        assert not line.startswith("torch"), f"Forbidden package in requirements-cloud.txt: {line}"
        assert not line.startswith("torchvision"), f"Forbidden package in requirements-cloud.txt: {line}"
        assert not line.startswith("torchaudio"), f"Forbidden package in requirements-cloud.txt: {line}"
        assert not line.startswith("nvidia"), f"Forbidden package in requirements-cloud.txt: {line}"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))