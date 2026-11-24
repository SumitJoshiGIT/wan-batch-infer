"""Tests for infra/launcher.py - stdlib only, audits imports."""

import sys
import ast
import importlib.util
import tempfile
import os
import subprocess
from pathlib import Path


def test_launcher_imports_stdlib_only():
    """Verify launcher.py only imports stdlib modules."""
    launcher_path = Path(__file__).parent.parent / "infra" / "launcher.py"
    source = launcher_path.read_text()
    tree = ast.parse(source)
    
    stdlib_modules = {
        'os', 'sys', 'subprocess', 'json', 'base64', 'traceback',
        'shutil', 'pathlib', 'typing', 're', 'tempfile', 'importlib', 'urllib'
    }
    
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    
    non_stdlib = imports - stdlib_modules
    assert not non_stdlib, f"Non-stdlib imports found: {non_stdlib}"


def test_sanitize_error_removes_tokens():
    """Test that sanitize_error removes tokens, headers, and base64."""
    launcher_path = Path(__file__).parent.parent / "infra" / "launcher.py"
    spec = importlib.util.spec_from_file_location("launcher", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    
    # Test Authorization header removal
    msg = 'Authorization: Basic dXNlcjpwYXNz'
    sanitized = launcher.sanitize_error(msg)
    assert 'Authorization: Basic' not in sanitized or '***' in sanitized
    
    # Test Bearer token removal
    msg = 'Authorization: Bearer ghp_abcdef123456'
    sanitized = launcher.sanitize_error(msg)
    assert 'Bearer' not in sanitized or '***' in sanitized
    
    # Test x-access-token removal
    msg = 'git -c http.extraHeader=Authorization: Basic x-access-token:ghp_xxx'
    sanitized = launcher.sanitize_error(msg)
    assert 'x-access-token:ghp_xxx' not in sanitized
    
    # Test base64-like strings
    msg = 'token: dXNlcm5hbWU6cGFzc3dvcmQ='
    sanitized = launcher.sanitize_error(msg)
    assert 'dXNlcm5hbWU6cGFzc3dvcmQ=' not in sanitized


def test_detect_platform():
    """Test platform detection logic."""
    launcher_path = Path(__file__).parent.parent / "infra" / "launcher.py"
    spec = importlib.util.spec_from_file_location("launcher", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    
    # Mock Kaggle env
    os.environ["KAGGLE_KERNEL_RUN_TYPE"] = "interactive"
    assert launcher.detect_platform() == "kaggle"
    del os.environ["KAGGLE_KERNEL_RUN_TYPE"]
    
    # Mock Colab path by patching os.path.exists
    original_exists = os.path.exists
    try:
        os.path.exists = lambda p: p == "/content" or original_exists(p)
        assert launcher.detect_platform() == "colab"
    finally:
        os.path.exists = original_exists
    
    # Default to local
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        del os.environ["KAGGLE_KERNEL_RUN_TYPE"]
    assert launcher.detect_platform() == "local"


def test_get_repo_dir():
    """Test repository directory resolution."""
    launcher_path = Path(__file__).parent.parent / "infra" / "launcher.py"
    spec = importlib.util.spec_from_file_location("launcher", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    
    assert launcher.get_repo_dir("kaggle") == Path("/kaggle/temp/wan-batch-infer")
    assert launcher.get_repo_dir("colab") == Path("/content/wan-batch-infer")
    assert launcher.get_repo_dir("local") == Path.cwd()


def test_embeded_run_config_sentinel():
    """Test EMBEDDED_RUN_CONFIG sentinel exists and is None initially."""
    launcher_path = Path(__file__).parent.parent / "infra" / "launcher.py"
    source = launcher_path.read_text()
    
    # Check exact sentinel line
    assert 'EMBEDDED_RUN_CONFIG = None' in source
    
    # Verify it's the sentinel value
    spec = importlib.util.spec_from_file_location("launcher", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    assert launcher.EMBEDDED_RUN_CONFIG is None


def test_secret_names_constant():
    """Test SECRET_NAMES contains expected secrets."""
    launcher_path = Path(__file__).parent.parent / "infra" / "launcher.py"
    spec = importlib.util.spec_from_file_location("launcher", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    
    assert "GITHUB_TOKEN" in launcher.SECRET_NAMES
    assert "HF_TOKEN" in launcher.SECRET_NAMES


def test_forced_clone_failure_no_token_in_output():
    """Test that clone failure doesn't leak tokens/headers."""
    # This test verifies the sanitize_error function handles various
    # token leak scenarios. The actual clone would require network.
    launcher_path = Path(__file__).parent.parent / "infra" / "launcher.py"
    spec = importlib.util.spec_from_file_location("launcher", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    
    # Simulate error with token in stderr
    test_errors = [
        "fatal: could not read Username for 'https://github.com': terminal prompts disabled\nAuthorization: Basic dXNlcjpwYXNz",
        "remote: Invalid token: ghp_abcdef1234567890",
        "error: x-access-token:ghp_xxx failed",
    ]
    
    for err in test_errors:
        sanitized = launcher.sanitize_error(err)
        # No raw tokens should appear
        assert "dXNlcjpwYXNz" not in sanitized
        assert "ghp_abcdef" not in sanitized
        assert "x-access-token:ghp" not in sanitized
        # Should contain sanitization marker
        assert "***" in sanitized


def test_clone_repo_forced_failure_sanitizes_stderr(capsys, monkeypatch):
    """Forced clone failure test proves no token/header in output."""
    import pytest
    launcher_path = Path(__file__).parent.parent / "infra" / "launcher.py"
    spec = importlib.util.spec_from_file_location("launcher", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    
    secret_token = "ghp_VERY_SECRET_TOKEN_DO_NOT_LEAK_12345"
    b64_token = launcher.base64.b64encode(f"x-access-token:{secret_token}".encode()).decode()
    
    # Run clone_repo against an invalid URL with secret_token
    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / "test_repo"
        with pytest.raises(SystemExit) as excinfo:
            launcher.clone_repo(
                repo_url="https://localhost:99999/nonexistent/repo.git",
                branch="main",
                target_dir=dest,
                github_token=secret_token
            )
        assert excinfo.value.code != 0
    
    captured = capsys.readouterr()
    err_output = captured.err
    
    # Verify neither the raw token, nor header, nor base64 is in stderr
    assert secret_token not in err_output
    assert b64_token not in err_output
    assert "Authorization: Basic" not in err_output or "Authorization: Basic ***" in err_output
    assert "Clone failed:" in err_output


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))


def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        "launcher_under_test", Path(__file__).parent.parent / "infra" / "launcher.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_clone_command_has_single_git_and_auth_before_clone(tmp_path, monkeypatch):
    mod = _load_launcher()
    seen = []

    def fake_run(cmd, **kw):
        seen.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    mod.clone_repo("https://example.com/r.git", "master", tmp_path / "r", "tok123")
    cmd, kw = seen[0]
    assert cmd.count("git") == 1 and cmd[0] == "git"
    assert cmd.index("-c") < cmd.index("clone")
    assert kw["env"]["GIT_TERMINAL_PROMPT"] == "0"


def test_clone_without_token_has_no_auth_header(tmp_path, monkeypatch):
    mod = _load_launcher()
    seen = []
    monkeypatch.setattr(mod.subprocess, "run",
                        lambda cmd, **kw: seen.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))
    mod.clone_repo("https://example.com/r.git", "master", tmp_path / "r", None)
    assert seen[0][:2] == ["git", "clone"]


def test_fetch_hub_job_reads_spec_and_validates(monkeypatch):
    import io, json, urllib.request
    spec = importlib.util.spec_from_file_location("launcher", Path(__file__).parent.parent / "infra" / "launcher.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    seen = {}

    class R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=0):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        return R(json.dumps({"name": "j", "repo": "r", "args": ["--x"]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    job = launcher.fetch_hub_job({"dataset": "o/d", "path": "jobs/next.json"}, "tok")
    assert job["args"] == ["--x"]
    assert seen["url"] == "https://huggingface.co/datasets/o/d/resolve/main/jobs/next.json"
    assert seen["auth"] == "Bearer tok"

    import pytest
    with pytest.raises(SystemExit):
        launcher.fetch_hub_job({"dataset": "o/d"}, None)
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=0: R(b'{"name": "j"}'))
    with pytest.raises(SystemExit):
        launcher.fetch_hub_job({"dataset": "o/d"}, "tok")
