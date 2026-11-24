"""Tests for infra/kernel-metadata.json - validates structure and slug."""

import json
import re
from pathlib import Path


def test_kernel_metadata_structure():
    """Test kernel-metadata.json has required fields with correct types."""
    metadata_path = Path(__file__).parent.parent / "infra" / "kernel-metadata.json"
    data = json.loads(metadata_path.read_text())
    
    # Required fields
    assert "id" in data
    assert "title" in data
    assert "code_file" in data
    assert "language" in data
    assert "kernel_type" in data
    assert "is_private" in data
    assert "enable_gpu" in data
    assert "enable_internet" in data
    assert "dataset_sources" in data
    assert "competition_sources" in data
    assert "kernel_sources" in data
    
    # Type checks
    assert isinstance(data["id"], str)
    assert isinstance(data["title"], str)
    assert isinstance(data["code_file"], str)
    assert isinstance(data["language"], str)
    assert isinstance(data["kernel_type"], str)
    assert isinstance(data["is_private"], bool)
    assert isinstance(data["enable_gpu"], bool)
    assert isinstance(data["enable_internet"], bool)
    assert isinstance(data["dataset_sources"], list)
    assert isinstance(data["competition_sources"], list)
    assert isinstance(data["kernel_sources"], list)


def test_kernel_metadata_values():
    """Test specific expected values."""
    metadata_path = Path(__file__).parent.parent / "infra" / "kernel-metadata.json"
    data = json.loads(metadata_path.read_text())
    
    assert data["code_file"] == "launcher.py"
    assert data["language"] == "python"
    assert data["kernel_type"] == "script"
    assert data["is_private"] is True
    assert data["enable_gpu"] is True
    assert data["enable_internet"] is True
    assert data["dataset_sources"] == ["kagglekimg99k/wan14-model-cache"]
    assert data["competition_sources"] == []
    assert data["kernel_sources"] == []


def test_id_slugifies_title():
    """Test that id slugifies from title (lowercase, non-alnum -> '-')."""
    metadata_path = Path(__file__).parent.parent / "infra" / "kernel-metadata.json"
    data = json.loads(metadata_path.read_text())
    
    title = data["title"]
    expected_slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
    # The id should be owner/slug
    id_parts = data["id"].split('/')
    assert len(id_parts) == 2
    owner, slug = id_parts
    assert slug == expected_slug
    # Owner should be the Kaggle username
    assert owner == "kagglekimg99k"


def test_no_push_trigger_in_workflow():
    """Test that .github/workflows/kaggle-infer.yml has workflow_dispatch only."""
    workflow_path = Path(__file__).parent.parent / ".github" / "workflows" / "kaggle-infer.yml"
    content = workflow_path.read_text()
    
    # Parse YAML to check triggers
    import yaml
    workflow = yaml.safe_load(content)
    
    assert "on" in workflow
    triggers = workflow["on"]
    # Should only have workflow_dispatch (no push, schedule, etc.)
    assert list(triggers.keys()) == ["workflow_dispatch"]


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main([__file__, "-v"]))