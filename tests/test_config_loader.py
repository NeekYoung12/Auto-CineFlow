"""Tests for config discovery and parsing helpers."""

import shutil
import tempfile
from pathlib import Path

from autocineflow.config_loader import (
    discover_default_config_path,
    normalize_openai_base_url,
    parse_key_value_config,
    resolve_openai_settings,
)


def _workspace_temp_dir() -> Path:
    """Create a writable temp directory inside the repository workspace."""

    root = Path.cwd() / ".pytest_tmp"
    root.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(dir=root))


def test_parse_key_value_config_ignores_section_labels():
    temp_dir = _workspace_temp_dir()
    try:
        config_path = temp_dir / "conf"
        config_path.write_text(
            "\n".join(
                [
                    "MiniMax:",
                    "OPENAI_BASE_URL=https://example.invalid/v1",
                    "OPENAI_API_KEY=sk-test",
                ]
            ),
            encoding="utf-8",
        )

        values = parse_key_value_config(config_path)
        assert values["OPENAI_BASE_URL"] == "https://example.invalid/v1"
        assert values["OPENAI_API_KEY"] == "sk-test"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_discover_default_config_path_finds_workspace_sibling(monkeypatch):
    temp_dir = _workspace_temp_dir()
    try:
        repo_root = temp_dir / "Auto-CineFlow"
        repo_root.mkdir()
        config_dir = temp_dir / "config"
        config_dir.mkdir()
        config_path = config_dir / "conf"
        config_path.write_text("OPENAI_API_KEY=sk-test", encoding="utf-8")

        monkeypatch.chdir(repo_root)

        assert discover_default_config_path() == config_path
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_resolve_openai_settings_prefers_explicit_values():
    temp_dir = _workspace_temp_dir()
    try:
        config_path = temp_dir / "conf"
        config_path.write_text(
            "OPENAI_BASE_URL=https://example.invalid/v1\nOPENAI_API_KEY=sk-file",
            encoding="utf-8",
        )

        api_key, base_url = resolve_openai_settings(
            api_key="sk-explicit",
            base_url="https://override.invalid/v1",
            config_path=config_path,
        )

        assert api_key == "sk-explicit"
        assert base_url == "https://override.invalid/v1"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_normalize_openai_base_url_appends_v1():
    assert normalize_openai_base_url("https://example.invalid") == "https://example.invalid/v1"
    assert normalize_openai_base_url("https://example.invalid/custom") == "https://example.invalid/custom/v1"
    assert normalize_openai_base_url("https://example.invalid/v1") == "https://example.invalid/v1"
