"""Tests for config discovery and parsing helpers."""

import shutil
import tempfile
from pathlib import Path

from autocineflow.config_loader import (
    discover_default_config_path,
    normalize_openai_base_url,
    parse_key_value_config,
    parse_sectioned_config,
    resolve_runninghub_api_format_dir,
    resolve_minimax_media_settings,
    resolve_openai_settings,
    resolve_runninghub_settings,
    resolve_runninghub_workflow_ids,
    resolve_volcengine_ark_settings,
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


def test_parse_sectioned_config_handles_duplicate_keys_by_section():
    temp_dir = _workspace_temp_dir()
    try:
        config_path = temp_dir / "conf"
        config_path.write_text(
            "\n".join(
                [
                    "MiniMax:",
                    "OPENAI_API_KEY=sk-openai",
                    "Image or Video Generation:",
                    "API_KEY=sk-media",
                ]
            ),
            encoding="utf-8",
        )

        sections = parse_sectioned_config(config_path)
        assert sections["MiniMax"]["OPENAI_API_KEY"] == "sk-openai"
        assert sections["Image or Video Generation"]["API_KEY"] == "sk-media"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_parse_sectioned_config_supports_bare_section_headers():
    temp_dir = _workspace_temp_dir()
    try:
        config_path = temp_dir / "conf"
        config_path.write_text(
            "\n".join(
                [
                    "Image or Video Generation:",
                    "API_KEY=sk-media",
                    "Kimi-Code",
                    "API_KEY=sk-kimi",
                ]
            ),
            encoding="utf-8",
        )

        sections = parse_sectioned_config(config_path)
        assert sections["Image or Video Generation"]["API_KEY"] == "sk-media"
        assert sections["Kimi-Code"]["API_KEY"] == "sk-kimi"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_parse_sectioned_config_supports_colon_section_headers():
    temp_dir = _workspace_temp_dir()
    try:
        config_path = temp_dir / "conf"
        config_path.write_text(
            "\n".join(
                [
                    "Kimi-Code:",
                    "API_KEY=sk-kimi",
                    "RunningHUB:",
                    "API_KEY=rh-key",
                ]
            ),
            encoding="utf-8",
        )

        sections = parse_sectioned_config(config_path)
        assert sections["Kimi-Code"]["API_KEY"] == "sk-kimi"
        assert sections["RunningHUB"]["API_KEY"] == "rh-key"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_resolve_minimax_media_settings_reads_media_section():
    temp_dir = _workspace_temp_dir()
    try:
        config_path = temp_dir / "conf"
        config_path.write_text(
            "\n".join(
                [
                    "Image or Video Generation:",
                    "API_KEY=sk-media",
                    "MINIMAX_BASE_URL=https://api.example.invalid/v1",
                ]
            ),
            encoding="utf-8",
        )

        api_key, base_url = resolve_minimax_media_settings(config_path)
        assert api_key == "sk-media"
        assert base_url == "https://api.example.invalid/v1"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_resolve_runninghub_and_volcengine_settings_read_sections():
    temp_dir = _workspace_temp_dir()
    try:
        config_path = temp_dir / "conf"
        config_path.write_text(
            "\n".join(
                [
                    "RunningHUB:",
                    "API_KEY=rh-key",
                    "RUNNINGHUB_BASE_URL=https://rh.example.invalid",
                    "volcengine:",
                    "ARK_API_KEY=ark-key",
                    "VOLCENGINE_ARK_BASE_URL=https://las.example.invalid",
                ]
            ),
            encoding="utf-8",
        )

        runninghub_api_key, runninghub_base_url = resolve_runninghub_settings(config_path)
        volcengine_api_key, volcengine_base_url = resolve_volcengine_ark_settings(config_path)

        assert runninghub_api_key == "rh-key"
        assert runninghub_base_url == "https://rh.example.invalid"
        assert volcengine_api_key == "ark-key"
        assert volcengine_base_url == "https://las.example.invalid"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_resolve_runninghub_workflow_ids_and_api_format_dir():
    temp_dir = _workspace_temp_dir()
    try:
        api_dir = temp_dir / "runninghub_api_formats"
        api_dir.mkdir()
        config_path = temp_dir / "conf"
        config_path.write_text(
            "\n".join(
                [
                    "RunningHUB:",
                    "API_KEY=rh-key",
                    "RUNNINGHUB_WORKFLOW_RH_SHOT_KEYFRAME_FACEID_V1=123456",
                    f"RUNNINGHUB_API_FORMAT_DIR={api_dir}",
                ]
            ),
            encoding="utf-8",
        )

        workflow_ids = resolve_runninghub_workflow_ids(config_path)
        api_format_dir = resolve_runninghub_api_format_dir(config_path)

        assert workflow_ids["RUNNINGHUB_WORKFLOW_RH_SHOT_KEYFRAME_FACEID_V1"] == "123456"
        assert api_format_dir == api_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
