"""Helpers for resolving OpenAI-compatible credentials from env or local config."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse, urlunparse


def parse_key_value_config(path: str | Path) -> dict[str, str]:
    """Parse a loose key=value config file while ignoring free-form section labels."""

    resolved = Path(path)
    if not resolved.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in resolved.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line or line.endswith(":"):
            continue

        key, value = line.split("=", 1)
        cleaned_key = key.strip()
        cleaned_value = value.strip().strip('"').strip("'")
        if cleaned_key and cleaned_value:
            values[cleaned_key] = cleaned_value
    return values


def discover_default_config_path(start: str | Path | None = None) -> Path | None:
    """Look for a sibling workspace config/conf file near the repository."""

    if env_override := os.environ.get("AUTOCINEFLOW_CONFIG_PATH"):
        candidate = Path(env_override)
        return candidate if candidate.exists() else None

    root = Path(start or Path.cwd()).resolve()
    candidates = []
    for base in (root, *root.parents):
        candidates.extend(
            [
                base / "config" / "conf",
                base.parent / "config" / "conf",
            ]
        )

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def normalize_openai_base_url(base_url: str | None) -> str | None:
    """Normalise common gateway URLs to an OpenAI-compatible `/v1` endpoint."""

    if not base_url:
        return None

    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if not path:
        path = "/v1"
    elif not path.endswith("/v1") and not path.endswith("/v1/"):
        path = f"{path}/v1"

    return urlunparse(parsed._replace(path=path))


def resolve_openai_settings(
    api_key: str | None = None,
    base_url: str | None = None,
    config_path: str | Path | None = None,
) -> tuple[str, str | None]:
    """Resolve API key and base URL from explicit args, env vars, or local config."""

    if api_key and base_url:
        return api_key, base_url

    resolved_config = Path(config_path) if config_path else discover_default_config_path()
    config_values = parse_key_value_config(resolved_config) if resolved_config else {}

    resolved_api_key = (
        api_key
        or os.environ.get("OPENAI_API_KEY")
        or config_values.get("OPENAI_API_KEY")
        or ""
    )
    resolved_base_url = (
        base_url
        or os.environ.get("OPENAI_BASE_URL")
        or config_values.get("OPENAI_BASE_URL")
        or None
    )
    return resolved_api_key, normalize_openai_base_url(resolved_base_url)
