"""Tests for provider readiness probes."""

import json
import shutil
import tempfile
from pathlib import Path

import httpx

from autocineflow.provider_probe import build_provider_probe_report, write_provider_probe_report


def _workspace_temp_dir() -> Path:
    root = Path.cwd() / ".pytest_tmp"
    root.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(dir=root))


def test_build_provider_probe_report(monkeypatch):
    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"code": 0, "msg": "ok", "data": {"balance": 12}}

    def fake_post(url, json, timeout):
        assert url == "https://rh.example.invalid/uc/openapi/accountStatus"
        assert json == {"apiKey": "rh-key"}
        return DummyResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

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
                    "VOLCENGINE_ARK_BASE_URL=https://ark.example.invalid/api/v3",
                ]
            ),
            encoding="utf-8",
        )
        report = build_provider_probe_report(config_path=str(config_path), timeout_seconds=10.0)
        assert len(report.results) == 2
        assert report.results[0].provider == "volcengine_ark"
        assert report.results[0].ok is True
        assert report.results[1].provider == "runninghub"
        assert report.results[1].ok is True

        files = write_provider_probe_report(report, temp_dir / "probe")
        payload = json.loads(files["probe_report"].read_text(encoding="utf-8"))
        assert payload["results"][1]["details"]["data"]["balance"] == 12
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
