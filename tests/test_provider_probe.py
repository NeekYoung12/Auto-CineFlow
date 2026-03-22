"""Tests for provider readiness probes."""

import json
import shutil
import tempfile
from pathlib import Path

import httpx
import autocineflow.provider_probe as provider_probe_module

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
    monkeypatch.setattr(
        provider_probe_module,
        "resolve_local_vlm_settings",
        lambda config_path=None: {
            "python_path": "D:/missing/python.exe",
            "model_path": "D:/missing/model",
            "device_preference": "cuda",
            "min_free_vram_gb": "4",
        },
    )

    temp_dir = _workspace_temp_dir()
    try:
        config_path = temp_dir / "conf"
        config_path.write_text(
            "\n".join(
                [
                    "RunningHUB:",
                    "API_KEY=rh-key",
                    "RUNNINGHUB_BASE_URL=https://rh.example.invalid",
                    "RUNNINGHUB_WORKFLOW_RH_CHAR_IDENTITY_FORGE_V1=100",
                    "RUNNINGHUB_WORKFLOW_RH_CHAR_SHEET_MULTIVIEW_V1=101",
                    "RUNNINGHUB_WORKFLOW_RH_SCENE_SET_FORGE_V1=102",
                    "RUNNINGHUB_WORKFLOW_RH_SHOT_KEYFRAME_FACEID_V1=103",
                    "RUNNINGHUB_WORKFLOW_RH_SHOT_RELIGHT_MATCH_V1=104",
                    "RUNNINGHUB_WORKFLOW_RH_SHOT_REPAIR_INPAINT_V1=105",
                    "RUNNINGHUB_WORKFLOW_RH_SHOT_I2V_WAN22_FULL_V1=106",
                    "RUNNINGHUB_WORKFLOW_RH_SHOT_I2V_WAN21_HQ_V1=107",
                    "RUNNINGHUB_WORKFLOW_RH_SHOT_I2V_FRAMEPACK_FAST_V1=108",
                    "volcengine:",
                    "ARK_API_KEY=ark-key",
                    "VOLCENGINE_ARK_BASE_URL=https://ark.example.invalid/api/v3",
                ]
            ),
            encoding="utf-8",
        )
        api_dir = temp_dir / "runninghub_api_formats"
        api_dir.mkdir()
        for name in (
            "rh_char_identity_forge_v1",
            "rh_char_sheet_multiview_v1",
            "rh_scene_set_forge_v1",
            "rh_shot_keyframe_faceid_v1",
            "rh_shot_relight_match_v1",
            "rh_shot_repair_inpaint_v1",
            "rh_shot_i2v_wan22_full_v1",
            "rh_shot_i2v_wan21_hq_v1",
            "rh_shot_i2v_framepack_fast_v1",
        ):
            (api_dir / f"{name}.json").write_text("{}", encoding="utf-8")
        with config_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\nRUNNINGHUB_API_FORMAT_DIR={api_dir}\n")
        report = build_provider_probe_report(config_path=str(config_path), timeout_seconds=10.0)
        assert len(report.results) == 4
        assert report.results[0].provider == "volcengine_ark"
        assert report.results[0].ok is True
        assert report.results[1].provider == "runninghub"
        assert report.results[1].ok is True
        assert report.results[2].provider == "runninghub_workflows"
        assert report.results[2].ok is True
        assert report.results[3].provider == "local_visual_review"
        assert report.results[3].ok is False

        files = write_provider_probe_report(report, temp_dir / "probe")
        payload = json.loads(files["probe_report"].read_text(encoding="utf-8"))
        assert payload["results"][1]["details"]["data"]["balance"] == 12
        assert payload["results"][2]["details"]["missing_api_formats"] == []
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
