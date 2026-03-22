"""Tests for scene runner helpers."""

from autocineflow.result_ingest import ArtifactDownloadBatch, ArtifactDownloadRecord
from autocineflow.scene_runner import (
    _build_rebuild_keyframe_jobs,
    _inject_bootstrap_keyframes,
    _is_runninghub_video_provider,
    _runninghub_ai_post_enhance_available,
)
from autocineflow.submission import SubmissionJob, SubmissionProvider


def test_is_runninghub_video_provider_recognizes_all_video_modes():
    assert _is_runninghub_video_provider(SubmissionProvider.RUNNINGHUB_VIDEO_AUTO) is True
    assert _is_runninghub_video_provider(SubmissionProvider.RUNNINGHUB_VIDEO_QUALITY) is True
    assert _is_runninghub_video_provider(SubmissionProvider.RUNNINGHUB_VIDEO_FAST) is True
    assert _is_runninghub_video_provider(SubmissionProvider.RUNNINGHUB_FACEID) is False


def test_inject_bootstrap_keyframes_prefers_downloaded_frame():
    jobs = [
        SubmissionJob(
            job_id="seg-01",
            shot_id="SHOT_01",
            scene_id="SCENE_01",
            provider=SubmissionProvider.RUNNINGHUB_VIDEO_AUTO,
            payload={
                "workflow_key": "rh_shot_i2v_wan22_full_v1",
                "workflow_id_env": "RUNNINGHUB_WORKFLOW_RH_SHOT_I2V_WAN22_FULL_V1",
                "request_contract": {
                    "first_frame_candidates": ["D:/old.png"],
                },
            },
        )
    ]
    downloads = ArtifactDownloadBatch(
        source_id="SCENE_01",
        records=[
            ArtifactDownloadRecord(
                job_id="bootstrap-01",
                shot_id="SHOT_01",
                url="https://example.invalid/frame.png",
                output_path="D:/bootstrap.png",
                downloaded=True,
            )
        ],
    )

    patched = _inject_bootstrap_keyframes(jobs, downloads)

    assert patched[0].payload["request_contract"]["first_frame_candidates"][0] == "D:/bootstrap.png"


def test_build_rebuild_keyframe_jobs_uses_bootstrap_outputs_and_boosts_steps():
    jobs = [
        SubmissionJob(
            job_id="kf-01",
            shot_id="SHOT_01",
            scene_id="SCENE_01",
            provider=SubmissionProvider.RUNNINGHUB_FACEID,
            payload={
                "workflow_key": "rh_shot_keyframe_faceid_v1",
                "workflow_id_env": "RUNNINGHUB_WORKFLOW_RH_SHOT_KEYFRAME_FACEID_V1",
                "workflow_inputs": {
                    "positive_prompt": "detective in alley",
                    "steps": 30,
                    "scene_reference_images": ["D:/scene.png"],
                },
            },
        )
    ]
    downloads = ArtifactDownloadBatch(
        source_id="SCENE_01",
        records=[
            ArtifactDownloadRecord(
                job_id="bootstrap-01",
                shot_id="SHOT_01",
                url="https://example.invalid/frame.png",
                output_path="D:/bootstrap.png",
                downloaded=True,
            )
        ],
    )

    rebuilt = _build_rebuild_keyframe_jobs(jobs, downloads)

    assert rebuilt[0].job_id == "kf-01_REBUILD"
    assert rebuilt[0].payload["workflow_inputs"]["scene_reference_images"][0] == "D:/bootstrap.png"
    assert rebuilt[0].payload["workflow_inputs"]["steps"] >= 42
    assert "high fidelity facial detail" in rebuilt[0].payload["workflow_inputs"]["positive_prompt"]


def test_runninghub_ai_post_enhance_available_looks_for_optional_workflow_id(monkeypatch):
    monkeypatch.setattr(
        "autocineflow.scene_runner.resolve_runninghub_workflow_ids",
        lambda config_path=None: {"RUNNINGHUB_WORKFLOW_RH_VIDEO_POST_ENHANCE_V1": "123"},
    )
    assert _runninghub_ai_post_enhance_available("conf") is True
