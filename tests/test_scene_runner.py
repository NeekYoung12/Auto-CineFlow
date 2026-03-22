"""Tests for scene runner helpers."""

from autocineflow.result_ingest import ArtifactDownloadBatch, ArtifactDownloadRecord
from autocineflow.scene_runner import _inject_bootstrap_keyframes, _is_runninghub_video_provider
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
