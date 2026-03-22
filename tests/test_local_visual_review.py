"""Tests for optional local visual review integration."""

from pathlib import Path

import autocineflow.local_visual_review as local_visual_review_module
from autocineflow.keyframe_qa import KeyframeQAChecks, KeyframeQAReport, KeyframeQAResult
from autocineflow.scene_runner import _should_block_video_from_visual_review
from autocineflow.submission import SubmissionProvider


def test_local_visual_review_skips_when_runtime_paths_missing(monkeypatch):
    monkeypatch.setattr(
        local_visual_review_module,
        "resolve_local_vlm_settings",
        lambda config_path=None: {
            "python_path": "D:/missing/python.exe",
            "model_path": "D:/missing/model",
            "device_preference": "cuda",
            "min_free_vram_gb": "4",
        },
    )

    report = local_visual_review_module.review_keyframes_with_local_vlm(
        KeyframeQAReport(source_id="SCENE_01", score=1.0, min_score=0.75, passes_gate=True, results=[]),
        config_path="conf",
    )
    assert report.enabled is False
    assert report.skipped is True


def test_visual_review_can_block_runninghub_video_when_repair_is_requested():
    review_report = local_visual_review_module.LocalVisualReviewReport(
        source_id="SCENE_01",
        enabled=True,
        skipped=False,
        results=[
            local_visual_review_module.LocalVisualReviewResult(
                shot_id="SHOT_01",
                output_path="frame.png",
                status="ok",
                score=0.2,
                recommendation="repair",
                issues=["text_artifact"],
            )
        ],
    )
    assert _should_block_video_from_visual_review(SubmissionProvider.RUNNINGHUB_VIDEO_AUTO, review_report) is True
    assert _should_block_video_from_visual_review(SubmissionProvider.RUNNINGHUB_FACEID, review_report) is False
