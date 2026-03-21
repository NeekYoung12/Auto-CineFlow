"""Tests for the enhanced shot planner and acceptance helpers."""

from autocineflow.models import BeatType, ShotType
from autocineflow.pipeline import CineFlowPipeline


def test_angry_scene_produces_planned_progression():
    pipeline = CineFlowPipeline()
    ctx = pipeline.run(
        description=(
            "Two people sit facing each other across a tavern table. "
            "One of them suddenly slams the glass down and erupts in anger."
        ),
        num_shots=5,
        use_llm=False,
    )

    shot_types = [shot.framing.shot_type for shot in ctx.shot_blocks]
    assert shot_types[0] == ShotType.MASTER_SHOT
    assert ShotType.OVER_SHOULDER in shot_types
    assert ctx.shot_blocks[-2].beat_type == BeatType.ESCALATION
    assert ctx.shot_blocks[-2].framing.shot_type == ShotType.CLOSE_UP
    assert ctx.shot_blocks[-1].beat_type == BeatType.REACTION
    assert ctx.shot_blocks[-1].framing.subjects == ["CHAR_B"]


def test_acceptance_report_and_render_metadata_are_populated():
    pipeline = CineFlowPipeline()
    ctx = pipeline.run(
        description="Two people sit facing each other across a tavern table, tension rising.",
        num_shots=5,
        use_llm=False,
        emotion_override="tense",
    )

    report = pipeline.acceptance_report(ctx)
    assert all(report.values())
    for shot in ctx.shot_blocks:
        assert shot.composition is not None
        assert shot.camera_position is not None
        assert shot.controlnet_points
    production = pipeline.production_readiness_report(ctx)
    assert production["subject_coverage"] is True
    assert production["cinematic_progression"] is True
    assert production["llm_analysis_active"] is False


def test_chinese_scene_runs_end_to_end():
    pipeline = CineFlowPipeline()
    ctx = pipeline.run(
        description="两人在酒馆对坐，空气越来越紧张，其中一人突然愤怒地拍桌。",
        num_shots=5,
        use_llm=False,
    )

    assert len(ctx.shot_blocks) == 5
    assert ctx.scene_location == "tavern"
    assert ctx.detected_emotion == "angry"
    assert pipeline.acceptance_report(ctx)["logic_axis_consistency"] is True
