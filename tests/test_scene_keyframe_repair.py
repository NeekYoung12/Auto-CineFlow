"""Tests for keyframe-repair job selection."""

from autocineflow.pipeline import CineFlowPipeline
from autocineflow.scene_keyframe_repair import select_keyframe_repair_jobs
from autocineflow.submission import SubmissionProvider


def test_select_keyframe_repair_jobs_filters_blocked_shots():
    pipeline = CineFlowPipeline()
    context = pipeline.run(
        description="A man in black coat faces a woman in red dress across a tavern table.",
        num_shots=5,
        scene_id="KEYFRAME_REPAIR_SCENE",
        use_llm=False,
        emotion_override="tense",
    )
    package = pipeline.build_storyboard_package(context, project_name="Keyframe Repair")
    jobs = pipeline.build_submission_jobs_from_package(package, provider=SubmissionProvider.RUNNINGHUB_FACEID)

    gate_payload = {
        "results": [
            {"shot_id": jobs[0].shot_id, "passes_gate": False},
            {"shot_id": jobs[1].shot_id, "passes_gate": True},
        ]
    }
    selected = select_keyframe_repair_jobs(jobs, gate_payload)

    assert len(selected) == 1
    assert selected[0].shot_id == jobs[0].shot_id
