"""Tests for selecting clip rerender jobs from a sequence repair plan."""

from autocineflow.pipeline import CineFlowPipeline
from autocineflow.scene_repair import select_repair_jobs
from autocineflow.sequence_qa import SequenceRepairAction, SequenceRepairPlan
from autocineflow.submission import SubmissionProvider


def test_select_repair_jobs_filters_by_clip_id():
    pipeline = CineFlowPipeline()
    context = pipeline.run(
        description="A man in black coat faces a woman in red dress across a tavern table.",
        num_shots=5,
        scene_id="REPAIR_SCENE",
        use_llm=False,
        emotion_override="tense",
    )
    package = pipeline.build_storyboard_package(context, project_name="Repair Project")
    jobs = pipeline.build_submission_jobs_from_package(package, provider=SubmissionProvider.MINIMAX_VIDEO)

    repair_plan = SequenceRepairPlan(
        scene_id="REPAIR_SCENE",
        action_count=2,
        actions=[
            SequenceRepairAction(
                clip_id=jobs[1].job_id,
                shot_id=jobs[1].shot_id,
                action="rerender_clip",
                reasons=["missing_clip_asset"],
            ),
            SequenceRepairAction(
                clip_id="FINAL_SEQUENCE",
                shot_id="FINAL_SEQUENCE",
                action="restitch_sequence",
                reasons=["missing_final_sequence"],
            ),
        ],
    )

    selected = select_repair_jobs(jobs, repair_plan)
    assert len(selected) == 1
    assert selected[0].job_id == jobs[1].job_id
