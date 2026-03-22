"""Tests for strategy-based failure recovery planning."""

import json
import shutil
import tempfile
from pathlib import Path

from autocineflow.pipeline import CineFlowPipeline
from autocineflow.submission import SubmissionBackend, SubmissionProvider, SubmissionTarget


def _workspace_temp_dir() -> Path:
    """Create a writable temp directory inside the repository workspace."""

    root = Path.cwd() / ".pytest_tmp"
    root.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(dir=root))


def test_recovery_plan_classifies_balance_and_retry_failures():
    pipeline = CineFlowPipeline()
    context = pipeline.run(
        description="A man in black coat faces a woman in red dress across a tavern table.",
        num_shots=5,
        scene_id="RECOVERY_SCENE",
        use_llm=False,
        emotion_override="tense",
    )
    package = pipeline.build_storyboard_package(context, project_name="Recovery Project")
    jobs = pipeline.build_submission_jobs_from_package(package, provider=SubmissionProvider.MINIMAX_VIDEO)
    batch = pipeline.submit_jobs(
        jobs[:2],
        SubmissionTarget(backend=SubmissionBackend.DRY_RUN),
        source_type="package",
        source_id=package.scene_id,
    )
    batch.records[0].provider_status_code = 1008
    batch.records[0].provider_status_message = "insufficient balance"
    batch.records[1].provider_status_code = 1000
    batch.records[1].provider_status_message = "unexpected error"

    plan = pipeline.build_recovery_plan(batch)
    assert plan.queue_paused is True
    assert len(plan.retry_job_ids) == 1
    assert plan.decisions[0].action.value == "PAUSE_QUEUE"
    assert plan.decisions[1].action.value == "RETRY"


def test_write_recovery_plan_creates_outputs():
    pipeline = CineFlowPipeline()
    context = pipeline.run(
        description="A man in black coat faces a woman in red dress across a tavern table.",
        num_shots=5,
        scene_id="RECOVERY_SCENE",
        use_llm=False,
        emotion_override="tense",
    )
    package = pipeline.build_storyboard_package(context, project_name="Recovery Project")
    jobs = pipeline.build_submission_jobs_from_package(package, provider=SubmissionProvider.MINIMAX_VIDEO)
    batch = pipeline.submit_jobs(
        jobs[:1],
        SubmissionTarget(backend=SubmissionBackend.DRY_RUN),
        source_type="package",
        source_id=package.scene_id,
    )
    batch.records[0].provider_status_code = 2013
    batch.records[0].provider_status_message = "invalid params"
    plan = pipeline.build_recovery_plan(batch)

    temp_dir = _workspace_temp_dir()
    try:
        files = pipeline.write_recovery_plan(plan, temp_dir)
        assert set(files.keys()) == {"plan_json", "plan_markdown"}
        assert files["plan_json"].exists()
        payload = json.loads(files["plan_json"].read_text(encoding="utf-8"))
        assert payload["decision_count"] == 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_recovery_plan_parses_status_from_raw_message_json():
    pipeline = CineFlowPipeline()
    context = pipeline.run(
        description="A man in black coat faces a woman in red dress across a tavern table.",
        num_shots=5,
        scene_id="RECOVERY_SCENE",
        use_llm=False,
        emotion_override="tense",
    )
    package = pipeline.build_storyboard_package(context, project_name="Recovery Project")
    jobs = pipeline.build_submission_jobs_from_package(package, provider=SubmissionProvider.MINIMAX_VIDEO)
    batch = pipeline.submit_jobs(
        jobs[:1],
        SubmissionTarget(backend=SubmissionBackend.DRY_RUN),
        source_type="package",
        source_id=package.scene_id,
    )
    batch.records[0].message = '{"task_id":"","base_resp":{"status_code":1008,"status_msg":"insufficient balance"}}'
    batch.records[0].backend_job_id = ""

    plan = pipeline.build_recovery_plan(batch)
    assert plan.queue_paused is True
    assert plan.decisions[0].reason == "provider_balance_exhausted"
