"""Tests for render submission job building and backends."""

import json
import shutil
import tempfile
from pathlib import Path

from autocineflow.pipeline import CineFlowPipeline
from autocineflow.project_delivery import ProjectSceneInput
from autocineflow.submission import SubmissionBackend, SubmissionProvider, SubmissionTarget


def _workspace_temp_dir() -> Path:
    """Create a writable temp directory inside the repository workspace."""

    root = Path.cwd() / ".pytest_tmp"
    root.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(dir=root))


def _build_package():
    pipeline = CineFlowPipeline()
    context = pipeline.run(
        description="A man in black coat faces a woman in red dress across a tavern table.",
        num_shots=5,
        scene_id="SUBMIT_SCENE",
        use_llm=False,
        emotion_override="tense",
    )
    package = pipeline.build_storyboard_package(context, project_name="Submission Project")
    return pipeline, package


def _build_execution_plan():
    pipeline = CineFlowPipeline()
    previous = pipeline.build_project_package(
        scene_inputs=[
            ProjectSceneInput(
                scene_id="SCENE_A",
                description="A man in black coat faces a woman in red dress across a tavern table.",
                num_shots=5,
                emotion_override="tense",
            )
        ],
        project_name="Submission Project",
        use_llm=False,
        min_score=0.8,
        max_attempts=2,
    )
    current = pipeline.build_project_package(
        scene_inputs=[
            ProjectSceneInput(
                scene_id="SCENE_A",
                description="A man in black coat faces a woman in red dress across a tavern table. He suddenly slams the glass down.",
                num_shots=5,
                emotion_override="tense",
            )
        ],
        project_name="Submission Project",
        use_llm=False,
        min_score=0.8,
        max_attempts=2,
    )
    temp_dir = _workspace_temp_dir()
    previous_files = pipeline.write_project_package(previous, temp_dir / "previous")
    return pipeline, temp_dir, previous, current, pipeline.build_project_execution_plan(previous, current, previous_files["scenes_dir"])


def test_build_submission_jobs_from_package_for_multiple_providers():
    pipeline, package = _build_package()
    generic_jobs = pipeline.build_submission_jobs_from_package(package)
    a1111_jobs = pipeline.build_submission_jobs_from_package(package, provider=SubmissionProvider.AUTOMATIC1111)
    comfy_jobs = pipeline.build_submission_jobs_from_package(package, provider=SubmissionProvider.COMFYUI)

    assert len(generic_jobs) == 5
    assert len(a1111_jobs) == 5
    assert len(comfy_jobs) == 5
    assert a1111_jobs[0].payload["seed"] == package.shots[0].render_seed
    assert comfy_jobs[0].payload["workflow"]["seed"] == package.shots[0].render_seed


def test_submit_jobs_to_filesystem_queue():
    pipeline, package = _build_package()
    jobs = pipeline.build_submission_jobs_from_package(package)
    temp_dir = _workspace_temp_dir()
    try:
        batch = pipeline.submit_jobs(
            jobs,
            SubmissionTarget(backend=SubmissionBackend.FILESYSTEM, spool_dir=str(temp_dir / "spool")),
            source_type="package",
            source_id=package.scene_id,
        )
        assert batch.job_count == 5
        assert all(record.status == "queued" for record in batch.records)
        queued_files = list((temp_dir / "spool" / "queued").glob("*.json"))
        assert len(queued_files) == 5
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_submit_jobs_from_execution_plan_dry_run():
    pipeline, temp_dir, previous, current, plan = _build_execution_plan()
    try:
        jobs = pipeline.build_submission_jobs_from_execution_plan(plan)
        batch = pipeline.submit_jobs(
            jobs,
            SubmissionTarget(backend=SubmissionBackend.DRY_RUN),
            source_type="execution_plan",
            source_id=plan.current_project_name,
        )
        assert batch.job_count == len(plan.ordered_rerender_queue)
        assert all(record.status == "dry_run" for record in batch.records)
        payload = json.loads(pipeline.submission_batch_json(batch, indent=2))
        assert payload["source_type"] == "execution_plan"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_write_submission_batch_creates_outputs():
    pipeline, package = _build_package()
    jobs = pipeline.build_submission_jobs_from_package(package)
    temp_dir = _workspace_temp_dir()
    try:
        batch = pipeline.submit_jobs(
            jobs,
            SubmissionTarget(backend=SubmissionBackend.DRY_RUN),
            source_type="package",
            source_id=package.scene_id,
        )
        files = pipeline.write_submission_batch(batch, temp_dir)
        assert set(files.keys()) == {"batch_json", "batch_markdown"}
        assert files["batch_json"].exists()
        assert files["batch_markdown"].exists()
        assert json.loads(files["batch_json"].read_text(encoding="utf-8"))["job_count"] == 5
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
