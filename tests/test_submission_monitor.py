"""Tests for filesystem submission monitoring."""

import json
import shutil
import tempfile
from pathlib import Path

from autocineflow.pipeline import CineFlowPipeline
from autocineflow.submission import SubmissionBackend, SubmissionTarget


def _workspace_temp_dir() -> Path:
    """Create a writable temp directory inside the repository workspace."""

    root = Path.cwd() / ".pytest_tmp"
    root.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(dir=root))


def _build_submission_batch(temp_dir: Path):
    pipeline = CineFlowPipeline()
    context = pipeline.run(
        description="A man in black coat faces a woman in red dress across a tavern table.",
        num_shots=5,
        scene_id="MONITOR_SCENE",
        use_llm=False,
        emotion_override="tense",
    )
    package = pipeline.build_storyboard_package(context, project_name="Monitor Project")
    jobs = pipeline.build_submission_jobs_from_package(package)
    batch = pipeline.submit_jobs(
        jobs,
        SubmissionTarget(backend=SubmissionBackend.FILESYSTEM, spool_dir=str(temp_dir / "spool")),
        source_type="package",
        source_id=package.scene_id,
    )
    return pipeline, batch, temp_dir / "spool"


def test_monitor_filesystem_submission_batch_reports_queued_jobs():
    temp_dir = _workspace_temp_dir()
    try:
        pipeline, batch, spool_dir = _build_submission_batch(temp_dir)
        report = pipeline.monitor_filesystem_submission_batch(batch, spool_dir)
        assert report.queued_jobs == 5
        assert report.completed_jobs == 0
        assert report.all_finished is False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_monitor_reports_completed_jobs_when_files_move():
    temp_dir = _workspace_temp_dir()
    try:
        pipeline, batch, spool_dir = _build_submission_batch(temp_dir)
        queued_dir = spool_dir / "queued"
        completed_dir = spool_dir / "completed"
        completed_dir.mkdir(parents=True, exist_ok=True)
        for job_file in queued_dir.glob("*.json"):
            job_file.rename(completed_dir / job_file.name)

        report = pipeline.monitor_filesystem_submission_batch(batch, spool_dir)
        assert report.completed_jobs == 5
        assert report.queued_jobs == 0
        assert report.all_finished is True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_write_submission_monitor_report_creates_outputs():
    temp_dir = _workspace_temp_dir()
    try:
        pipeline, batch, spool_dir = _build_submission_batch(temp_dir)
        report = pipeline.monitor_filesystem_submission_batch(batch, spool_dir)
        files = pipeline.write_submission_monitor_report(report, temp_dir / "monitor")
        assert set(files.keys()) == {"report_json", "report_markdown"}
        assert files["report_json"].exists()
        assert files["report_markdown"].exists()
        payload = json.loads(files["report_json"].read_text(encoding="utf-8"))
        assert payload["source_id"] == "MONITOR_SCENE"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
