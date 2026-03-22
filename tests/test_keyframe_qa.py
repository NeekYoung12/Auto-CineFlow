"""Tests for keyframe QA and repair planning."""

import shutil
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from autocineflow.keyframe_qa import (
    analyze_keyframe,
    build_keyframe_repair_jobs,
    keyframe_qa_report,
    select_best_keyframe_downloads,
)
from autocineflow.result_ingest import ArtifactDownloadBatch, ArtifactDownloadRecord
from autocineflow.submission import SubmissionJob, SubmissionProvider


def _workspace_temp_dir() -> Path:
    root = Path.cwd() / ".pytest_tmp"
    root.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(dir=root))


def _write_clean_image(path: Path) -> None:
    image = Image.new("RGB", (1536, 864), (45, 55, 70))
    draw = ImageDraw.Draw(image)
    draw.rectangle((980, 130, 1320, 760), fill=(60, 35, 20))
    draw.rectangle((120, 580, 420, 830), fill=(75, 30, 28))
    image.save(path)


def _write_texty_image(path: Path) -> None:
    image = Image.new("RGB", (960, 1440), (220, 190, 80))
    draw = ImageDraw.Draw(image)
    for index in range(20):
        x = 70 + (index % 3) * 40
        y = 80 + index * 55
        draw.line((x, y, x + 15, y + 35), fill=(10, 10, 10), width=5)
        draw.line((x + 15, y, x, y + 35), fill=(10, 10, 10), width=5)
    image.save(path)


def test_analyze_keyframe_flags_text_artifact_and_low_resolution():
    temp_dir = _workspace_temp_dir()
    try:
        image_path = temp_dir / "bad.png"
        _write_texty_image(image_path)
        result = analyze_keyframe(image_path)

        assert result.checks.resolution_ok is False
        assert result.checks.text_artifact_risk_low is False
        assert "text_artifact_risk" in result.issues
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_select_best_keyframe_downloads_prefers_clean_image():
    temp_dir = _workspace_temp_dir()
    try:
        clean = temp_dir / "clean.png"
        bad = temp_dir / "bad.png"
        _write_clean_image(clean)
        _write_texty_image(bad)

        bad_batch = ArtifactDownloadBatch(
            source_id="SCENE_01",
            records=[ArtifactDownloadRecord(job_id="bad", shot_id="SHOT_01", url="", output_path=str(bad), downloaded=True)],
        )
        clean_batch = ArtifactDownloadBatch(
            source_id="SCENE_01",
            records=[ArtifactDownloadRecord(job_id="clean", shot_id="SHOT_01", url="", output_path=str(clean), downloaded=True)],
        )

        selected = select_best_keyframe_downloads(bad_batch, clean_batch)
        assert selected.records[0].output_path == str(clean)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_build_keyframe_repair_jobs_strengthens_prompt_when_text_artifact_found():
    temp_dir = _workspace_temp_dir()
    try:
        bad = temp_dir / "bad.png"
        _write_texty_image(bad)
        batch = ArtifactDownloadBatch(
            source_id="SCENE_01",
            records=[ArtifactDownloadRecord(job_id="bad", shot_id="SHOT_01", url="", output_path=str(bad), downloaded=True)],
        )
        report = keyframe_qa_report(batch)

        jobs = [
            SubmissionJob(
                job_id="kf-01",
                shot_id="SHOT_01",
                scene_id="SCENE_01",
                provider=SubmissionProvider.RUNNINGHUB_FACEID,
                payload={
                    "workflow_key": "rh_shot_keyframe_faceid_v1",
                    "workflow_inputs": {
                        "positive_prompt": "detective in alley",
                        "negative_prompt": "blurry",
                        "steps": 30,
                    },
                },
            )
        ]
        repaired = build_keyframe_repair_jobs(jobs, report)
        prompt = repaired[0].payload["workflow_inputs"]["positive_prompt"]
        negative = repaired[0].payload["workflow_inputs"]["negative_prompt"]
        assert "no visible text" in prompt
        assert "garbled typography" in negative
        assert repaired[0].payload["workflow_inputs"]["steps"] >= 50
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
