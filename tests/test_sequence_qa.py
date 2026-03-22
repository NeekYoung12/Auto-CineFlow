"""Tests for assembled sequence QA and repair planning."""

import json
import shutil
import tempfile
from pathlib import Path

from autocineflow.pipeline import CineFlowPipeline


def _workspace_temp_dir() -> Path:
    """Create a writable temp directory inside the repository workspace."""

    root = Path.cwd() / ".pytest_tmp"
    root.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(dir=root))


def _build_plan():
    pipeline = CineFlowPipeline()
    context = pipeline.run(
        description="A man in black coat faces a woman in red dress across a tavern table.",
        num_shots=5,
        scene_id="SEQ_QA_SCENE",
        use_llm=False,
        emotion_override="tense",
    )
    package = pipeline.build_storyboard_package(context, project_name="Sequence QA")
    plan = pipeline.build_sequence_assembly_plan(package, artifacts_dir="artifacts")
    return pipeline, plan


def test_sequence_qa_report_detects_missing_clips():
    pipeline, plan = _build_plan()
    temp_dir = _workspace_temp_dir()
    try:
        report = pipeline.build_sequence_qa_report(plan, temp_dir)
        assert report.passes_gate is False
        assert report.missing_clips == report.clip_count
        repair_plan = pipeline.build_sequence_repair_plan(report)
        assert repair_plan.action_count >= report.clip_count
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_sequence_qa_report_passes_when_clips_and_sequence_exist():
    pipeline, plan = _build_plan()
    temp_dir = _workspace_temp_dir()
    try:
        for clip in plan.clips:
            clip_path = temp_dir / Path(clip.expected_asset_path).name
            clip_path.write_bytes(b"fake-video")
        (temp_dir / f"{plan.scene_id.lower()}_sequence.mp4").write_bytes(b"fake-sequence")

        report = pipeline.build_sequence_qa_report(plan, temp_dir)
        assert report.passes_gate is True
        assert report.score == 1.0
        repair_plan = pipeline.build_sequence_repair_plan(report)
        assert repair_plan.action_count == 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_write_sequence_qc_outputs_creates_files():
    pipeline, plan = _build_plan()
    temp_dir = _workspace_temp_dir()
    try:
        report = pipeline.build_sequence_qa_report(plan, temp_dir)
        repair_plan = pipeline.build_sequence_repair_plan(report)
        files = pipeline.write_sequence_qc_outputs(report, repair_plan, temp_dir / "qc")
        assert set(files.keys()) == {"report_json", "report_markdown", "repair_json", "repair_markdown"}
        assert files["report_json"].exists()
        assert files["repair_json"].exists()
        payload = json.loads(files["report_json"].read_text(encoding="utf-8"))
        assert payload["scene_id"] == "SEQ_QA_SCENE"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
