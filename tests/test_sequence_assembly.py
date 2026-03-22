"""Tests for sequence assembly planning."""

import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from autocineflow.pipeline import CineFlowPipeline


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
        scene_id="ASSEMBLY_SCENE",
        use_llm=False,
        emotion_override="tense",
    )
    package = pipeline.build_storyboard_package(context, project_name="Assembly Project")
    return pipeline, package


def test_recommended_video_shot_count_prefers_finer_coverage():
    pipeline = CineFlowPipeline()
    assert pipeline.recommended_video_shot_count(30.0, clip_duration_seconds=4.0) == 8
    assert pipeline.recommended_video_shot_count(80.0, clip_duration_seconds=4.0) == 20


def test_sequence_assembly_plan_matches_video_segments():
    pipeline, package = _build_package()
    plan = pipeline.build_sequence_assembly_plan(package)
    payload = json.loads(pipeline.sequence_assembly_json(plan, indent=2))

    assert plan.scene_id == "ASSEMBLY_SCENE"
    assert plan.clip_count == len(package.video_segments)
    assert payload["clip_count"] == len(package.video_segments)
    assert plan.clips[0].expected_asset_path.endswith(".mp4")


def test_write_sequence_assembly_plan_creates_ffmpeg_files():
    pipeline, package = _build_package()
    plan = pipeline.build_sequence_assembly_plan(package)
    temp_dir = _workspace_temp_dir()
    try:
        files = pipeline.write_sequence_assembly_plan(plan, temp_dir)
        assert set(files.keys()) == {"assembly_json", "concat_manifest", "concat_script"}
        assert files["assembly_json"].exists()
        assert files["concat_manifest"].exists()
        assert files["concat_script"].exists()
        assert "ffmpeg" in files["concat_script"].read_text(encoding="utf-8").lower()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_assemble_sequence_with_ffmpeg_invokes_expected_command(monkeypatch):
    pipeline, package = _build_package()
    plan = pipeline.build_sequence_assembly_plan(package)
    temp_dir = _workspace_temp_dir()
    try:
        files = pipeline.write_sequence_assembly_plan(plan, temp_dir)
        output_path = temp_dir / "assembly_sequence.mp4"

        monkeypatch.setattr(shutil, "which", lambda name: "ffmpeg.exe")

        def fake_run(command, capture_output, text, check):
            output_path.write_bytes(b"fake-sequence")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        import subprocess

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = pipeline.assemble_sequence_with_ffmpeg(plan, temp_dir, output_path=output_path)
        assert result.assembled is True
        assert result.output_path == str(output_path)
        assert result.command[0] == "ffmpeg"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
