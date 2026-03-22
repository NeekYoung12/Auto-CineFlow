"""Tests for asset library indexing and latest-version selection."""

import json
import shutil
import tempfile
from pathlib import Path

from autocineflow.pipeline import CineFlowPipeline
from autocineflow.project_delivery import ProjectSceneInput


def _workspace_temp_dir() -> Path:
    """Create a writable temp directory inside the repository workspace."""

    root = Path.cwd() / ".pytest_tmp"
    root.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(dir=root))


def _mark_manifest_rendered(manifest_path: Path) -> None:
    """Mark a render manifest as rendered for indexing tests."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in payload:
        item["status"] = "rendered"
        item["output_path"] = f"renders/{item['expected_filename']}"
        item["actual_seed"] = item["expected_seed"]
        item["actual_width"] = item["expected_width"]
        item["actual_height"] = item["expected_height"]
        item["actual_prompt_hash"] = item["expected_prompt_hash"]
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_asset_library_indexes_scene_and_project_versions():
    pipeline = CineFlowPipeline()
    temp_dir = _workspace_temp_dir()
    try:
        # Scene output
        scene_ctx = pipeline.run(
            description="A man in black coat faces a woman in red dress across a tavern table.",
            num_shots=5,
            scene_id="LIB_SCENE",
            use_llm=False,
            emotion_override="tense",
        )
        scene_package = pipeline.build_storyboard_package(scene_ctx, project_name="Library Project")
        scene_files = pipeline.write_delivery_package(scene_package, temp_dir / "scene_run")
        # Add recovery metadata for indexing.
        submission_payload = {
            "provider": "minimax_video",
            "job_count": 1,
            "records": [
                {
                    "backend_job_id": "",
                    "provider_status_code": 1008,
                    "provider_status_message": "insufficient balance",
                }
            ],
        }
        (temp_dir / "scene_run" / "submission").mkdir(exist_ok=True)
        (temp_dir / "scene_run" / "submission" / "submission_batch.json").write_text(
            json.dumps(submission_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        recovery_payload = {"decision_count": 1, "queue_paused": True}
        (temp_dir / "scene_run" / "recovery").mkdir(exist_ok=True)
        (temp_dir / "scene_run" / "recovery" / "recovery_plan.json").write_text(
            json.dumps(recovery_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Project output
        project = pipeline.build_project_package(
            scene_inputs=[
                ProjectSceneInput(
                    scene_id="LIB_SCENE",
                    description="A man in black coat faces a woman in red dress across a tavern table.",
                    num_shots=5,
                    emotion_override="tense",
                )
            ],
            project_name="Library Project",
            use_llm=False,
            min_score=0.8,
            max_attempts=2,
        )
        project_files = pipeline.write_project_package(project, temp_dir / "project_run")

        library = pipeline.build_asset_library(temp_dir)
        assert len(library.scene_versions) >= 2
        assert len(library.project_versions) == 1
        latest_scene = pipeline.latest_scene_versions(library)[0]
        latest_project = pipeline.latest_project_versions(library)[0]
        assert latest_scene.scene_id == "LIB_SCENE"
        assert latest_project.project_name == "Library Project"
        assert any(scene.failed_submission_count == 1 for scene in library.scene_versions)
        assert any(scene.recovery_decision_count == 1 for scene in library.scene_versions)
        assert any(scene.queue_paused is True for scene in library.scene_versions)
        assert latest_project.total_failed_submissions >= 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_write_asset_library_creates_outputs():
    pipeline = CineFlowPipeline()
    temp_dir = _workspace_temp_dir()
    try:
        project = pipeline.build_project_package(
            scene_inputs=[
                ProjectSceneInput(
                    scene_id="LIB_SCENE",
                    description="A man in black coat faces a woman in red dress across a tavern table.",
                    num_shots=5,
                    emotion_override="tense",
                )
            ],
            project_name="Library Project",
            use_llm=False,
            min_score=0.8,
            max_attempts=2,
        )
        pipeline.write_project_package(project, temp_dir / "project_run")
        library = pipeline.build_asset_library(temp_dir)
        files = pipeline.write_asset_library(library, temp_dir / "library")
        assert set(files.keys()) == {"library_json", "library_markdown"}
        assert files["library_json"].exists()
        assert files["library_markdown"].exists()
        payload = json.loads(files["library_json"].read_text(encoding="utf-8"))
        assert payload["root_dir"] == str(temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
