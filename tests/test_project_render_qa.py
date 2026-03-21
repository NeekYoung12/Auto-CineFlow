"""Tests for project-level render QA aggregation."""

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


def _build_project():
    pipeline = CineFlowPipeline()
    project = pipeline.build_project_package(
        scene_inputs=[
            ProjectSceneInput(
                scene_id="SCENE_A",
                description="A man in black coat faces a woman in red dress across a tavern table.",
                num_shots=5,
                emotion_override="tense",
            ),
            ProjectSceneInput(
                scene_id="SCENE_B",
                description="A detective faces a wounded informant in a neon alley at night.",
                num_shots=5,
                emotion_override="tense",
            ),
        ],
        project_name="Project QA",
        use_llm=False,
        min_score=0.8,
        max_attempts=2,
    )
    return pipeline, project


def _mark_render_manifests_rendered(scenes_dir: Path) -> None:
    """Fill project render manifests with matching actual outputs."""

    for manifest_path in scenes_dir.glob("*/render_manifest_template.json"):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in payload:
            item["status"] = "rendered"
            item["output_path"] = f"renders/{item['expected_filename']}"
            item["actual_seed"] = item["expected_seed"]
            item["actual_width"] = item["expected_width"]
            item["actual_height"] = item["expected_height"]
            item["actual_prompt_hash"] = item["expected_prompt_hash"]
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_project_render_qa_report_passes_when_all_scenes_match():
    pipeline, project = _build_project()
    temp_dir = _workspace_temp_dir()
    try:
        files = pipeline.write_project_package(project, temp_dir / "project")
        _mark_render_manifests_rendered(files["scenes_dir"])

        report = pipeline.build_project_render_qa_report(project, files["scenes_dir"], min_score=0.9)
        assert report.all_scenes_pass is True
        assert report.average_score == 1.0
        assert report.total_expected_shots == 10
        assert report.total_rendered_shots == 10
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_project_render_qa_report_fails_when_scene_outputs_missing():
    pipeline, project = _build_project()
    temp_dir = _workspace_temp_dir()
    try:
        files = pipeline.write_project_package(project, temp_dir / "project")
        _mark_render_manifests_rendered(files["scenes_dir"])
        # Remove one scene manifest to simulate a missing render handoff.
        next(files["scenes_dir"].glob("scene-b/render_manifest_template.json")).unlink()

        report = pipeline.build_project_render_qa_report(project, files["scenes_dir"], min_score=0.9)
        assert report.all_scenes_pass is False
        assert "SCENE_B" in report.failing_scene_ids
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_write_project_render_qa_report_creates_outputs():
    pipeline, project = _build_project()
    temp_dir = _workspace_temp_dir()
    try:
        files = pipeline.write_project_package(project, temp_dir / "project")
        _mark_render_manifests_rendered(files["scenes_dir"])
        report = pipeline.build_project_render_qa_report(project, files["scenes_dir"], min_score=0.9)

        qa_files = pipeline.write_project_render_qa_report(report, temp_dir / "qa")
        assert set(qa_files.keys()) == {"report_json", "review_markdown"}
        assert qa_files["report_json"].exists()
        assert qa_files["review_markdown"].exists()
        payload = json.loads(qa_files["report_json"].read_text(encoding="utf-8"))
        assert payload["project_name"] == "Project QA"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
