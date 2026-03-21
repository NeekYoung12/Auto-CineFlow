"""Tests for project dashboard aggregation."""

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


def _build_project(pipeline: CineFlowPipeline):
    return pipeline.build_project_package(
        scene_inputs=[
            ProjectSceneInput(
                scene_id="SCENE_A",
                description="A man in black coat faces a woman in red dress across a tavern table.",
                num_shots=5,
                emotion_override="tense",
            )
        ],
        project_name="Dashboard Project",
        use_llm=False,
        min_score=0.8,
        max_attempts=2,
    )


def _mark_render_manifest_rendered(scenes_dir: Path) -> None:
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


def test_project_dashboard_combines_storyboard_render_and_execution_data():
    pipeline = CineFlowPipeline()
    previous = _build_project(pipeline)
    current = _build_project(pipeline)

    temp_dir = _workspace_temp_dir()
    try:
        previous_files = pipeline.write_project_package(previous, temp_dir / "previous")
        _mark_render_manifest_rendered(previous_files["scenes_dir"])
        render_report = pipeline.build_project_render_qa_report(previous, previous_files["scenes_dir"], min_score=0.9)
        execution_plan = pipeline.build_project_execution_plan(previous, current, previous_files["scenes_dir"])
        dashboard = pipeline.build_project_dashboard(current, render_qa=render_report, execution_plan=execution_plan)

        assert dashboard.scene_count == 1
        assert dashboard.storyboard_ready_count == 1
        assert dashboard.render_ready_count == 1
        assert dashboard.total_reuse_count == 5
        assert dashboard.total_rerender_count == 0
        assert dashboard.scene_rows[0].overall_status == "ready"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_write_project_dashboard_creates_outputs():
    pipeline = CineFlowPipeline()
    project = _build_project(pipeline)
    dashboard = pipeline.build_project_dashboard(project)
    temp_dir = _workspace_temp_dir()
    try:
        files = pipeline.write_project_dashboard(dashboard, temp_dir)
        assert set(files.keys()) == {"dashboard_json", "dashboard_markdown"}
        assert files["dashboard_json"].exists()
        assert files["dashboard_markdown"].exists()
        payload = json.loads(files["dashboard_json"].read_text(encoding="utf-8"))
        assert payload["project_name"] == "Dashboard Project"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
