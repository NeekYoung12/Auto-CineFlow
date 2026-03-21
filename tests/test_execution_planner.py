"""Tests for reuse/rerender execution planning."""

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


def _build_project(pipeline: CineFlowPipeline, description: str):
    return pipeline.build_project_package(
        scene_inputs=[
            ProjectSceneInput(
                scene_id="SCENE_A",
                description=description,
                num_shots=5,
                emotion_override="tense",
            )
        ],
        project_name="Execution Project",
        use_llm=False,
        min_score=0.8,
        max_attempts=2,
    )


def _mark_scene_manifests_rendered(scenes_dir: Path) -> None:
    """Fill render manifests with successful render metadata."""

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


def test_execution_plan_reuses_valid_previous_renders():
    pipeline = CineFlowPipeline()
    previous = _build_project(
        pipeline,
        "A man in black coat faces a woman in red dress across a tavern table.",
    )
    current = _build_project(
        pipeline,
        "A man in black coat faces a woman in red dress across a tavern table.",
    )

    temp_dir = _workspace_temp_dir()
    try:
        previous_files = pipeline.write_project_package(previous, temp_dir / "previous")
        _mark_scene_manifests_rendered(previous_files["scenes_dir"])

        plan = pipeline.build_project_execution_plan(previous, current, previous_files["scenes_dir"])
        assert len(plan.reuse_manifest) == 5
        assert len(plan.rerender_queue) == 0
        assert all(item["scene_id"] == "SCENE_A" for item in plan.reuse_manifest)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_execution_plan_rerenders_changed_shots():
    pipeline = CineFlowPipeline()
    previous = _build_project(
        pipeline,
        "A man in black coat faces a woman in red dress across a tavern table.",
    )
    current = _build_project(
        pipeline,
        "A man in black coat faces a woman in red dress across a tavern table. He suddenly slams the glass down.",
    )

    temp_dir = _workspace_temp_dir()
    try:
        previous_files = pipeline.write_project_package(previous, temp_dir / "previous")
        _mark_scene_manifests_rendered(previous_files["scenes_dir"])

        plan = pipeline.build_project_execution_plan(previous, current, previous_files["scenes_dir"])
        assert len(plan.rerender_queue) > 0
        assert any(decision.decision.value == "RERENDER" for decision in plan.decisions)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_write_project_execution_plan_creates_outputs():
    pipeline = CineFlowPipeline()
    previous = _build_project(
        pipeline,
        "A man in black coat faces a woman in red dress across a tavern table.",
    )
    current = _build_project(
        pipeline,
        "A man in black coat faces a woman in red dress across a tavern table.",
    )

    temp_dir = _workspace_temp_dir()
    try:
        previous_files = pipeline.write_project_package(previous, temp_dir / "previous")
        _mark_scene_manifests_rendered(previous_files["scenes_dir"])
        plan = pipeline.build_project_execution_plan(previous, current, previous_files["scenes_dir"])

        files = pipeline.write_project_execution_plan(plan, temp_dir / "execution")
        assert set(files.keys()) == {"plan_json", "review_markdown", "reuse_manifest", "rerender_queue"}
        assert files["plan_json"].exists()
        assert files["review_markdown"].exists()
        assert files["reuse_manifest"].exists()
        assert files["rerender_queue"].exists()
        payload = json.loads(files["plan_json"].read_text(encoding="utf-8"))
        assert payload["current_project_name"] == "Execution Project"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
