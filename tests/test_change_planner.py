"""Tests for incremental rerender planning."""

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


def _build_project(pipeline: CineFlowPipeline, description: str, project_name: str):
    return pipeline.build_project_package(
        scene_inputs=[
            ProjectSceneInput(
                scene_id="SCENE_A",
                description=description,
                num_shots=5,
                emotion_override="tense",
            )
        ],
        project_name=project_name,
        use_llm=False,
        min_score=0.8,
        max_attempts=2,
    )


def test_change_plan_is_empty_for_identical_projects():
    pipeline = CineFlowPipeline()
    previous = _build_project(
        pipeline,
        "A man in black coat faces a woman in red dress across a tavern table.",
        "Project Alpha",
    )
    current = _build_project(
        pipeline,
        "A man in black coat faces a woman in red dress across a tavern table.",
        "Project Alpha",
    )

    plan = pipeline.build_project_change_plan(previous, current)
    assert len(plan.rerender_queue) == 0
    assert any(summary.unchanged_shots == 5 for summary in plan.scene_summaries)


def test_change_plan_marks_changed_shots_for_rerender():
    pipeline = CineFlowPipeline()
    previous = _build_project(
        pipeline,
        "A man in black coat faces a woman in red dress across a tavern table.",
        "Project Alpha",
    )
    current = _build_project(
        pipeline,
        "A man in black coat faces a woman in red dress across a tavern table. He suddenly slams the glass down.",
        "Project Alpha",
    )

    plan = pipeline.build_project_change_plan(previous, current)
    assert len(plan.rerender_queue) > 0
    assert any(change.change_type.value in {"RERENDER", "NEW_RENDER"} for change in plan.shot_changes)


def test_write_project_change_plan_creates_files():
    pipeline = CineFlowPipeline()
    previous = _build_project(
        pipeline,
        "A man in black coat faces a woman in red dress across a tavern table.",
        "Project Alpha",
    )
    current = _build_project(
        pipeline,
        "A man in black coat faces a woman in red dress across a tavern table. He suddenly slams the glass down.",
        "Project Alpha",
    )
    plan = pipeline.build_project_change_plan(previous, current)

    temp_dir = _workspace_temp_dir()
    try:
        files = pipeline.write_project_change_plan(plan, temp_dir)
        assert set(files.keys()) == {"plan_json", "review_markdown", "rerender_queue"}
        assert files["plan_json"].exists()
        assert files["review_markdown"].exists()
        assert files["rerender_queue"].exists()
        assert json.loads(files["plan_json"].read_text(encoding="utf-8"))["current_project_name"] == "Project Alpha"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
