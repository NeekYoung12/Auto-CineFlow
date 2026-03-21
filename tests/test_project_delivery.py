"""Tests for project-level packaging and quality gating."""

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


def test_quality_report_returns_score_and_gate():
    pipeline = CineFlowPipeline()
    context = pipeline.run(
        description="A man in black coat faces a woman in red dress across a tavern table.",
        num_shots=5,
        use_llm=False,
        emotion_override="tense",
    )

    report = pipeline.quality_report(context, min_score=0.8)
    assert 0.0 <= report["score"] <= 1.0
    assert report["passes_gate"] is True
    assert report["metrics"]["axis_consistency"] == 1.0


def test_run_with_quality_gate_returns_best_attempt():
    pipeline = CineFlowPipeline()
    context, report = pipeline.run_with_quality_gate(
        description="A man in black coat faces a woman in red dress across a tavern table.",
        num_shots=5,
        use_llm=False,
        emotion_override="tense",
        min_score=0.8,
        max_attempts=2,
    )

    assert context.shot_blocks
    assert report["passes_gate"] is True
    assert report["attempt"] == 1
    assert report["strategy"] == "requested"


def test_build_project_package_aggregates_multiple_scenes():
    pipeline = CineFlowPipeline()
    project_package = pipeline.build_project_package(
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
        project_name="Batch Project",
        use_llm=False,
        min_score=0.8,
        max_attempts=2,
    )

    assert project_package.scene_count == 2
    assert len(project_package.scenes) == 2
    assert project_package.all_scenes_ready is True
    assert project_package.average_quality_score >= 0.8
    assert project_package.scene_summaries[0].passes_gate is True


def test_write_project_package_creates_manifest_and_scene_dirs():
    pipeline = CineFlowPipeline()
    project_package = pipeline.build_project_package(
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
        project_name="Batch Project",
        use_llm=False,
        min_score=0.8,
        max_attempts=2,
    )
    temp_dir = _workspace_temp_dir()

    try:
        written = pipeline.write_project_package(project_package, temp_dir)
        assert set(written.keys()) == {"manifest", "shotlist", "review_markdown", "scenes_dir"}
        assert written["manifest"].exists()
        assert written["shotlist"].exists()
        assert written["review_markdown"].exists()
        assert written["scenes_dir"].exists()
        manifest = json.loads(written["manifest"].read_text(encoding="utf-8"))
        assert manifest["scene_count"] == 2
        assert (written["scenes_dir"] / "scene-a" / "storyboard_package.json").exists()
        assert (written["scenes_dir"] / "scene-b" / "storyboard_package.json").exists()
        assert (written["scenes_dir"] / "scene-a" / "render_manifest_template.json").exists()
        assert (written["scenes_dir"] / "scene-b" / "render_manifest_template.json").exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
