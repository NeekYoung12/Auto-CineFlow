"""Tests for render manifest generation and render QA."""

import json
import shutil
import tempfile
from pathlib import Path

from autocineflow.pipeline import CineFlowPipeline
from autocineflow.render_qa import build_render_manifest_template


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
        scene_id="QA_SCENE",
        use_llm=False,
        emotion_override="tense",
    )
    package = pipeline.build_storyboard_package(context, project_name="QA Project")
    return pipeline, package


def test_render_manifest_template_matches_package_shots():
    pipeline, package = _build_package()
    manifest = build_render_manifest_template(package)
    manifest_json = pipeline.render_manifest_template_json(package, indent=2)
    payload = json.loads(manifest_json)

    assert len(manifest) == len(package.shots) == 5
    assert payload[0]["shot_id"] == package.shots[0].shot_id
    assert payload[0]["expected_seed"] == package.shots[0].render_seed


def test_render_qa_passes_with_matching_manifest():
    pipeline, package = _build_package()
    manifest = build_render_manifest_template(package)
    for item in manifest:
        item.status = "rendered"
        item.output_path = f"renders/{item.expected_filename}"
        item.actual_seed = item.expected_seed
        item.actual_width = item.expected_width
        item.actual_height = item.expected_height
        item.actual_prompt_hash = item.expected_prompt_hash

    report = pipeline.render_qa_report(package, manifest, min_score=0.9)
    assert report.passes_gate is True
    assert report.score == 1.0
    assert all(not shot.issues for shot in report.shot_results)


def test_render_qa_fails_when_seed_and_dimensions_do_not_match():
    pipeline, package = _build_package()
    manifest = build_render_manifest_template(package)
    for item in manifest:
        item.status = "rendered"
        item.output_path = f"renders/{item.expected_filename}"
        item.actual_seed = item.expected_seed
        item.actual_width = item.expected_width
        item.actual_height = item.expected_height
        item.actual_prompt_hash = item.expected_prompt_hash

    manifest[2].actual_seed += 1
    manifest[2].actual_width -= 64

    report = pipeline.render_qa_report(package, manifest, min_score=0.9)
    assert report.passes_gate is False
    assert "seed_mismatch" in report.shot_results[2].issues
    assert "dimension_mismatch" in report.shot_results[2].issues


def test_write_delivery_and_render_qa_outputs():
    pipeline, package = _build_package()
    temp_dir = _workspace_temp_dir()
    try:
        delivery_files = pipeline.write_delivery_package(package, temp_dir / "delivery")
        assert "render_manifest_template" in delivery_files
        manifest = build_render_manifest_template(package)
        for item in manifest:
            item.status = "rendered"
            item.output_path = f"renders/{item.expected_filename}"
            item.actual_seed = item.expected_seed
            item.actual_width = item.expected_width
            item.actual_height = item.expected_height
            item.actual_prompt_hash = item.expected_prompt_hash

        report = pipeline.render_qa_report(package, manifest, min_score=0.9)
        qa_files = pipeline.write_render_qa_report(report, temp_dir / "qa")
        assert qa_files["report_json"].exists()
        assert qa_files["review_markdown"].exists()
        assert json.loads(qa_files["report_json"].read_text(encoding="utf-8"))["passes_gate"] is True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
