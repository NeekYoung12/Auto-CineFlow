"""Tests for production delivery packaging and exports."""

import json
import shutil
import tempfile
from pathlib import Path

from autocineflow.delivery import RenderPreset, slugify
from autocineflow.pipeline import CineFlowPipeline


def _workspace_temp_dir() -> Path:
    """Create a writable temp directory inside the repository workspace."""

    root = Path.cwd() / ".pytest_tmp"
    root.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(dir=root))


def _build_context():
    pipeline = CineFlowPipeline()
    context = pipeline.run(
        description=(
            "A man in black coat faces a woman in red dress across a tavern table. "
            '"We end this tonight," Alice said.'
        ),
        num_shots=5,
        scene_id="DELIVERY_SCENE",
        use_llm=False,
        emotion_override="tense",
    )
    return pipeline, context


def test_slugify_produces_stable_ascii_output():
    assert slugify("DELIVERY_SCENE SH001 Close Up") == "delivery-scene-sh001-close-up"
    assert slugify("###", default="fallback") == "fallback"


def test_build_storyboard_package_contains_stable_shot_ids():
    pipeline, context = _build_context()
    package = pipeline.build_storyboard_package(
        context,
        project_name="Delivery Test",
        render_preset=RenderPreset(fps=24),
        generated_at="2026-03-22T00:00:00+00:00",
    )

    assert package.project_name == "Delivery Test"
    assert package.generated_at == "2026-03-22T00:00:00+00:00"
    assert package.shots[0].shot_id == "DELIVERY_SCENE_SH001"
    assert package.shots[-1].shot_id == "DELIVERY_SCENE_SH005"
    assert package.shots[0].timeline_in == "00:00:00:00"
    assert package.shots[1].timeline_in == package.shots[0].timeline_out
    assert package.character_bible[0].char_id == "CHAR_A"
    assert package.character_bible[0].default_seed >= 420000
    assert package.shots[0].render_seed >= 420000
    assert package.video_segments[0].generation_duration_seconds in {6.0, 10.0}
    assert all(segment.generation_duration_seconds in {6.0, 10.0} for segment in package.video_segments)
    assert package.video_segments[0].segment_id.endswith("SEG01")
    assert package.shots[0].reference_shot_id == ""
    assert package.shots[1].reference_shot_id == "DELIVERY_SCENE_SH001"
    assert all(shot.duration_seconds > 0 for shot in package.shots)
    assert all(shot.frame_count == round(shot.duration_seconds * 24) for shot in package.shots)
    assert package.render_queue[0].metadata["scene_id"] == "DELIVERY_SCENE"
    assert package.render_queue[1].metadata["reference_shot_id"] == "DELIVERY_SCENE_SH001"


def test_export_package_to_csv_and_json():
    pipeline, context = _build_context()
    package = pipeline.build_storyboard_package(context, project_name="Delivery Test")

    manifest_json = pipeline.storyboard_package_json(package, indent=2)
    render_queue_json = pipeline.render_queue_json(package, indent=2)
    character_bible_json = pipeline.character_bible_json(package, indent=2)
    video_plan_json = pipeline.video_plan_json(package, indent=2)
    automatic1111_json = pipeline.automatic1111_bundle_json(package, indent=2)
    comfyui_json = pipeline.comfyui_bundle_json(package, indent=2)
    csv_text = pipeline.shotlist_csv(package)
    edl = pipeline.edl_text(package)
    review_markdown = pipeline.storyboard_review_markdown(package)

    manifest = json.loads(manifest_json)
    render_queue = json.loads(render_queue_json)
    character_bible = json.loads(character_bible_json)
    video_plan = json.loads(video_plan_json)
    automatic1111_bundle = json.loads(automatic1111_json)
    comfyui_bundle = json.loads(comfyui_json)

    assert manifest["scene_id"] == "DELIVERY_SCENE"
    assert len(manifest["shots"]) == 5
    assert len(render_queue) == 5
    assert len(character_bible) == 2
    assert len(video_plan) == 5
    assert len(automatic1111_bundle) == 5
    assert len(comfyui_bundle) == 5
    assert automatic1111_bundle[0]["seed"] == manifest["shots"][0]["render_seed"]
    assert comfyui_bundle[0]["workflow"]["seed"] == manifest["shots"][0]["render_seed"]
    assert "shot_id,shot_slug,shot_index,beat_type,shot_type" in csv_text
    assert "DELIVERY_SCENE_SH001" in csv_text
    assert "render_seed" in csv_text
    assert "TITLE: Delivery Test DELIVERY_SCENE" in edl
    assert "* FROM CLIP NAME: DELIVERY_SCENE_SH001" in edl
    assert "# Delivery Test" in review_markdown
    assert "### DELIVERY_SCENE_SH001" in review_markdown
    assert "**Prompt**" in review_markdown


def test_write_delivery_package_creates_output_files():
    pipeline, context = _build_context()
    package = pipeline.build_storyboard_package(context, project_name="Delivery Test")
    temp_dir = _workspace_temp_dir()

    try:
        written = pipeline.write_delivery_package(package, temp_dir)
        assert set(written.keys()) == {
            "manifest",
            "shotlist",
            "render_queue",
            "character_bible",
            "video_plan",
            "edl",
            "review_markdown",
            "automatic1111",
            "comfyui",
            "runninghub_faceid",
            "volcengine_seedream",
            "render_manifest_template",
            "assembly_json",
            "concat_manifest",
            "concat_script",
        }
        assert written["manifest"].exists()
        assert written["shotlist"].exists()
        assert written["render_queue"].exists()
        assert written["character_bible"].exists()
        assert written["video_plan"].exists()
        assert written["edl"].exists()
        assert written["review_markdown"].exists()
        assert written["automatic1111"].exists()
        assert written["comfyui"].exists()
        assert written["runninghub_faceid"].exists()
        assert written["volcengine_seedream"].exists()
        assert written["render_manifest_template"].exists()
        assert written["assembly_json"].exists()
        assert written["concat_manifest"].exists()
        assert written["concat_script"].exists()
        assert json.loads(written["manifest"].read_text(encoding="utf-8"))["scene_id"] == "DELIVERY_SCENE"
        assert len(json.loads(written["character_bible"].read_text(encoding="utf-8"))) == 2
        assert len(json.loads(written["video_plan"].read_text(encoding="utf-8"))) == 5
        assert json.loads(written["assembly_json"].read_text(encoding="utf-8"))["scene_id"] == "DELIVERY_SCENE"
        assert len(json.loads(written["automatic1111"].read_text(encoding="utf-8"))) == 5
        assert len(json.loads(written["comfyui"].read_text(encoding="utf-8"))) == 5
        assert len(json.loads(written["runninghub_faceid"].read_text(encoding="utf-8"))) == 5
        assert len(json.loads(written["volcengine_seedream"].read_text(encoding="utf-8"))) == 5
        assert len(json.loads(written["render_manifest_template"].read_text(encoding="utf-8"))) == 5
        assert "# Delivery Test" in written["review_markdown"].read_text(encoding="utf-8")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
