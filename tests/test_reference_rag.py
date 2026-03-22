"""Tests for local reference retrieval and consistency packaging."""

import json
import shutil
import tempfile
from pathlib import Path

from PIL import Image

from autocineflow.pipeline import CineFlowPipeline
from autocineflow.reference_rag import build_reference_library, retrieve_character_assets, retrieve_scene_assets


def _workspace_temp_dir() -> Path:
    root = Path.cwd() / ".pytest_tmp"
    root.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(dir=root))


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path)


def _build_reference_root(base_dir: Path) -> Path:
    reference_root = base_dir / "reference_library"

    character_dir = reference_root / "cyberpunk_demo" / "characters" / "char-detective"
    base_image = character_dir / "char-detective_base.png"
    close_up = character_dir / "char-detective_close_up.png"
    wide_shot = character_dir / "char-detective_wide_shot.png"
    _write_image(base_image, (48, 62, 82))
    _write_image(close_up, (120, 110, 108))
    _write_image(wide_shot, (35, 44, 67))
    (reference_root / "cyberpunk_demo" / "characters" / "char-detective_manifest.json").write_text(
        json.dumps(
            {
                "character_id": "char-detective",
                "base_image_path": str(base_image),
                "standardized_views": {
                    "close_up": str(close_up),
                    "wide_shot": str(wide_shot),
                },
                "prompt": "rain-soaked detective, black trench coat, scar on cheek, cinematic noir",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    scene_dir = reference_root / "scene_neon"
    scene_artifact = scene_dir / "artifacts" / "scene_frame.png"
    _write_image(scene_artifact, (18, 28, 88))
    (scene_dir / "storyboard_package.json").write_text(
        json.dumps(
            {
                "project_name": "Reference Project",
                "scene_id": "SCENE_NEON",
                "scene_location": "neon alley",
                "shots": [
                    {"prompt": "rainy neon alley, blue and magenta practical lights"},
                ],
                "scene_tags": ["rain", "night", "neon"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return reference_root


def test_reference_library_scans_character_and_scene_assets():
    temp_dir = _workspace_temp_dir()
    try:
        reference_root = _build_reference_root(temp_dir)
        library = build_reference_library([reference_root])

        assert library.roots == [str(reference_root)]
        assert len(library.character_assets) == 1
        assert len(library.scene_assets) == 1
        assert library.character_assets[0].faceid_profile is not None
        assert library.scene_assets[0].palette_signature
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_reference_library_scans_scene_artifacts_from_delivery_layout():
    temp_dir = _workspace_temp_dir()
    try:
        root = temp_dir / "reference_delivery_layout"
        delivery_dir = root / "scene_run" / "delivery"
        artifacts_dir = root / "scene_run" / "artifacts"
        delivery_dir.mkdir(parents=True, exist_ok=True)
        _write_image(artifacts_dir / "scene_frame.png", (12, 44, 92))
        (delivery_dir / "storyboard_package.json").write_text(
            json.dumps(
                {
                    "project_name": "Delivery Layout",
                    "scene_id": "DELIVERY_LAYOUT_SCENE",
                    "scene_location": "warehouse",
                    "shots": [{"prompt": "industrial warehouse interior, sodium vapor haze"}],
                    "scene_tags": ["industrial", "night"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        library = build_reference_library([root])

        assert len(library.scene_assets) == 1
        assert library.scene_assets[0].preview_paths
        assert "artifacts" in library.scene_assets[0].preview_paths[0]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_reference_retrieval_and_consistency_package_enrich_storyboard():
    temp_dir = _workspace_temp_dir()
    try:
        reference_root = _build_reference_root(temp_dir)
        library = build_reference_library([reference_root])
        characters = retrieve_character_assets(library, "rain-soaked detective in black trench coat", top_k=1)
        scenes = retrieve_scene_assets(library, "night neon alley rain", top_k=1)

        assert characters[0].character_id == "char-detective"
        assert scenes[0].scene_id == "SCENE_NEON"

        pipeline = CineFlowPipeline()
        context = pipeline.run(
            description="A detective in a black trench coat confronts an informant in a neon alley at night.",
            num_shots=5,
            scene_id="CONSISTENCY_SCENE",
            use_llm=False,
            emotion_override="tense",
        )
        package = pipeline.build_storyboard_package(
            context,
            project_name="Consistency Project",
            reference_roots=[reference_root],
        )

        assert package.consistency_package is not None
        assert package.character_bible[0].identity_prompt
        assert package.character_bible[0].reference_image_paths
        assert package.character_bible[0].faceid_profile_id
        assert package.render_queue[0].metadata["character_reference_images"]
        assert package.render_queue[0].metadata["scene_reference_images"]
        assert package.shots[0].identity_prompt_suffix

        output_files = pipeline.write_delivery_package(package, temp_dir / "delivery")
        assert "consistency_package" in output_files
        assert "consistency_review" in output_files
        assert output_files["consistency_package"].exists()
        assert output_files["consistency_review"].exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
