"""Tests for local visual review helpers and CLI-adjacent behavior."""

import shutil
import tempfile
from pathlib import Path

from PIL import Image

from autocineflow.local_visual_review import build_keyframe_download_batch_from_dir


def _workspace_temp_dir() -> Path:
    root = Path.cwd() / ".pytest_tmp"
    root.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(dir=root))


def test_build_keyframe_download_batch_from_dir_collects_images_only():
    temp_dir = _workspace_temp_dir()
    try:
        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir()
        Image.new("RGB", (64, 64), (10, 10, 10)).save(artifacts_dir / "shot_a.png")
        Image.new("RGB", (64, 64), (20, 20, 20)).save(artifacts_dir / "shot_b.jpg")
        (artifacts_dir / "ignore.txt").write_text("x", encoding="utf-8")

        batch = build_keyframe_download_batch_from_dir(artifacts_dir, source_id="SCENE_X")

        assert batch.source_id == "SCENE_X"
        assert len(batch.records) == 2
        assert {record.shot_id for record in batch.records} == {"shot_a", "shot_b"}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
