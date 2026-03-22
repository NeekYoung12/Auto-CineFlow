"""Tests for local FFmpeg-based video enhancement."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import autocineflow.video_enhance as video_enhance_module
from autocineflow.result_ingest import ArtifactDownloadBatch, ArtifactDownloadRecord
from autocineflow.video_enhance import build_video_enhance_plan, enhance_videos


def _workspace_temp_dir() -> Path:
    root = Path.cwd() / ".pytest_tmp"
    root.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(dir=root))


def test_build_video_enhance_plan_and_run(monkeypatch):
    temp_dir = _workspace_temp_dir()
    try:
        input_video = temp_dir / "SHOT_01.mp4"
        input_video.write_bytes(b"fake-video")
        downloads = ArtifactDownloadBatch(
            source_id="SCENE_01",
            records=[
                ArtifactDownloadRecord(
                    job_id="job-1",
                    shot_id="SHOT_01",
                    url="https://example.invalid/shot.mp4",
                    output_path=str(input_video),
                    downloaded=True,
                )
            ],
        )

        monkeypatch.setattr(video_enhance_module.shutil, "which", lambda name: f"C:/tools/{name}.exe")

        def fake_run(command, capture_output, text, check=False):
            if "ffprobe" in command[0]:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        {
                            "streams": [{"width": 1024, "height": 576, "codec_name": "h264"}],
                            "format": {"duration": "6.0"},
                        }
                    ),
                    stderr="",
                )
            output_path = Path(command[-1])
            output_path.write_bytes(b"enhanced-video")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr(video_enhance_module.subprocess, "run", fake_run)

        plan = build_video_enhance_plan(downloads, temp_dir / "enhanced")
        assert plan[0].target_width == 1920
        assert "hqdn3d" in plan[0].filter_chain

        report = enhance_videos(downloads, temp_dir / "enhanced")
        assert report.results[0].enhanced is True
        assert Path(report.results[0].output_path).exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
