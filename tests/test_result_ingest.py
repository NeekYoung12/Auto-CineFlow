"""Tests for artifact download and render-manifest syncing."""

import json
import shutil
import tempfile
from pathlib import Path

import httpx

from autocineflow.pipeline import CineFlowPipeline
from autocineflow.submission import SubmissionBackend, SubmissionProvider, SubmissionTarget


def _workspace_temp_dir() -> Path:
    """Create a writable temp directory inside the repository workspace."""

    root = Path.cwd() / ".pytest_tmp"
    root.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(dir=root))


def _build_submission_batch():
    pipeline = CineFlowPipeline()
    context = pipeline.run(
        description="A man in black coat faces a woman in red dress across a tavern table.",
        num_shots=5,
        scene_id="INGEST_SCENE",
        use_llm=False,
        emotion_override="tense",
    )
    package = pipeline.build_storyboard_package(context, project_name="Ingest Project")
    jobs = pipeline.build_submission_jobs_from_package(package, provider=SubmissionProvider.MINIMAX_IMAGE)
    batch = pipeline.submit_jobs(
        jobs[:1],
        SubmissionTarget(backend=SubmissionBackend.DRY_RUN),
        source_type="package",
        source_id=package.scene_id,
    )
    batch.records[0].message = "https://example.invalid/generated.png"
    return pipeline, package, batch


def test_download_submission_artifacts_and_sync_manifest(monkeypatch):
    pipeline, package, batch = _build_submission_batch()

    class DummyResponse:
        content = b"fake-image-bytes"

        def raise_for_status(self):
            return None

    def fake_get(url, timeout):
        assert url == "https://example.invalid/generated.png"
        return DummyResponse()

    monkeypatch.setattr(httpx, "get", fake_get)

    temp_dir = _workspace_temp_dir()
    try:
        delivery_files = pipeline.write_delivery_package(package, temp_dir / "delivery")
        downloads = pipeline.download_submission_artifacts(batch, temp_dir / "artifacts")
        assert downloads.records[0].downloaded is True
        assert Path(downloads.records[0].output_path).exists()

        updated_entries = pipeline.update_render_manifest_from_downloads(
            delivery_files["render_manifest_template"],
            downloads,
        )
        assert updated_entries[0].status == "rendered"
        assert updated_entries[0].output_path.endswith(".png")
        assert updated_entries[0].actual_seed == updated_entries[0].expected_seed
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_write_artifact_download_batch_creates_json(monkeypatch):
    pipeline, package, batch = _build_submission_batch()

    class DummyResponse:
        content = b"fake-image-bytes"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(httpx, "get", lambda url, timeout: DummyResponse())

    temp_dir = _workspace_temp_dir()
    try:
        downloads = pipeline.download_submission_artifacts(batch, temp_dir / "artifacts")
        files = pipeline.write_artifact_download_batch(downloads, temp_dir / "downloads")
        assert set(files.keys()) == {"batch_json"}
        assert files["batch_json"].exists()
        payload = json.loads(files["batch_json"].read_text(encoding="utf-8"))
        assert payload["source_id"] == "INGEST_SCENE"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_download_submission_artifacts_skips_existing_file(monkeypatch):
    pipeline, package, batch = _build_submission_batch()
    temp_dir = _workspace_temp_dir()
    try:
        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        existing = artifacts_dir / "INGEST_SCENE_SH001.png"
        existing.write_bytes(b"already-downloaded")

        def fail_get(*args, **kwargs):
            raise AssertionError("httpx.get should not be called when skip_existing=True")

        monkeypatch.setattr(httpx, "get", fail_get)
        downloads = pipeline.download_submission_artifacts(batch, artifacts_dir, skip_existing=True)
        assert downloads.records[0].downloaded is True
        assert downloads.records[0].output_path == str(existing)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_download_minimax_video_artifact_via_task_polling(monkeypatch):
    pipeline, package, batch = _build_submission_batch()
    batch.records[0].provider = SubmissionProvider.MINIMAX_VIDEO
    batch.records[0].backend_job_id = "task-123"
    batch.records[0].message = ""

    class DummyResponse:
        def __init__(self, payload=None, content=None):
            self._payload = payload or {}
            self.content = content or b""

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    calls = {"query": 0}

    def fake_get(url, headers=None, params=None, timeout=None):
        if url == "https://api.example.invalid/v1/query/video_generation":
            calls["query"] += 1
            return DummyResponse({"status": "Success", "file_id": "file-123"})
        if url == "https://api.example.invalid/v1/files/retrieve":
            return DummyResponse({"file": {"download_url": "https://example.invalid/video.mp4"}})
        if url == "https://example.invalid/video.mp4":
            return DummyResponse(content=b"fake-video-bytes")
        raise AssertionError(url)

    monkeypatch.setattr(httpx, "get", fake_get)

    temp_dir = _workspace_temp_dir()
    try:
        config_path = temp_dir / "conf"
        config_path.write_text(
            "\n".join(
                [
                    "Image or Video Generation:",
                    "API_KEY=sk-media",
                    "MINIMAX_BASE_URL=https://api.example.invalid/v1",
                ]
            ),
            encoding="utf-8",
        )
        downloads = pipeline.download_submission_artifacts(
            batch,
            temp_dir / "artifacts",
            config_path=str(config_path),
            timeout_seconds=30.0,
            poll_interval_seconds=0.01,
        )
        assert downloads.records[0].downloaded is True
        assert downloads.records[0].output_path.endswith(".mp4")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
