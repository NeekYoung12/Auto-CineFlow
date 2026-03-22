"""Tests for render submission job building and backends."""

import json
import shutil
import tempfile
from pathlib import Path

import httpx

from autocineflow.pipeline import CineFlowPipeline
from autocineflow.project_delivery import ProjectSceneInput
from autocineflow.submission import SubmissionBackend, SubmissionProvider, SubmissionTarget


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
        scene_id="SUBMIT_SCENE",
        use_llm=False,
        emotion_override="tense",
    )
    package = pipeline.build_storyboard_package(context, project_name="Submission Project")
    return pipeline, package


def _build_execution_plan():
    pipeline = CineFlowPipeline()
    previous = pipeline.build_project_package(
        scene_inputs=[
            ProjectSceneInput(
                scene_id="SCENE_A",
                description="A man in black coat faces a woman in red dress across a tavern table.",
                num_shots=5,
                emotion_override="tense",
            )
        ],
        project_name="Submission Project",
        use_llm=False,
        min_score=0.8,
        max_attempts=2,
    )
    current = pipeline.build_project_package(
        scene_inputs=[
            ProjectSceneInput(
                scene_id="SCENE_A",
                description="A man in black coat faces a woman in red dress across a tavern table. He suddenly slams the glass down.",
                num_shots=5,
                emotion_override="tense",
            )
        ],
        project_name="Submission Project",
        use_llm=False,
        min_score=0.8,
        max_attempts=2,
    )
    temp_dir = _workspace_temp_dir()
    previous_files = pipeline.write_project_package(previous, temp_dir / "previous")
    return pipeline, temp_dir, previous, current, pipeline.build_project_execution_plan(previous, current, previous_files["scenes_dir"])


def test_build_submission_jobs_from_package_for_multiple_providers():
    pipeline, package = _build_package()
    generic_jobs = pipeline.build_submission_jobs_from_package(package)
    a1111_jobs = pipeline.build_submission_jobs_from_package(package, provider=SubmissionProvider.AUTOMATIC1111)
    comfy_jobs = pipeline.build_submission_jobs_from_package(package, provider=SubmissionProvider.COMFYUI)
    minimax_jobs = pipeline.build_submission_jobs_from_package(package, provider=SubmissionProvider.MINIMAX_IMAGE)
    minimax_video_jobs = pipeline.build_submission_jobs_from_package(package, provider=SubmissionProvider.MINIMAX_VIDEO)
    runninghub_jobs = pipeline.build_submission_jobs_from_package(package, provider=SubmissionProvider.RUNNINGHUB_FACEID)
    volcengine_jobs = pipeline.build_submission_jobs_from_package(package, provider=SubmissionProvider.VOLCENGINE_SEEDREAM)

    assert len(generic_jobs) == 5
    assert len(a1111_jobs) == 5
    assert len(comfy_jobs) == 5
    assert len(minimax_jobs) == 5
    assert len(minimax_video_jobs) == 5
    assert len(runninghub_jobs) == 5
    assert len(volcengine_jobs) == 5
    assert a1111_jobs[0].payload["seed"] == package.shots[0].render_seed
    assert comfy_jobs[0].payload["workflow"]["seed"] == package.shots[0].render_seed
    assert minimax_jobs[0].payload["model"] == "image-01"
    assert minimax_jobs[0].payload["aspect_ratio"] == "16:9"
    assert minimax_jobs[0].payload["seed"] == package.shots[0].render_seed
    assert "metadata" not in minimax_jobs[0].payload
    assert minimax_video_jobs[0].payload["model"] == "MiniMax-Hailuo-02"
    assert minimax_video_jobs[0].payload["duration"] == 10
    assert minimax_video_jobs[0].payload["resolution"] == "768P"
    assert len(minimax_video_jobs[0].payload["prompt"]) <= len(package.video_segments[0].prompt)
    assert runninghub_jobs[0].payload["workflow_family"] == "runninghub_comfyui_faceid"
    assert volcengine_jobs[0].payload["model"] == "doubao-seedream-4-0-250828"
    assert volcengine_jobs[0].payload["response_format"] == "url"


def test_submit_jobs_to_filesystem_queue():
    pipeline, package = _build_package()
    jobs = pipeline.build_submission_jobs_from_package(package)
    temp_dir = _workspace_temp_dir()
    try:
        batch = pipeline.submit_jobs(
            jobs,
            SubmissionTarget(backend=SubmissionBackend.FILESYSTEM, spool_dir=str(temp_dir / "spool")),
            source_type="package",
            source_id=package.scene_id,
        )
        assert batch.job_count == 5
        assert all(record.status == "queued" for record in batch.records)
        queued_files = list((temp_dir / "spool" / "queued").glob("*.json"))
        assert len(queued_files) == 5
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_submit_jobs_from_execution_plan_dry_run():
    pipeline, temp_dir, previous, current, plan = _build_execution_plan()
    try:
        jobs = pipeline.build_submission_jobs_from_execution_plan(plan)
        batch = pipeline.submit_jobs(
            jobs,
            SubmissionTarget(backend=SubmissionBackend.DRY_RUN),
            source_type="execution_plan",
            source_id=plan.current_project_name,
        )
        assert batch.job_count == len(plan.ordered_rerender_queue)
        assert all(record.status == "dry_run" for record in batch.records)
        payload = json.loads(pipeline.submission_batch_json(batch, indent=2))
        assert payload["source_type"] == "execution_plan"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_submit_jobs_to_minimax_api_backend(monkeypatch):
    pipeline, package = _build_package()
    jobs = pipeline.build_submission_jobs_from_package(package, provider=SubmissionProvider.MINIMAX_IMAGE)

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "id": "minimax-job-1",
                "data": {"image_urls": ["https://example.invalid/image.png"]},
                "base_resp": {"status_code": 0, "status_msg": "ok"},
            }

        @property
        def text(self):
            return '{"ok":true}'

    def fake_post(url, headers, json, timeout):
        assert url == "https://api.example.invalid/v1/image_generation"
        assert headers["Authorization"] == "Bearer sk-media"
        assert json["model"] == "image-01"
        assert "metadata" not in json
        return DummyResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

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
        batch = pipeline.submit_jobs(
            jobs[:1],
            SubmissionTarget(
                backend=SubmissionBackend.MINIMAX_API,
                config_path=str(config_path),
            ),
            source_type="package",
            source_id=package.scene_id,
        )
        assert batch.records[0].status == "submitted"
        assert batch.records[0].backend_job_id == "minimax-job-1"
        assert batch.records[0].provider_status_code == 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_build_minimax_image_jobs_include_subject_reference_for_remote_urls():
    pipeline, package = _build_package()
    first_job = package.render_queue[0].model_copy(
        update={
            "metadata": {
                **package.render_queue[0].metadata,
                "character_reference_images": ["https://example.invalid/ref-face.png"],
            }
        }
    )
    updated_package = package.model_copy(
        update={
            "render_queue": [first_job, *package.render_queue[1:]],
        }
    )

    jobs = pipeline.build_submission_jobs_from_package(updated_package, provider=SubmissionProvider.MINIMAX_IMAGE)

    assert jobs[0].payload["subject_reference"][0]["image_file"] == "https://example.invalid/ref-face.png"
    assert jobs[0].payload["subject_reference"][0]["type"] == "character"


def test_submit_video_jobs_to_minimax_api_backend(monkeypatch):
    pipeline, package = _build_package()
    jobs = pipeline.build_submission_jobs_from_package(package, provider=SubmissionProvider.MINIMAX_VIDEO)

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"task_id": "minimax-video-task-1", "base_resp": {"status_code": 0, "status_msg": "ok"}}

        @property
        def text(self):
            return '{"ok":true}'

    def fake_post(url, headers, json, timeout):
        assert url == "https://api.example.invalid/v1/video_generation"
        assert json["model"] == "MiniMax-Hailuo-02"
        return DummyResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

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
        batch = pipeline.submit_jobs(
            jobs[:1],
            SubmissionTarget(
                backend=SubmissionBackend.MINIMAX_API,
                config_path=str(config_path),
            ),
            source_type="package",
            source_id=package.scene_id,
        )
        assert batch.records[0].status == "submitted"
        assert batch.records[0].backend_job_id == "minimax-video-task-1"
        assert batch.records[0].provider_status_message == "ok"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_submit_seedream_jobs_to_volcengine_ark_backend(monkeypatch):
    pipeline, package = _build_package()
    jobs = pipeline.build_submission_jobs_from_package(package, provider=SubmissionProvider.VOLCENGINE_SEEDREAM)

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "created": 1764041608,
                "data": [{"url": "https://example.invalid/seedream.png"}],
            }

        @property
        def text(self):
            return '{"ok":true}'

    def fake_post(url, headers, json, timeout):
        assert url == "https://las.example.invalid/api/v3/images/generations"
        assert headers["Authorization"] == "Bearer ark-key"
        assert json["model"] == "doubao-seedream-4-0-250828"
        assert json["response_format"] == "url"
        return DummyResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    temp_dir = _workspace_temp_dir()
    try:
        config_path = temp_dir / "conf"
        config_path.write_text(
            "\n".join(
                [
                    "volcengine:",
                    "ARK_API_KEY=ark-key",
                    "VOLCENGINE_ARK_BASE_URL=https://las.example.invalid",
                ]
            ),
            encoding="utf-8",
        )
        batch = pipeline.submit_jobs(
            jobs[:1],
            SubmissionTarget(
                backend=SubmissionBackend.VOLCENGINE_ARK,
                config_path=str(config_path),
            ),
            source_type="package",
            source_id=package.scene_id,
        )
        assert batch.records[0].status == "submitted"
        assert batch.records[0].message == "https://example.invalid/seedream.png"
        assert batch.records[0].backend_job_id == "1764041608"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_write_submission_batch_creates_outputs():
    pipeline, package = _build_package()
    jobs = pipeline.build_submission_jobs_from_package(package)
    temp_dir = _workspace_temp_dir()
    try:
        batch = pipeline.submit_jobs(
            jobs,
            SubmissionTarget(backend=SubmissionBackend.DRY_RUN),
            source_type="package",
            source_id=package.scene_id,
        )
        files = pipeline.write_submission_batch(batch, temp_dir)
        assert set(files.keys()) == {"batch_json", "batch_markdown"}
        assert files["batch_json"].exists()
        assert files["batch_markdown"].exists()
        assert json.loads(files["batch_json"].read_text(encoding="utf-8"))["job_count"] == 5
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
