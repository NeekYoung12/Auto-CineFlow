"""Render task submission builders and backends."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from .config_loader import resolve_minimax_media_settings
from .delivery import StoryboardPackage
from .execution_planner import ProjectExecutionPlan
from .provider_payloads import automatic1111_bundle, comfyui_bundle


class SubmissionBackend(str, Enum):
    """Supported submission backends."""

    FILESYSTEM = "filesystem"
    WEBHOOK = "webhook"
    DRY_RUN = "dry_run"
    MINIMAX_API = "minimax_api"


class SubmissionProvider(str, Enum):
    """Payload provider variants."""

    GENERIC = "generic"
    AUTOMATIC1111 = "automatic1111"
    COMFYUI = "comfyui"
    MINIMAX_IMAGE = "minimax_image"
    MINIMAX_VIDEO = "minimax_video"


class SubmissionTarget(BaseModel):
    """Submission backend configuration."""

    backend: SubmissionBackend
    spool_dir: str = ""
    webhook_url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=30.0, ge=0.1)
    config_path: str = ""


class SubmissionJob(BaseModel):
    """Single render submission job."""

    job_id: str
    shot_id: str
    scene_id: str = ""
    provider: SubmissionProvider
    payload: dict[str, Any]


class SubmissionRecord(BaseModel):
    """Result of submitting a single job."""

    job_id: str
    shot_id: str
    provider: SubmissionProvider
    backend: SubmissionBackend
    status: str
    submitted_at: str
    backend_job_id: str = ""
    request_path: str = ""
    response_path: str = ""
    message: str = ""
    provider_status_code: int = 0
    provider_status_message: str = ""


class SubmissionBatch(BaseModel):
    """Submission manifest for a batch of jobs."""

    generated_at: str
    provider: SubmissionProvider
    backend: SubmissionBackend
    source_type: str
    source_id: str
    job_count: int = Field(..., ge=0)
    records: list[SubmissionRecord] = Field(default_factory=list)


def merge_submission_batches(*batches: SubmissionBatch) -> SubmissionBatch:
    """Merge multiple submission batches, keeping the latest record per job."""

    merged_records: dict[str, SubmissionRecord] = {}
    ordered_ids: list[str] = []
    source_id = ""
    source_type = ""
    provider = SubmissionProvider.GENERIC
    backend = SubmissionBackend.DRY_RUN

    for batch in batches:
        source_id = batch.source_id or source_id
        source_type = batch.source_type or source_type
        provider = batch.provider or provider
        backend = batch.backend or backend
        for record in batch.records:
            if record.job_id not in merged_records:
                ordered_ids.append(record.job_id)
            merged_records[record.job_id] = record

    return SubmissionBatch(
        generated_at=datetime.now(timezone.utc).isoformat(),
        provider=provider,
        backend=backend,
        source_type=source_type,
        source_id=source_id,
        job_count=len(merged_records),
        records=[merged_records[job_id] for job_id in ordered_ids],
    )


def build_submission_jobs_from_package(
    package: StoryboardPackage,
    provider: SubmissionProvider = SubmissionProvider.GENERIC,
) -> list[SubmissionJob]:
    """Build submission jobs from a scene package."""

    if provider == SubmissionProvider.MINIMAX_IMAGE:
        jobs: list[SubmissionJob] = []
        for job in package.render_queue:
            payload = {
                "model": "image-01",
                "prompt": job.prompt,
                "aspect_ratio": _aspect_ratio_label(job.width, job.height),
                "response_format": "url",
                "seed": job.render_seed,
                "n": 1,
                "prompt_optimizer": False,
                "aigc_watermark": False,
            }
            reference_urls = _external_reference_urls(job.metadata.get("character_reference_images", []))
            if reference_urls:
                payload["subject_reference"] = [
                    {"type": "character", "image_file": url}
                    for url in reference_urls[:3]
                ]
            jobs.append(
                SubmissionJob(
                    job_id=job.job_id,
                    shot_id=job.shot_id,
                    scene_id=job.metadata.get("scene_id", package.scene_id),
                    provider=provider,
                    payload=payload,
                )
            )
        return jobs

    if provider == SubmissionProvider.MINIMAX_VIDEO:
        return [
            SubmissionJob(
                job_id=segment.segment_id,
                shot_id=segment.shot_id,
                scene_id=segment.scene_id,
                provider=provider,
                payload={
                    "model": "MiniMax-Hailuo-02",
                    "prompt": _condense_minimax_video_prompt(segment.prompt),
                    "duration": int(segment.generation_duration_seconds),
                    "resolution": "768P",
                },
            )
            for segment in package.video_segments
        ]

    if provider == SubmissionProvider.AUTOMATIC1111:
        bundle = automatic1111_bundle(package)
        return [
            SubmissionJob(
                job_id=item["job_id"],
                shot_id=item["shot_id"],
                scene_id=item["metadata"].get("scene_id", package.scene_id),
                provider=provider,
                payload=item,
            )
            for item in bundle
        ]

    if provider == SubmissionProvider.COMFYUI:
        bundle = comfyui_bundle(package)
        return [
            SubmissionJob(
                job_id=item["job_id"],
                shot_id=item["shot_id"],
                scene_id=item["metadata"].get("scene_id", package.scene_id),
                provider=provider,
                payload=item,
            )
            for item in bundle
        ]

    return [
        SubmissionJob(
            job_id=job.job_id,
            shot_id=job.shot_id,
            scene_id=job.metadata.get("scene_id", package.scene_id),
            provider=provider,
            payload=job.model_dump(mode="json"),
        )
        for job in package.render_queue
    ]


def build_submission_jobs_from_execution_plan(
    plan: ProjectExecutionPlan,
    provider: SubmissionProvider = SubmissionProvider.GENERIC,
) -> list[SubmissionJob]:
    """Build submission jobs from a project execution plan."""

    queue = plan.ordered_rerender_queue or plan.rerender_queue
    jobs: list[SubmissionJob] = []
    for item in queue:
        if provider == SubmissionProvider.MINIMAX_IMAGE:
            width = int(item.get("width", 1536))
            height = int(item.get("height", 864))
            payload = {
                "model": "image-01",
                "prompt": item.get("prompt", ""),
                "aspect_ratio": _aspect_ratio_label(width, height),
                "response_format": "url",
                "seed": int(item.get("render_seed", 0)),
                "n": 1,
                "prompt_optimizer": False,
                "aigc_watermark": False,
            }
        elif provider == SubmissionProvider.MINIMAX_VIDEO:
            payload = {
                "model": "MiniMax-Hailuo-02",
                "prompt": _condense_minimax_video_prompt(item.get("prompt", "")),
                "duration": 6,
                "resolution": "768P",
            }
        else:
            payload = dict(item)

        jobs.append(
            SubmissionJob(
                job_id=item["job_id"],
                shot_id=item["shot_id"],
                scene_id=item.get("metadata", {}).get("scene_id", ""),
                provider=provider,
                payload=payload,
            )
        )
    return jobs


def _aspect_ratio_label(width: int, height: int) -> str:
    """Map width/height to the closest supported aspect-ratio label."""

    ratio = width / max(height, 1)
    supported = {
        "21:9": 21 / 9,
        "16:9": 16 / 9,
        "4:3": 4 / 3,
        "1:1": 1.0,
        "3:4": 3 / 4,
        "9:16": 9 / 16,
    }
    return min(supported, key=lambda label: abs(supported[label] - ratio))


def _external_reference_urls(paths: list[Any]) -> list[str]:
    """Keep only remote URLs that a hosted API can resolve directly."""

    urls: list[str] = []
    for path in paths:
        if isinstance(path, str) and path.startswith(("http://", "https://")) and path not in urls:
            urls.append(path)
    return urls


def _condense_minimax_video_prompt(prompt: str, max_chars: int = 320) -> str:
    """Reduce a long cinematic prompt to a shorter video-friendly prompt."""

    banned_fragments = {
        "8k resolution",
        "highly detailed",
        "professional cinematography",
        "cinematic photography",
        "film grain",
        "balanced contrast",
        "extreme contrast",
        "static camera",
        "steady",
        "professional",
    }
    parts: list[str] = []
    for raw_part in prompt.split(","):
        cleaned = raw_part.strip()
        if not cleaned:
            continue
        if cleaned.lower() in banned_fragments:
            continue
        if cleaned not in parts:
            parts.append(cleaned)

    kept: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}, {part}".strip(", ")
        if len(candidate) > max_chars:
            break
        kept.append(part)
        current = candidate

    return ", ".join(kept)


def _submit_to_filesystem(job: SubmissionJob, target: SubmissionTarget) -> SubmissionRecord:
    """Write a submission payload into a filesystem-backed queue."""

    spool_dir = Path(target.spool_dir)
    queued_dir = spool_dir / "queued"
    queued_dir.mkdir(parents=True, exist_ok=True)
    request_path = queued_dir / f"{job.job_id}.json"
    request_path.write_text(json.dumps(job.payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return SubmissionRecord(
        job_id=job.job_id,
        shot_id=job.shot_id,
        provider=job.provider,
        backend=target.backend,
        status="queued",
        submitted_at=datetime.now(timezone.utc).isoformat(),
        request_path=str(request_path),
    )


def _submit_to_webhook(job: SubmissionJob, target: SubmissionTarget) -> SubmissionRecord:
    """Submit a job to a generic webhook endpoint."""

    if not target.webhook_url:
        raise ValueError("webhook_url is required for webhook submissions.")

    response = httpx.post(
        target.webhook_url,
        json=job.payload,
        headers=target.headers,
        timeout=target.timeout_seconds,
    )
    response.raise_for_status()
    response_payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    backend_job_id = str(response_payload.get("job_id", response_payload.get("id", "")))

    return SubmissionRecord(
        job_id=job.job_id,
        shot_id=job.shot_id,
        provider=job.provider,
        backend=target.backend,
        status="submitted",
        submitted_at=datetime.now(timezone.utc).isoformat(),
        backend_job_id=backend_job_id,
        message=response.text[:300],
    )


def _submit_to_minimax_api(job: SubmissionJob, target: SubmissionTarget) -> SubmissionRecord:
    """Submit a media generation request to the MiniMax API."""

    api_key, base_url = resolve_minimax_media_settings(target.config_path or None)
    if not api_key:
        raise ValueError("MiniMax media API key not found in config.")

    endpoint = "/image_generation" if job.provider == SubmissionProvider.MINIMAX_IMAGE else "/video_generation"
    response = httpx.post(
        f"{base_url}{endpoint}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=job.payload,
        timeout=target.timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    media_message = ""
    backend_job_id = ""
    provider_status_code = 0
    provider_status_message = ""
    if isinstance(payload, dict):
        base_resp = payload.get("base_resp", {})
        if isinstance(base_resp, dict):
            provider_status_code = int(base_resp.get("status_code", 0) or 0)
            provider_status_message = str(base_resp.get("status_msg", "") or "")
        if job.provider == SubmissionProvider.MINIMAX_IMAGE:
            data = payload.get("data", {})
            if isinstance(data, dict):
                image_urls = data.get("image_urls", [])
                if image_urls:
                    media_message = str(image_urls[0])
        else:
            backend_job_id = str(payload.get("task_id", ""))
            media_message = backend_job_id

    return SubmissionRecord(
        job_id=job.job_id,
        shot_id=job.shot_id,
        provider=job.provider,
        backend=target.backend,
        status="submitted",
        submitted_at=datetime.now(timezone.utc).isoformat(),
        backend_job_id=backend_job_id or (str(payload.get("id", "")) if isinstance(payload, dict) else ""),
        message=media_message or response.text[:300],
        provider_status_code=provider_status_code,
        provider_status_message=provider_status_message,
    )


def _submit_dry_run(job: SubmissionJob, target: SubmissionTarget) -> SubmissionRecord:
    """Return a synthetic submission record without side effects."""

    return SubmissionRecord(
        job_id=job.job_id,
        shot_id=job.shot_id,
        provider=job.provider,
        backend=target.backend,
        status="dry_run",
        submitted_at=datetime.now(timezone.utc).isoformat(),
        message="dry run only",
    )


def submit_jobs(
    jobs: list[SubmissionJob],
    target: SubmissionTarget,
    source_type: str,
    source_id: str,
) -> SubmissionBatch:
    """Submit jobs using the selected backend."""

    records: list[SubmissionRecord] = []
    for job in jobs:
        if target.backend == SubmissionBackend.FILESYSTEM:
            record = _submit_to_filesystem(job, target)
        elif target.backend == SubmissionBackend.MINIMAX_API:
            record = _submit_to_minimax_api(job, target)
        elif target.backend == SubmissionBackend.WEBHOOK:
            record = _submit_to_webhook(job, target)
        else:
            record = _submit_dry_run(job, target)
        records.append(record)

    provider = jobs[0].provider if jobs else SubmissionProvider.GENERIC
    return SubmissionBatch(
        generated_at=datetime.now(timezone.utc).isoformat(),
        provider=provider,
        backend=target.backend,
        source_type=source_type,
        source_id=source_id,
        job_count=len(jobs),
        records=records,
    )


def submission_batch_json(batch: SubmissionBatch, indent: int = 2) -> str:
    """Serialise a submission batch to JSON."""

    return batch.model_dump_json(indent=indent)


def submission_batch_markdown(batch: SubmissionBatch) -> str:
    """Export a human-readable submission manifest."""

    lines = [
        f"# Submission Batch {batch.source_id}",
        "",
        f"- Backend: `{batch.backend.value}`",
        f"- Provider: `{batch.provider.value}`",
        f"- Source Type: `{batch.source_type}`",
        f"- Job Count: `{batch.job_count}`",
        "",
        "## Records",
        "",
    ]
    for record in batch.records:
        lines.extend(
            [
                f"### {record.job_id}",
                "",
                f"- Shot: `{record.shot_id}`",
                f"- Status: `{record.status}`",
                f"- Backend Job ID: `{record.backend_job_id or 'n/a'}`",
                f"- Request Path: `{record.request_path or 'n/a'}`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def write_submission_batch(batch: SubmissionBatch, output_dir: str | Path) -> dict[str, Path]:
    """Write submission batch outputs to disk."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    json_path = target_dir / "submission_batch.json"
    markdown_path = target_dir / "submission_batch.md"
    json_path.write_text(submission_batch_json(batch, indent=2), encoding="utf-8")
    markdown_path.write_text(submission_batch_markdown(batch), encoding="utf-8")
    return {
        "batch_json": json_path,
        "batch_markdown": markdown_path,
    }


def main() -> int:
    """CLI entry point for submitting package or execution-plan jobs."""

    import argparse

    from .delivery import StoryboardPackage
    from .execution_planner import ProjectExecutionPlan

    parser = argparse.ArgumentParser(description="Submit render jobs from a package or execution plan.")
    parser.add_argument("--package-file", default=None, help="Path to storyboard_package.json")
    parser.add_argument("--execution-plan-file", default=None, help="Path to project_execution_plan.json")
    parser.add_argument("--provider", default="generic", choices=[item.value for item in SubmissionProvider])
    parser.add_argument("--backend", default="dry_run", choices=[item.value for item in SubmissionBackend])
    parser.add_argument("--spool-dir", default="", help="Filesystem spool directory")
    parser.add_argument("--webhook-url", default="", help="Webhook endpoint")
    parser.add_argument("--config-path", default="", help="Path to config file for API-backed submissions")
    parser.add_argument("--output-dir", required=True, help="Directory for submission outputs")
    args = parser.parse_args()

    if bool(args.package_file) == bool(args.execution_plan_file):
        raise SystemExit("Provide exactly one of --package-file or --execution-plan-file.")

    provider = SubmissionProvider(args.provider)
    backend = SubmissionBackend(args.backend)
    target = SubmissionTarget(
        backend=backend,
        spool_dir=args.spool_dir,
        webhook_url=args.webhook_url,
        config_path=args.config_path,
    )

    if args.package_file:
        package = StoryboardPackage.model_validate_json(Path(args.package_file).read_text(encoding="utf-8"))
        jobs = build_submission_jobs_from_package(package, provider=provider)
        batch = submit_jobs(jobs, target, source_type="package", source_id=package.scene_id)
    else:
        plan = ProjectExecutionPlan.model_validate_json(Path(args.execution_plan_file).read_text(encoding="utf-8"))
        jobs = build_submission_jobs_from_execution_plan(plan, provider=provider)
        batch = submit_jobs(jobs, target, source_type="execution_plan", source_id=plan.current_project_name)

    output_files = write_submission_batch(batch, args.output_dir)
    print(
        json.dumps(
            {
                "job_count": batch.job_count,
                "backend": batch.backend.value,
                "provider": batch.provider.value,
                "output_files": {key: str(value) for key, value in output_files.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
