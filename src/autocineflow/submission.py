"""Render task submission builders and backends."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .delivery import StoryboardPackage
from .execution_planner import ProjectExecutionPlan
from .provider_payloads import automatic1111_bundle, comfyui_bundle


class SubmissionBackend(str, Enum):
    """Supported submission backends."""

    FILESYSTEM = "filesystem"
    WEBHOOK = "webhook"
    DRY_RUN = "dry_run"


class SubmissionProvider(str, Enum):
    """Payload provider variants."""

    GENERIC = "generic"
    AUTOMATIC1111 = "automatic1111"
    COMFYUI = "comfyui"


class SubmissionTarget(BaseModel):
    """Submission backend configuration."""

    backend: SubmissionBackend
    spool_dir: str = ""
    webhook_url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=30.0, ge=0.1)


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


class SubmissionBatch(BaseModel):
    """Submission manifest for a batch of jobs."""

    generated_at: str
    provider: SubmissionProvider
    backend: SubmissionBackend
    source_type: str
    source_id: str
    job_count: int = Field(..., ge=0)
    records: list[SubmissionRecord] = Field(default_factory=list)


def build_submission_jobs_from_package(
    package: StoryboardPackage,
    provider: SubmissionProvider = SubmissionProvider.GENERIC,
) -> list[SubmissionJob]:
    """Build submission jobs from a scene package."""

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
    return [
        SubmissionJob(
            job_id=item["job_id"],
            shot_id=item["shot_id"],
            scene_id=item.get("metadata", {}).get("scene_id", ""),
            provider=provider,
            payload=dict(item),
        )
        for item in queue
    ]


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

    import httpx

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
