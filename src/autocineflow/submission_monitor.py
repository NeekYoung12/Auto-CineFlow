"""Monitor submission batches against a filesystem-backed spool."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from .submission import SubmissionBatch


class SubmissionJobStatus(BaseModel):
    """Observed status for a submitted job."""

    job_id: str
    shot_id: str
    status: str
    location: str = ""
    response_path: str = ""


class SubmissionMonitorReport(BaseModel):
    """Filesystem spool monitoring report for a submission batch."""

    generated_at: str
    backend: str
    provider: str
    source_id: str
    total_jobs: int = Field(..., ge=0)
    queued_jobs: int = Field(..., ge=0)
    processing_jobs: int = Field(..., ge=0)
    completed_jobs: int = Field(..., ge=0)
    failed_jobs: int = Field(..., ge=0)
    unknown_jobs: int = Field(..., ge=0)
    all_finished: bool
    job_statuses: list[SubmissionJobStatus] = Field(default_factory=list)


def monitor_filesystem_submission_batch(
    batch: SubmissionBatch,
    spool_dir: str | Path,
) -> SubmissionMonitorReport:
    """Inspect a filesystem spool and report job status for a submission batch."""

    spool_dir = Path(spool_dir)
    status_dirs = {
        "queued": spool_dir / "queued",
        "processing": spool_dir / "processing",
        "completed": spool_dir / "completed",
        "failed": spool_dir / "failed",
    }

    statuses: list[SubmissionJobStatus] = []
    counts = {key: 0 for key in ["queued", "processing", "completed", "failed", "unknown"]}

    for record in batch.records:
        matched_status = "unknown"
        matched_path = ""
        for status_name, directory in status_dirs.items():
            candidate = directory / f"{record.job_id}.json"
            if candidate.exists():
                matched_status = status_name
                matched_path = str(candidate)
                break

        counts[matched_status] += 1
        statuses.append(
            SubmissionJobStatus(
                job_id=record.job_id,
                shot_id=record.shot_id,
                status=matched_status,
                location=matched_path,
                response_path=record.response_path,
            )
        )

    return SubmissionMonitorReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        backend=batch.backend.value,
        provider=batch.provider.value,
        source_id=batch.source_id,
        total_jobs=len(batch.records),
        queued_jobs=counts["queued"],
        processing_jobs=counts["processing"],
        completed_jobs=counts["completed"],
        failed_jobs=counts["failed"],
        unknown_jobs=counts["unknown"],
        all_finished=(counts["completed"] + counts["failed"]) == len(batch.records),
        job_statuses=statuses,
    )


def submission_monitor_report_json(report: SubmissionMonitorReport, indent: int = 2) -> str:
    """Serialise a submission monitor report to JSON."""

    return report.model_dump_json(indent=indent)


def submission_monitor_markdown(report: SubmissionMonitorReport) -> str:
    """Export a human-readable submission monitor document."""

    lines = [
        f"# Submission Monitor {report.source_id}",
        "",
        f"- Backend: `{report.backend}`",
        f"- Provider: `{report.provider}`",
        f"- Total Jobs: `{report.total_jobs}`",
        f"- Queued / Processing / Completed / Failed / Unknown: `{report.queued_jobs}/{report.processing_jobs}/{report.completed_jobs}/{report.failed_jobs}/{report.unknown_jobs}`",
        f"- All Finished: `{report.all_finished}`",
        "",
        "## Job Status",
        "",
    ]
    for status in report.job_statuses:
        lines.extend(
            [
                f"### {status.job_id}",
                "",
                f"- Shot: `{status.shot_id}`",
                f"- Status: `{status.status}`",
                f"- Location: `{status.location or 'n/a'}`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def write_submission_monitor_report(
    report: SubmissionMonitorReport,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write submission monitor outputs to disk."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    json_path = target_dir / "submission_monitor_report.json"
    markdown_path = target_dir / "submission_monitor_report.md"
    json_path.write_text(submission_monitor_report_json(report, indent=2), encoding="utf-8")
    markdown_path.write_text(submission_monitor_markdown(report), encoding="utf-8")
    return {
        "report_json": json_path,
        "report_markdown": markdown_path,
    }


def main() -> int:
    """CLI entry point for submission monitoring."""

    import argparse

    from .submission import SubmissionBatch

    parser = argparse.ArgumentParser(description="Monitor a submission batch against a filesystem spool.")
    parser.add_argument("--batch-file", required=True, help="Path to submission_batch.json")
    parser.add_argument("--spool-dir", required=True, help="Filesystem spool directory")
    parser.add_argument("--output-dir", required=True, help="Directory for monitor outputs")
    args = parser.parse_args()

    batch = SubmissionBatch.model_validate_json(Path(args.batch_file).read_text(encoding="utf-8"))
    report = monitor_filesystem_submission_batch(batch, args.spool_dir)
    output_files = write_submission_monitor_report(report, args.output_dir)
    print(
        json.dumps(
            {
                "source_id": report.source_id,
                "all_finished": report.all_finished,
                "output_files": {key: str(value) for key, value in output_files.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
