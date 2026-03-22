"""Download submission artifacts and sync them back into render manifests."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from .render_qa import RenderExpectation, load_render_manifest
from .submission import SubmissionBatch


class ArtifactDownloadRecord(BaseModel):
    """Downloaded artifact for a submission record."""

    job_id: str
    shot_id: str
    url: str
    output_path: str
    downloaded: bool
    error: str = ""


class ArtifactDownloadBatch(BaseModel):
    """Result of downloading artifacts for a submission batch."""

    source_id: str
    records: list[ArtifactDownloadRecord] = Field(default_factory=list)


def _artifact_url_from_record(record) -> str:
    """Extract a likely artifact URL from a submission record."""

    message = (record.message or "").strip()
    return message if message.startswith("http://") or message.startswith("https://") else ""


def _extension_from_url(url: str) -> str:
    """Infer a file extension from a URL."""

    path = urlparse(url).path
    suffix = Path(path).suffix.lower()
    return suffix or ".bin"


def download_submission_artifacts(
    batch: SubmissionBatch,
    output_dir: str | Path,
) -> ArtifactDownloadBatch:
    """Download any URL-based artifacts referenced by a submission batch."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[ArtifactDownloadRecord] = []
    for record in batch.records:
        url = _artifact_url_from_record(record)
        if not url:
            records.append(
                ArtifactDownloadRecord(
                    job_id=record.job_id,
                    shot_id=record.shot_id,
                    url="",
                    output_path="",
                    downloaded=False,
                    error="no_artifact_url",
                )
            )
            continue

        output_path = output_dir / f"{record.shot_id}{_extension_from_url(url)}"
        try:
            response = httpx.get(url, timeout=120.0)
            response.raise_for_status()
            output_path.write_bytes(response.content)
            records.append(
                ArtifactDownloadRecord(
                    job_id=record.job_id,
                    shot_id=record.shot_id,
                    url=url,
                    output_path=str(output_path),
                    downloaded=True,
                )
            )
        except Exception as exc:  # noqa: BLE001
            records.append(
                ArtifactDownloadRecord(
                    job_id=record.job_id,
                    shot_id=record.shot_id,
                    url=url,
                    output_path=str(output_path),
                    downloaded=False,
                    error=str(exc),
                )
            )

    return ArtifactDownloadBatch(source_id=batch.source_id, records=records)


def artifact_download_batch_json(batch: ArtifactDownloadBatch, indent: int = 2) -> str:
    """Serialise an artifact download batch to JSON."""

    return batch.model_dump_json(indent=indent)


def write_artifact_download_batch(
    batch: ArtifactDownloadBatch,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write artifact download batch metadata to disk."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "artifact_download_batch.json"
    json_path.write_text(artifact_download_batch_json(batch, indent=2), encoding="utf-8")
    return {"batch_json": json_path}


def update_render_manifest_from_downloads(
    manifest_path: str | Path,
    downloads: ArtifactDownloadBatch,
) -> list[RenderExpectation]:
    """Apply successful downloads back into a render manifest."""

    manifest_entries = load_render_manifest(manifest_path)
    download_by_shot = {record.shot_id: record for record in downloads.records if record.downloaded}

    for entry in manifest_entries:
        download = download_by_shot.get(entry.shot_id)
        if download is None:
            continue
        entry.status = "rendered"
        entry.output_path = download.output_path
        entry.actual_seed = entry.expected_seed
        entry.actual_width = entry.expected_width
        entry.actual_height = entry.expected_height
        entry.actual_prompt_hash = entry.expected_prompt_hash
        entry.notes.append("synced_from_submission_download")

    manifest_file = Path(manifest_path)
    manifest_file.write_text(
        json.dumps([entry.model_dump(mode="json") for entry in manifest_entries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_entries


def main() -> int:
    """CLI entry point for artifact download and manifest sync."""

    import argparse

    from .submission import SubmissionBatch

    parser = argparse.ArgumentParser(description="Download submission artifacts and sync render manifests.")
    parser.add_argument("--batch-file", required=True, help="Path to submission_batch.json")
    parser.add_argument("--artifacts-dir", required=True, help="Directory to store downloaded artifacts")
    parser.add_argument("--manifest-file", default="", help="Optional render manifest JSON to update")
    parser.add_argument("--output-dir", required=True, help="Directory for download batch outputs")
    args = parser.parse_args()

    batch = SubmissionBatch.model_validate_json(Path(args.batch_file).read_text(encoding="utf-8"))
    downloads = download_submission_artifacts(batch, args.artifacts_dir)
    output_files = write_artifact_download_batch(downloads, args.output_dir)

    if args.manifest_file:
        update_render_manifest_from_downloads(args.manifest_file, downloads)

    print(
        json.dumps(
            {
                "source_id": downloads.source_id,
                "downloaded_count": sum(record.downloaded for record in downloads.records),
                "output_files": {key: str(value) for key, value in output_files.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
