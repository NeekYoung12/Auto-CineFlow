"""Optional local visual review using an external Python env and local Qwen-VL weights."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from .config_loader import resolve_local_vlm_settings
from .keyframe_qa import KeyframeQAReport, keyframe_qa_report
from .result_ingest import ArtifactDownloadBatch, ArtifactDownloadRecord


class LocalVisualReviewResult(BaseModel):
    """Review result for a single keyframe image."""

    shot_id: str
    output_path: str
    status: str
    score: float | None = None
    recommendation: str = ""
    issues: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    reason: str = ""


class LocalVisualReviewReport(BaseModel):
    """Aggregate report for optional local visual review."""

    source_id: str
    enabled: bool
    skipped: bool
    reviewer: str = "qwen3-vl-local"
    results: list[LocalVisualReviewResult] = Field(default_factory=list)


def review_keyframes_with_local_vlm(
    keyframe_report: KeyframeQAReport,
    config_path: str | None = None,
    timeout_seconds: float = 600.0,
) -> LocalVisualReviewReport:
    """Run local visual review for keyframes using an external Python environment."""

    settings = resolve_local_vlm_settings(config_path)
    python_path = Path(settings["python_path"])
    model_path = Path(settings["model_path"])
    worker_path = Path(__file__).with_name("local_vlm_worker.py")

    if not python_path.exists() or not model_path.exists():
        return LocalVisualReviewReport(
            source_id=keyframe_report.source_id,
            enabled=False,
            skipped=True,
            results=[],
        )

    results: list[LocalVisualReviewResult] = []
    for item in keyframe_report.results:
        command = [
            str(python_path),
            str(worker_path),
            "--model-path",
            str(model_path),
            "--image-path",
            item.output_path,
            "--device-preference",
            settings["device_preference"],
            "--min-free-vram-gb",
            settings["min_free_vram_gb"],
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout_seconds)
        stdout = completed.stdout.strip()
        payload = json.loads(stdout or "{}") if stdout else {}
        status = str(payload.get("status", "error"))
        if status == "ok":
            result = payload.get("result", {})
            raw_score = float(result.get("score", 0.0))
            normalized_score = raw_score / 10.0 if raw_score > 1.0 and raw_score <= 10.0 else raw_score
            issues = [
                name
                for name in (
                    "text_artifact",
                    "face_distortion",
                    "anatomy_issue",
                    "blur_issue",
                    "wardrobe_drift",
                    "scene_incoherence",
                )
                if result.get(name)
            ]
            results.append(
                LocalVisualReviewResult(
                    shot_id=item.shot_id,
                    output_path=item.output_path,
                    status="ok",
                    score=normalized_score,
                    recommendation=str(result.get("recommendation", "")),
                    issues=issues,
                    notes=[str(note) for note in result.get("notes", [])],
                )
            )
        else:
            results.append(
                LocalVisualReviewResult(
                    shot_id=item.shot_id,
                    output_path=item.output_path,
                    status=status,
                    recommendation="skip",
                    reason=str(payload.get("reason", "") or completed.stderr[-300:]),
                )
            )

    return LocalVisualReviewReport(
        source_id=keyframe_report.source_id,
        enabled=True,
        skipped=all(result.status == "skipped" for result in results) if results else True,
        results=results,
    )


def build_keyframe_download_batch_from_dir(
    artifacts_dir: str | Path,
    source_id: str = "KEYFRAME_REVIEW",
) -> ArtifactDownloadBatch:
    """Build a download-like batch from an existing directory of keyframe images."""

    artifacts_dir = Path(artifacts_dir)
    records: list[ArtifactDownloadRecord] = []
    for path in sorted(artifacts_dir.glob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            continue
        records.append(
            ArtifactDownloadRecord(
                job_id=path.stem,
                shot_id=path.stem,
                url="",
                output_path=str(path),
                downloaded=True,
            )
        )
    return ArtifactDownloadBatch(source_id=source_id, records=records)


def write_local_visual_review_report(report: LocalVisualReviewReport, output_dir: str | Path) -> dict[str, Path]:
    """Write local visual review outputs to disk."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "local_visual_review_report.json"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return {"report_json": json_path}


def main() -> int:
    """CLI entry point for local visual keyframe review."""

    import argparse

    from .keyframe_qa import KeyframeQAReport

    parser = argparse.ArgumentParser(description="Run optional local visual review on keyframe images.")
    parser.add_argument("--artifacts-dir", default="", help="Directory containing keyframe images")
    parser.add_argument("--keyframe-qa-file", default="", help="Existing keyframe_qa_report.json")
    parser.add_argument("--source-id", default="KEYFRAME_REVIEW", help="Source identifier for ad-hoc review")
    parser.add_argument("--config-path", default=None, help="Path to config file")
    parser.add_argument("--output-dir", required=True, help="Directory for review outputs")
    parser.add_argument("--timeout-seconds", type=float, default=600.0, help="Timeout for the external VLM subprocess")
    args = parser.parse_args()

    if not args.artifacts_dir and not args.keyframe_qa_file:
        raise SystemExit("Provide --artifacts-dir or --keyframe-qa-file.")

    if args.keyframe_qa_file:
        keyframe_report = KeyframeQAReport.model_validate_json(Path(args.keyframe_qa_file).read_text(encoding="utf-8"))
    else:
        downloads = build_keyframe_download_batch_from_dir(args.artifacts_dir, source_id=args.source_id)
        keyframe_report = keyframe_qa_report(downloads)

    review_report = review_keyframes_with_local_vlm(
        keyframe_report,
        config_path=args.config_path,
        timeout_seconds=args.timeout_seconds,
    )
    output_files = write_local_visual_review_report(review_report, args.output_dir)
    print(
        json.dumps(
            {
                "source_id": review_report.source_id,
                "enabled": review_report.enabled,
                "skipped": review_report.skipped,
                "result_count": len(review_report.results),
                "output_files": {key: str(value) for key, value in output_files.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
