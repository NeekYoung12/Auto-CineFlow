"""Targeted keyframe repair runner for scenes blocked before video generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .delivery import StoryboardPackage
from .pipeline import CineFlowPipeline
from .scene_runner import (
    _build_rebuild_keyframe_jobs,
    _inject_bootstrap_keyframes,
)
from .submission import SubmissionBackend, SubmissionProvider, SubmissionTarget


def select_keyframe_repair_jobs(keyframe_jobs, gate_report_payload: dict | None):
    """Select only the shots that failed a prior keyframe gate, else keep all jobs."""

    if not gate_report_payload:
        return list(keyframe_jobs)
    blocked_shots = {
        item.get("shot_id", "")
        for item in gate_report_payload.get("results", []) or []
        if not item.get("passes_gate", True)
    }
    if not blocked_shots:
        return list(keyframe_jobs)
    return [job for job in keyframe_jobs if job.shot_id in blocked_shots]


def main() -> int:
    """CLI entry point for rerunning only keyframe generation/repair stages."""

    parser = argparse.ArgumentParser(description="Rerun only keyframe stages for a blocked scene.")
    parser.add_argument("--run-dir", required=True, help="Root output directory of a previous scene run")
    parser.add_argument("--config-path", default=None, help="Path to config file")
    parser.add_argument("--timeout-seconds", type=float, default=120.0, help="Submission timeout")
    parser.add_argument("--poll-interval-seconds", type=float, default=10.0, help="Artifact polling interval")
    parser.add_argument("--enable-local-vlm-review", action="store_true", help="Run optional local visual review")
    parser.add_argument("--local-vlm-timeout-seconds", type=float, default=600.0, help="Timeout for local visual review subprocess")
    parser.add_argument("--continue-video", action="store_true", help="If keyframes pass, continue directly into video generation")
    parser.add_argument("--video-provider", default="runninghub_video_auto", choices=[item.value for item in SubmissionProvider])
    parser.add_argument("--video-backend", default="runninghub_api", choices=[item.value for item in SubmissionBackend])
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    pipeline = CineFlowPipeline(config_path=args.config_path)
    package = StoryboardPackage.model_validate_json((run_dir / "delivery" / "storyboard_package.json").read_text(encoding="utf-8"))

    gate_path = run_dir / "keyframe_qc" / "keyframe_gate_report.json"
    gate_payload = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else None

    all_keyframe_jobs = pipeline.build_submission_jobs_from_package(package, provider=SubmissionProvider.RUNNINGHUB_FACEID)
    keyframe_jobs = select_keyframe_repair_jobs(all_keyframe_jobs, gate_payload)

    target = SubmissionTarget(
        backend=SubmissionBackend.RUNNINGHUB_API,
        config_path=args.config_path or "",
        timeout_seconds=args.timeout_seconds,
    )

    batch = pipeline.submit_jobs(
        keyframe_jobs,
        target,
        source_type="keyframe_repair",
        source_id=package.scene_id,
    )
    submission_files = pipeline.write_submission_batch(batch, run_dir / "keyframe_repair" / "submission")
    downloads = pipeline.download_submission_artifacts(
        batch,
        run_dir / "keyframe_repair" / "artifacts",
        config_path=args.config_path,
        timeout_seconds=max(args.timeout_seconds, 900.0),
        poll_interval_seconds=args.poll_interval_seconds,
        skip_existing=False,
    )
    download_files = pipeline.write_artifact_download_batch(downloads, run_dir / "keyframe_repair" / "downloads")
    qa_report = pipeline.keyframe_qa_report(downloads)
    qa_files = pipeline.write_keyframe_qa_report(qa_report, run_dir / "keyframe_repair" / "qa")

    rebuild_jobs = _build_rebuild_keyframe_jobs(keyframe_jobs, downloads)
    rebuild_files = {}
    selected_downloads = downloads
    if rebuild_jobs:
        rebuild_batch = pipeline.submit_jobs(
            rebuild_jobs,
            target,
            source_type="keyframe_repair_rebuild",
            source_id=package.scene_id,
        )
        rebuild_files.update(
            {
                f"rebuild_{key}": str(value)
                for key, value in pipeline.write_submission_batch(
                    rebuild_batch,
                    run_dir / "keyframe_repair" / "rebuild_submission",
                ).items()
            }
        )
        rebuild_downloads = pipeline.download_submission_artifacts(
            rebuild_batch,
            run_dir / "keyframe_repair" / "rebuild_artifacts",
            config_path=args.config_path,
            timeout_seconds=max(args.timeout_seconds, 900.0),
            poll_interval_seconds=args.poll_interval_seconds,
            skip_existing=False,
        )
        rebuild_files.update(
            {
                f"rebuild_{key}": str(value)
                for key, value in pipeline.write_artifact_download_batch(
                    rebuild_downloads,
                    run_dir / "keyframe_repair" / "rebuild_downloads",
                ).items()
            }
        )
        rebuild_report = pipeline.keyframe_qa_report(rebuild_downloads)
        rebuild_files.update(
            {
                f"rebuild_{key}": str(value)
                for key, value in pipeline.write_keyframe_qa_report(
                    rebuild_report,
                    run_dir / "keyframe_repair" / "rebuild_qa",
                ).items()
            }
        )
        selected_downloads = pipeline.select_best_keyframe_downloads(downloads, rebuild_downloads)

    local_review_files = {}
    if args.enable_local_vlm_review:
        review_report = pipeline.review_keyframes_with_local_vlm(
            pipeline.keyframe_qa_report(selected_downloads),
            config_path=args.config_path,
            timeout_seconds=args.local_vlm_timeout_seconds,
        )
        local_review_files = pipeline.write_local_visual_review_report(
            review_report,
            run_dir / "keyframe_repair" / "local_vlm",
        )
    else:
        review_report = None

    gate_report = pipeline.build_keyframe_gate_report(
        pipeline.keyframe_qa_report(selected_downloads),
        local_visual_report=review_report,
    )
    gate_files = pipeline.write_keyframe_gate_report(gate_report, run_dir / "keyframe_repair")

    continued_video_files = {}
    if args.continue_video and gate_report.passes_gate:
        video_jobs = pipeline.build_submission_jobs_from_package(package, provider=SubmissionProvider(args.video_provider))
        target_shots = {record.shot_id for record in selected_downloads.records}
        if target_shots:
            video_jobs = [job for job in video_jobs if job.shot_id in target_shots]
        video_jobs = _inject_bootstrap_keyframes(video_jobs, selected_downloads)
        video_batch = pipeline.submit_jobs(
            video_jobs,
            SubmissionTarget(
                backend=SubmissionBackend(args.video_backend),
                config_path=args.config_path or "",
                timeout_seconds=args.timeout_seconds,
            ),
            source_type="keyframe_repair_continue_video",
            source_id=package.scene_id,
        )
        continued_video_files.update(
            {
                f"video_{key}": str(value)
                for key, value in pipeline.write_submission_batch(
                    video_batch,
                    run_dir / "keyframe_repair" / "video_submission",
                ).items()
            }
        )
        video_downloads = pipeline.download_submission_artifacts(
            video_batch,
            run_dir / "artifacts",
            config_path=args.config_path,
            timeout_seconds=max(args.timeout_seconds, 900.0),
            poll_interval_seconds=args.poll_interval_seconds,
            skip_existing=False,
        )
        continued_video_files.update(
            {
                f"video_{key}": str(value)
                for key, value in pipeline.write_artifact_download_batch(
                    video_downloads,
                    run_dir / "keyframe_repair" / "video_downloads",
                ).items()
            }
        )

    print(
        json.dumps(
            {
                "scene_id": package.scene_id,
                "keyframe_jobs": len(keyframe_jobs),
                "passes_gate": gate_report.passes_gate,
                "blocking_reason": gate_report.blocking_reason,
                "submission_files": {key: str(value) for key, value in submission_files.items()},
                "download_files": {key: str(value) for key, value in download_files.items()},
                "qa_files": {key: str(value) for key, value in qa_files.items()},
                "rebuild_files": rebuild_files,
                "local_review_files": {key: str(value) for key, value in local_review_files.items()},
                "gate_files": {key: str(value) for key, value in gate_files.items()},
                "continued_video_files": continued_video_files,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
