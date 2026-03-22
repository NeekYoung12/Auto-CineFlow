"""One-shot orchestration for scene generation, submission, download, and QA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .delivery import RenderPreset
from .pipeline import CineFlowPipeline
from .submission import SubmissionBackend, SubmissionProvider, SubmissionTarget


def main() -> int:
    """CLI entry point for the end-to-end scene run."""

    parser = argparse.ArgumentParser(description="Run the full scene pipeline end-to-end.")
    parser.add_argument("--description", required=True, help="Scene description")
    parser.add_argument("--scene-id", default="SCENE_01", help="Scene identifier")
    parser.add_argument("--project-name", default="Auto-CineFlow Project", help="Project name")
    parser.add_argument("--output-dir", required=True, help="Directory for all outputs")
    parser.add_argument("--config-path", default=None, help="Path to config file")
    parser.add_argument("--num-shots", type=int, default=5, help="Number of shots")
    parser.add_argument("--target-duration-seconds", type=float, default=0.0, help="Optional final sequence target duration")
    parser.add_argument("--clip-duration-seconds", type=float, default=4.0, help="Preferred short clip duration for assembly planning")
    parser.add_argument("--job-limit", type=int, default=0, help="Optional limit on submitted jobs; 0 means all")
    parser.add_argument("--offline", action="store_true", help="Disable LLM analysis")
    parser.add_argument("--emotion-override", default=None, help="Override detected emotion")
    parser.add_argument("--provider", default="minimax_image", choices=[item.value for item in SubmissionProvider])
    parser.add_argument("--backend", default="minimax_api", choices=[item.value for item in SubmissionBackend])
    parser.add_argument("--spool-dir", default="", help="Filesystem spool directory")
    parser.add_argument("--timeout-seconds", type=float, default=120.0, help="Submission timeout")
    parser.add_argument("--poll-interval-seconds", type=float, default=10.0, help="Artifact polling interval for async providers")
    parser.add_argument("--auto-retry-transient", type=int, default=1, help="Automatic retry attempts for transient provider failures")
    parser.add_argument("--skip-download", action="store_true", help="Do not download URL artifacts")
    parser.add_argument("--skip-assemble", action="store_true", help="Do not auto-stitch downloaded video clips")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    pipeline = CineFlowPipeline(config_path=args.config_path)
    num_shots = args.num_shots
    if args.target_duration_seconds > 0:
        num_shots = pipeline.recommended_video_shot_count(
            target_duration_seconds=args.target_duration_seconds,
            clip_duration_seconds=args.clip_duration_seconds,
        )
    context = pipeline.run(
        description=args.description,
        num_shots=num_shots,
        scene_id=args.scene_id,
        use_llm=not args.offline,
        emotion_override=args.emotion_override,
    )
    render_preset = RenderPreset(
        video_target_duration_seconds=args.clip_duration_seconds,
        video_max_duration_seconds=max(args.clip_duration_seconds, 10.0),
        video_handle_seconds=min(1.0, args.clip_duration_seconds / 4.0),
    )
    package = pipeline.build_storyboard_package(
        context,
        project_name=args.project_name,
        render_preset=render_preset,
    )
    delivery_files = pipeline.write_delivery_package(package, output_dir / "delivery")

    jobs = pipeline.build_submission_jobs_from_package(
        package,
        provider=SubmissionProvider(args.provider),
    )
    selected_jobs = jobs if args.job_limit <= 0 else jobs[: args.job_limit]
    submission_batch = pipeline.submit_jobs(
        selected_jobs,
        SubmissionTarget(
            backend=SubmissionBackend(args.backend),
            spool_dir=args.spool_dir,
            config_path=args.config_path or "",
            timeout_seconds=args.timeout_seconds,
        ),
        source_type="package",
        source_id=package.scene_id,
    )
    recovery_plan = pipeline.build_recovery_plan(submission_batch)
    recovery_files = pipeline.write_recovery_plan(recovery_plan, output_dir / "recovery")

    for _ in range(max(args.auto_retry_transient, 0)):
        if recovery_plan.queue_paused or not recovery_plan.retry_job_ids:
            break
        retry_jobs = [job for job in selected_jobs if job.job_id in set(recovery_plan.retry_job_ids)]
        if not retry_jobs:
            break
        retry_batch = pipeline.submit_jobs(
            retry_jobs,
            SubmissionTarget(
                backend=SubmissionBackend(args.backend),
                spool_dir=args.spool_dir,
                config_path=args.config_path or "",
                timeout_seconds=args.timeout_seconds,
            ),
            source_type="package_retry",
            source_id=package.scene_id,
        )
        submission_batch = pipeline.merge_submission_batches(submission_batch, retry_batch)
        recovery_plan = pipeline.build_recovery_plan(submission_batch)
        recovery_files = pipeline.write_recovery_plan(recovery_plan, output_dir / "recovery")

    submission_files = pipeline.write_submission_batch(submission_batch, output_dir / "submission")

    downloads = None
    qa_files = {}
    sequence_files = {}
    if not args.skip_download and not recovery_plan.queue_paused:
        downloads = pipeline.download_submission_artifacts(
            submission_batch,
            output_dir / "artifacts",
            config_path=args.config_path,
            timeout_seconds=max(args.timeout_seconds, 900.0),
            poll_interval_seconds=args.poll_interval_seconds,
        )
        pipeline.write_artifact_download_batch(downloads, output_dir / "downloads")
        updated_manifest = pipeline.update_render_manifest_from_downloads(delivery_files["render_manifest_template"], downloads)
        qa_report = pipeline.render_qa_report(package, updated_manifest)
        qa_files = pipeline.write_render_qa_report(qa_report, output_dir / "qa")

        if SubmissionProvider(args.provider) == SubmissionProvider.MINIMAX_VIDEO:
            assembly_plan = pipeline.build_sequence_assembly_plan(package, artifacts_dir=str(output_dir / "artifacts"))
            if not args.skip_assemble and len(selected_jobs) == len(package.video_segments):
                assembly_result = pipeline.assemble_sequence_with_ffmpeg(
                    assembly_plan,
                    output_dir / "delivery" / "assembly",
                    output_path=output_dir / "artifacts" / f"{package.scene_id.lower()}_sequence.mp4",
                )
                sequence_report = pipeline.build_sequence_qa_report(
                    assembly_plan,
                    output_dir / "artifacts",
                    final_sequence_filename=Path(assembly_result.output_path).name if assembly_result.output_path else None,
                )
            else:
                sequence_report = pipeline.build_sequence_qa_report(assembly_plan, output_dir / "artifacts")
            sequence_repair = pipeline.build_sequence_repair_plan(sequence_report)
            sequence_files = pipeline.write_sequence_qc_outputs(sequence_report, sequence_repair, output_dir / "sequence_qc")

    print(
        json.dumps(
            {
                "scene_id": package.scene_id,
                "analysis_source": package.analysis_source,
                "planned_shots": num_shots,
                "submission_backend": submission_batch.backend.value,
                "submission_provider": submission_batch.provider.value,
                "submitted_jobs": len(selected_jobs),
                "downloaded_artifacts": sum(record.downloaded for record in downloads.records) if downloads else 0,
                "queue_paused": recovery_plan.queue_paused,
                "delivery_files": {key: str(value) for key, value in delivery_files.items()},
                "submission_files": {key: str(value) for key, value in submission_files.items()},
                "recovery_files": {key: str(value) for key, value in recovery_files.items()},
                "qa_files": {key: str(value) for key, value in qa_files.items()},
                "sequence_files": {key: str(value) for key, value in sequence_files.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
