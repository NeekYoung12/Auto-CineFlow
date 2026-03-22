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
    parser.add_argument("--skip-download", action="store_true", help="Do not download URL artifacts")
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
    submission_files = pipeline.write_submission_batch(submission_batch, output_dir / "submission")

    downloads = None
    qa_files = {}
    if not args.skip_download:
        downloads = pipeline.download_submission_artifacts(
            submission_batch,
            output_dir / "artifacts",
            config_path=args.config_path,
        )
        pipeline.write_artifact_download_batch(downloads, output_dir / "downloads")
        updated_manifest = pipeline.update_render_manifest_from_downloads(delivery_files["render_manifest_template"], downloads)
        qa_report = pipeline.render_qa_report(package, updated_manifest)
        qa_files = pipeline.write_render_qa_report(qa_report, output_dir / "qa")

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
                "delivery_files": {key: str(value) for key, value in delivery_files.items()},
                "submission_files": {key: str(value) for key, value in submission_files.items()},
                "qa_files": {key: str(value) for key, value in qa_files.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
