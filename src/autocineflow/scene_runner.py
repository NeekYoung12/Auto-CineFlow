"""One-shot orchestration for scene generation, submission, download, and QA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config_loader import resolve_runninghub_workflow_ids
from .delivery import RenderPreset
from .pipeline import CineFlowPipeline
from .submission import SubmissionBackend, SubmissionJob, SubmissionProvider, SubmissionTarget


def _is_runninghub_video_provider(provider: SubmissionProvider) -> bool:
    return provider in {
        SubmissionProvider.RUNNINGHUB_VIDEO_AUTO,
        SubmissionProvider.RUNNINGHUB_VIDEO_QUALITY,
        SubmissionProvider.RUNNINGHUB_VIDEO_FAST,
    }


def _inject_bootstrap_keyframes(
    video_jobs: list[SubmissionJob],
    downloads,
) -> list[SubmissionJob]:
    """Prefer freshly rendered keyframes as the first-frame candidate for RunningHub video jobs."""

    download_by_shot = {
        record.shot_id: record.output_path
        for record in downloads.records
        if record.downloaded and record.output_path
    }
    patched_jobs: list[SubmissionJob] = []
    for job in video_jobs:
        output_path = download_by_shot.get(job.shot_id)
        if not output_path:
            patched_jobs.append(job)
            continue

        payload = dict(job.payload)
        contract = dict(payload.get("request_contract", {}))
        existing_candidates = list(contract.get("first_frame_candidates", []))
        merged_candidates = [output_path, *[candidate for candidate in existing_candidates if candidate != output_path]]
        contract["first_frame_candidates"] = merged_candidates
        payload["request_contract"] = contract
        patched_jobs.append(job.model_copy(update={"payload": payload}))
    return patched_jobs


def _build_rebuild_keyframe_jobs(
    keyframe_jobs: list[SubmissionJob],
    downloads,
) -> list[SubmissionJob]:
    """Create a second-pass keyframe rebuild using bootstrap outputs as init references."""

    download_by_shot = {
        record.shot_id: record.output_path
        for record in downloads.records
        if record.downloaded and record.output_path
    }
    rebuilt_jobs: list[SubmissionJob] = []
    for job in keyframe_jobs:
        bootstrap_path = download_by_shot.get(job.shot_id)
        if not bootstrap_path:
            continue
        payload = dict(job.payload)
        inputs = dict(payload.get("workflow_inputs", {}))
        positive_prompt = str(inputs.get("positive_prompt", "") or "")
        if "high fidelity facial detail" not in positive_prompt:
            positive_prompt = f"{positive_prompt}, high fidelity facial detail, crisp garment texture, clean signage-free background"
        scene_refs = [bootstrap_path, *[path for path in inputs.get("scene_reference_images", []) if path != bootstrap_path]]
        inputs["scene_reference_images"] = scene_refs[:4]
        inputs["positive_prompt"] = positive_prompt
        inputs["steps"] = max(int(inputs.get("steps", 30) or 30), 42)
        payload["workflow_inputs"] = inputs
        rebuilt_jobs.append(
            job.model_copy(
                update={
                    "job_id": f"{job.job_id}_REBUILD",
                    "payload": payload,
                }
            )
        )
    return rebuilt_jobs


def _video_provider_needs_enhancement(provider: SubmissionProvider) -> bool:
    return provider in {
        SubmissionProvider.MINIMAX_VIDEO,
        SubmissionProvider.RUNNINGHUB_VIDEO_AUTO,
        SubmissionProvider.RUNNINGHUB_VIDEO_QUALITY,
        SubmissionProvider.RUNNINGHUB_VIDEO_FAST,
    }


def _is_sequence_video_provider(provider: SubmissionProvider) -> bool:
    return provider == SubmissionProvider.MINIMAX_VIDEO or _is_runninghub_video_provider(provider)


def _runninghub_ai_post_enhance_available(config_path: str | None) -> bool:
    workflow_ids = resolve_runninghub_workflow_ids(config_path)
    return bool(workflow_ids.get("RUNNINGHUB_WORKFLOW_RH_VIDEO_POST_ENHANCE_V1"))


def _build_runninghub_post_enhance_jobs(downloads) -> list[SubmissionJob]:
    jobs: list[SubmissionJob] = []
    for record in downloads.records:
        if not record.downloaded or not record.output_path.lower().endswith(".mp4"):
            continue
        jobs.append(
            SubmissionJob(
                job_id=f"{record.shot_id}_POST_ENH",
                shot_id=record.shot_id,
                scene_id=downloads.source_id,
                provider=SubmissionProvider.RUNNINGHUB_VIDEO_AUTO,
                payload={
                    "workflow_key": "rh_video_post_enhance_v1",
                    "workflow_id_env": "RUNNINGHUB_WORKFLOW_RH_VIDEO_POST_ENHANCE_V1",
                    "source_video_path": record.output_path,
                },
            )
        )
    return jobs


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
    parser.add_argument("--reference-root", action="append", default=[], help="Optional local reference asset root; may be repeated")
    parser.add_argument("--disable-consistency-rag", action="store_true", help="Disable local reference retrieval and consistency packaging")
    parser.add_argument("--character-reference-top-k", type=int, default=3, help="Top character candidates to keep per role")
    parser.add_argument("--scene-reference-top-k", type=int, default=4, help="Top scene candidates to keep")
    parser.add_argument("--disable-runninghub-bootstrap", action="store_true", help="Do not generate RunningHub bootstrap keyframes before RunningHub video jobs")
    parser.add_argument("--disable-runninghub-keyframe-rebuild", action="store_true", help="Do not run the second-pass high-fidelity keyframe rebuild before video generation")
    parser.add_argument("--disable-video-enhance", action="store_true", help="Do not run local FFmpeg-based enhancement after video download")
    parser.add_argument("--disable-runninghub-ai-post", action="store_true", help="Do not run the optional RunningHub AI video post-enhancement stage")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    pipeline = CineFlowPipeline(config_path=args.config_path)
    num_shots = args.num_shots
    if args.target_duration_seconds > 0:
        num_shots = pipeline.recommended_video_shot_count(
            target_duration_seconds=args.target_duration_seconds,
            clip_duration_seconds=args.clip_duration_seconds,
        )
    reference_roots = None
    if not args.disable_consistency_rag:
        reference_roots = args.reference_root or [str(path) for path in pipeline.default_reference_roots()]

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
        reference_roots=reference_roots,
        character_reference_top_k=args.character_reference_top_k,
        scene_reference_top_k=args.scene_reference_top_k,
    )
    delivery_files = pipeline.write_delivery_package(package, output_dir / "delivery")

    jobs = pipeline.build_submission_jobs_from_package(
        package,
        provider=SubmissionProvider(args.provider),
    )
    selected_jobs = jobs if args.job_limit <= 0 else jobs[: args.job_limit]
    bootstrap_files = {}
    provider = SubmissionProvider(args.provider)
    backend = SubmissionBackend(args.backend)
    target = SubmissionTarget(
        backend=backend,
        spool_dir=args.spool_dir,
        config_path=args.config_path or "",
        timeout_seconds=args.timeout_seconds,
    )

    if (
        _is_runninghub_video_provider(provider)
        and backend == SubmissionBackend.RUNNINGHUB_API
        and not args.disable_runninghub_bootstrap
        and not args.skip_download
    ):
        keyframe_jobs = [
            job
            for job in pipeline.build_submission_jobs_from_package(package, provider=SubmissionProvider.RUNNINGHUB_FACEID)
            if job.shot_id in {item.shot_id for item in selected_jobs}
        ]
        if keyframe_jobs:
            keyframe_batch = pipeline.submit_jobs(
                keyframe_jobs,
                target,
                source_type="package_bootstrap_keyframes",
                source_id=package.scene_id,
            )
            bootstrap_files.update(
                {
                    f"bootstrap_submission_{key}": str(value)
                    for key, value in pipeline.write_submission_batch(
                        keyframe_batch,
                        output_dir / "bootstrap_keyframes" / "submission",
                    ).items()
                }
            )
            bootstrap_downloads = pipeline.download_submission_artifacts(
                keyframe_batch,
                output_dir / "bootstrap_keyframes" / "artifacts",
                config_path=args.config_path,
                timeout_seconds=max(args.timeout_seconds, 900.0),
                poll_interval_seconds=args.poll_interval_seconds,
            )
            bootstrap_files.update(
                {
                    f"bootstrap_download_{key}": str(value)
                    for key, value in pipeline.write_artifact_download_batch(
                        bootstrap_downloads,
                        output_dir / "bootstrap_keyframes" / "downloads",
                    ).items()
                }
            )
            rebuild_downloads = bootstrap_downloads
            if not args.disable_runninghub_keyframe_rebuild:
                rebuild_jobs = _build_rebuild_keyframe_jobs(keyframe_jobs, bootstrap_downloads)
                if rebuild_jobs:
                    rebuild_batch = pipeline.submit_jobs(
                        rebuild_jobs,
                        target,
                        source_type="package_rebuild_keyframes",
                        source_id=package.scene_id,
                    )
                    bootstrap_files.update(
                        {
                            f"rebuild_submission_{key}": str(value)
                            for key, value in pipeline.write_submission_batch(
                                rebuild_batch,
                                output_dir / "rebuild_keyframes" / "submission",
                            ).items()
                        }
                    )
                    rebuild_downloads = pipeline.download_submission_artifacts(
                        rebuild_batch,
                        output_dir / "rebuild_keyframes" / "artifacts",
                        config_path=args.config_path,
                        timeout_seconds=max(args.timeout_seconds, 900.0),
                        poll_interval_seconds=args.poll_interval_seconds,
                    )
                    bootstrap_files.update(
                        {
                            f"rebuild_download_{key}": str(value)
                            for key, value in pipeline.write_artifact_download_batch(
                                rebuild_downloads,
                                output_dir / "rebuild_keyframes" / "downloads",
                            ).items()
                        }
                    )
            selected_jobs = _inject_bootstrap_keyframes(selected_jobs, rebuild_downloads)

    submission_batch = pipeline.submit_jobs(
        selected_jobs,
        target,
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
    enhance_files = {}
    ai_post_files = {}
    if not args.skip_download and not recovery_plan.queue_paused:
        downloads = pipeline.download_submission_artifacts(
            submission_batch,
            output_dir / "artifacts",
            config_path=args.config_path,
            timeout_seconds=max(args.timeout_seconds, 900.0),
            poll_interval_seconds=args.poll_interval_seconds,
        )
        pipeline.write_artifact_download_batch(downloads, output_dir / "downloads")
        enhancement_dir = output_dir / "enhanced_artifacts"
        sequence_source_dir = output_dir / "artifacts"
        if (
            _is_runninghub_video_provider(provider)
            and backend == SubmissionBackend.RUNNINGHUB_API
            and not args.disable_runninghub_ai_post
            and _runninghub_ai_post_enhance_available(args.config_path)
        ):
            ai_jobs = _build_runninghub_post_enhance_jobs(downloads)
            if ai_jobs:
                ai_batch = pipeline.submit_jobs(
                    ai_jobs,
                    target,
                    source_type="package_video_post_enhance",
                    source_id=package.scene_id,
                )
                ai_post_files.update(
                    {
                        f"ai_post_submission_{key}": str(value)
                        for key, value in pipeline.write_submission_batch(
                            ai_batch,
                            output_dir / "video_post_enhance" / "submission",
                        ).items()
                    }
                )
                ai_downloads = pipeline.download_submission_artifacts(
                    ai_batch,
                    output_dir / "video_post_enhance" / "artifacts",
                    config_path=args.config_path,
                    timeout_seconds=max(args.timeout_seconds, 900.0),
                    poll_interval_seconds=args.poll_interval_seconds,
                )
                ai_post_files.update(
                    {
                        f"ai_post_download_{key}": str(value)
                        for key, value in pipeline.write_artifact_download_batch(
                            ai_downloads,
                            output_dir / "video_post_enhance" / "downloads",
                        ).items()
                    }
                )
                ai_map = {
                    record.shot_id: record.output_path
                    for record in ai_downloads.records
                    if record.downloaded and record.output_path
                }
                for record in downloads.records:
                    if record.shot_id in ai_map:
                        record.output_path = ai_map[record.shot_id]
                if ai_map:
                    sequence_source_dir = output_dir / "video_post_enhance" / "artifacts"
        if _video_provider_needs_enhancement(provider) and not args.disable_video_enhance:
            enhance_report = pipeline.enhance_videos(downloads, enhancement_dir)
            enhance_files = pipeline.write_video_enhance_report(enhance_report, output_dir / "enhance")
            if any(result.enhanced for result in enhance_report.results):
                enhanced_map = {
                    result.shot_id: result.output_path
                    for result in enhance_report.results
                    if result.enhanced and result.output_path
                }
                for record in downloads.records:
                    if record.shot_id in enhanced_map:
                        record.output_path = enhanced_map[record.shot_id]
                sequence_source_dir = enhancement_dir

        updated_manifest = pipeline.update_render_manifest_from_downloads(delivery_files["render_manifest_template"], downloads)
        qa_report = pipeline.render_qa_report(package, updated_manifest)
        qa_files = pipeline.write_render_qa_report(qa_report, output_dir / "qa")

        if _is_sequence_video_provider(provider):
            assembly_plan = pipeline.build_sequence_assembly_plan(package, artifacts_dir=str(sequence_source_dir))
            if not args.skip_assemble and len(selected_jobs) == len(package.video_segments):
                assembly_result = pipeline.assemble_sequence_with_ffmpeg(
                    assembly_plan,
                    output_dir / "delivery" / "assembly",
                    output_path=sequence_source_dir / f"{package.scene_id.lower()}_sequence.mp4",
                )
                sequence_report = pipeline.build_sequence_qa_report(
                    assembly_plan,
                    sequence_source_dir,
                    final_sequence_filename=Path(assembly_result.output_path).name if assembly_result.output_path else None,
                )
            else:
                sequence_report = pipeline.build_sequence_qa_report(assembly_plan, sequence_source_dir)
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
                "bootstrap_files": bootstrap_files,
                "delivery_files": {key: str(value) for key, value in delivery_files.items()},
                "submission_files": {key: str(value) for key, value in submission_files.items()},
                "recovery_files": {key: str(value) for key, value in recovery_files.items()},
                "ai_post_files": {key: str(value) for key, value in ai_post_files.items()},
                "enhance_files": {key: str(value) for key, value in enhance_files.items()},
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
