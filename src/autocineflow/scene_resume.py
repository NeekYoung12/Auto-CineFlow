"""Resume an existing scene run by downloading missing artifacts and restitching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .delivery import StoryboardPackage
from .pipeline import CineFlowPipeline
from .submission import SubmissionBatch, SubmissionProvider


def main() -> int:
    """CLI entry point for resuming a previously generated scene run."""

    parser = argparse.ArgumentParser(description="Resume a scene run by fetching missing artifacts and restitching.")
    parser.add_argument("--run-dir", required=True, help="Root output directory of a previous scene run")
    parser.add_argument("--config-path", default=None, help="Path to config file")
    parser.add_argument("--poll-interval-seconds", type=float, default=10.0, help="Artifact polling interval")
    parser.add_argument("--timeout-seconds", type=float, default=900.0, help="Polling timeout")
    parser.add_argument("--skip-assemble", action="store_true", help="Do not auto-stitch even if all clips are present")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    pipeline = CineFlowPipeline(config_path=args.config_path)

    package = StoryboardPackage.model_validate_json((run_dir / "delivery" / "storyboard_package.json").read_text(encoding="utf-8"))
    submission_batch = SubmissionBatch.model_validate_json((run_dir / "submission" / "submission_batch.json").read_text(encoding="utf-8"))

    downloads = pipeline.download_submission_artifacts(
        submission_batch,
        run_dir / "artifacts",
        config_path=args.config_path,
        timeout_seconds=args.timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        skip_existing=True,
    )
    pipeline.write_artifact_download_batch(downloads, run_dir / "downloads")

    updated_manifest = pipeline.update_render_manifest_from_downloads(run_dir / "delivery" / "render_manifest_template.json", downloads)
    qa_report = pipeline.render_qa_report(package, updated_manifest)
    qa_files = pipeline.write_render_qa_report(qa_report, run_dir / "qa")

    sequence_files = {}
    if submission_batch.provider == SubmissionProvider.MINIMAX_VIDEO:
        assembly_plan = pipeline.build_sequence_assembly_plan(package, artifacts_dir=str(run_dir / "artifacts"))
        all_clip_assets = all((run_dir / "artifacts" / f"{clip.shot_id}.mp4").exists() for clip in assembly_plan.clips)
        final_sequence_name = None
        if all_clip_assets and not args.skip_assemble:
            assembly_result = pipeline.assemble_sequence_with_ffmpeg(
                assembly_plan,
                run_dir / "delivery" / "assembly",
                output_path=run_dir / "artifacts" / f"{package.scene_id.lower()}_sequence.mp4",
            )
            if assembly_result.output_path:
                final_sequence_name = Path(assembly_result.output_path).name
        sequence_report = pipeline.build_sequence_qa_report(
            assembly_plan,
            run_dir / "artifacts",
            final_sequence_filename=final_sequence_name,
        )
        sequence_repair = pipeline.build_sequence_repair_plan(sequence_report)
        sequence_files = pipeline.write_sequence_qc_outputs(sequence_report, sequence_repair, run_dir / "sequence_qc")

    print(
        json.dumps(
            {
                "scene_id": package.scene_id,
                "downloaded_artifacts": sum(record.downloaded for record in downloads.records),
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
