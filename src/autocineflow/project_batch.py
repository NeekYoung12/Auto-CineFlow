"""CLI for project-scale batch packaging."""

from __future__ import annotations

import argparse
import json

from .pipeline import CineFlowPipeline
from .project_delivery import load_scene_inputs


def main() -> int:
    """CLI entry point for batch project processing."""

    parser = argparse.ArgumentParser(description="Batch-package multiple Auto-CineFlow scenes.")
    parser.add_argument("--input-file", required=True, help="JSON file containing scene inputs")
    parser.add_argument("--project-name", default="Auto-CineFlow Project", help="Project name")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--config-path", default=None, help="Path to OpenAI-compatible config")
    parser.add_argument("--offline", action="store_true", help="Disable LLM analysis")
    parser.add_argument("--model", default="gpt-5", help="LLM model name")
    parser.add_argument("--min-score", type=float, default=0.85, help="Quality gate threshold")
    parser.add_argument("--max-attempts", type=int, default=3, help="Maximum retry attempts per scene")
    args = parser.parse_args()

    pipeline = CineFlowPipeline(config_path=args.config_path, model=args.model)
    scene_inputs = load_scene_inputs(args.input_file)
    project_package = pipeline.build_project_package(
        scene_inputs=scene_inputs,
        project_name=args.project_name,
        use_llm=not args.offline,
        min_score=args.min_score,
        max_attempts=args.max_attempts,
    )
    output_files = pipeline.write_project_package(project_package, args.output_dir)

    print(
        json.dumps(
            {
                "project_name": project_package.project_name,
                "scene_count": project_package.scene_count,
                "all_scenes_ready": project_package.all_scenes_ready,
                "output_files": {key: str(value) for key, value in output_files.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
