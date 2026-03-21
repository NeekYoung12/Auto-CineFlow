"""Project-level aggregation for per-scene render QA."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from .project_delivery import ProjectPackage
from .render_qa import load_render_manifest, render_qa_report


class SceneRenderQASummary(BaseModel):
    """Aggregated render QA summary for a single scene."""

    scene_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    coverage_rate: float = Field(..., ge=0.0, le=1.0)
    passes_gate: bool
    shot_count: int = Field(..., ge=0)
    rendered_shots: int = Field(..., ge=0)
    critical_issues: list[str] = Field(default_factory=list)


class ProjectRenderQAReport(BaseModel):
    """Project-level render QA report."""

    project_name: str
    generated_at: str
    scene_count: int = Field(..., ge=1)
    average_score: float = Field(..., ge=0.0, le=1.0)
    average_coverage_rate: float = Field(..., ge=0.0, le=1.0)
    total_expected_shots: int = Field(..., ge=0)
    total_rendered_shots: int = Field(..., ge=0)
    all_scenes_pass: bool
    failing_scene_ids: list[str] = Field(default_factory=list)
    scene_results: list[SceneRenderQASummary] = Field(default_factory=list)


def _resolve_scene_dir(scenes_dir: Path, scene_id: str) -> Path | None:
    """Find the on-disk directory for a scene export."""

    if not scenes_dir.exists():
        return None

    for candidate in scenes_dir.iterdir():
        if not candidate.is_dir():
            continue
        manifest_path = candidate / "storyboard_package.json"
        if not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("scene_id") == scene_id:
            return candidate
    return None


def build_project_render_qa_report(
    project: ProjectPackage,
    scenes_dir: str | Path,
    min_score: float = 0.9,
) -> ProjectRenderQAReport:
    """Aggregate per-scene render QA into a project-level report."""

    scenes_dir = Path(scenes_dir)
    scene_results: list[SceneRenderQASummary] = []

    for scene in project.scenes:
        scene_dir = _resolve_scene_dir(scenes_dir, scene.scene_id)
        manifest = []
        if scene_dir is not None:
            manifest_path = scene_dir / "render_manifest_template.json"
            if manifest_path.exists():
                manifest = load_render_manifest(manifest_path)

        report = render_qa_report(scene, manifest, min_score=min_score)
        rendered_shots = sum(result.checks.rendered for result in report.shot_results)
        scene_results.append(
            SceneRenderQASummary(
                scene_id=scene.scene_id,
                score=report.score,
                coverage_rate=report.coverage_rate,
                passes_gate=report.passes_gate,
                shot_count=len(report.shot_results),
                rendered_shots=rendered_shots,
                critical_issues=list(report.critical_issues),
            )
        )

    total_expected_shots = sum(result.shot_count for result in scene_results)
    total_rendered_shots = sum(result.rendered_shots for result in scene_results)
    average_score = round(sum(result.score for result in scene_results) / len(scene_results), 4)
    average_coverage_rate = round(sum(result.coverage_rate for result in scene_results) / len(scene_results), 4)
    failing_scene_ids = [result.scene_id for result in scene_results if not result.passes_gate]

    return ProjectRenderQAReport(
        project_name=project.project_name,
        generated_at=datetime.now(timezone.utc).isoformat(),
        scene_count=len(scene_results),
        average_score=average_score,
        average_coverage_rate=average_coverage_rate,
        total_expected_shots=total_expected_shots,
        total_rendered_shots=total_rendered_shots,
        all_scenes_pass=not failing_scene_ids,
        failing_scene_ids=failing_scene_ids,
        scene_results=scene_results,
    )


def project_render_qa_report_json(report: ProjectRenderQAReport, indent: int = 2) -> str:
    """Serialise a project render QA report."""

    return report.model_dump_json(indent=indent)


def project_render_qa_review_markdown(report: ProjectRenderQAReport) -> str:
    """Export a human-readable project render QA review document."""

    lines = [
        f"# Project Render QA {report.project_name}",
        "",
        f"- Scene Count: `{report.scene_count}`",
        f"- Average Score: `{report.average_score:.3f}`",
        f"- Average Coverage: `{report.average_coverage_rate:.3f}`",
        f"- All Scenes Pass: `{report.all_scenes_pass}`",
        f"- Expected Shots: `{report.total_expected_shots}`",
        f"- Rendered Shots: `{report.total_rendered_shots}`",
        "",
        "## Scene Results",
        "",
    ]
    for scene in report.scene_results:
        lines.extend(
            [
                f"### {scene.scene_id}",
                "",
                f"- Score: `{scene.score:.3f}`",
                f"- Coverage: `{scene.coverage_rate:.3f}`",
                f"- Passes Gate: `{scene.passes_gate}`",
                f"- Rendered / Expected: `{scene.rendered_shots}/{scene.shot_count}`",
                f"- Critical Issues: `{', '.join(scene.critical_issues) if scene.critical_issues else 'none'}`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def write_project_render_qa_report(
    report: ProjectRenderQAReport,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write a project render QA report to disk."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    report_json_path = target_dir / "project_render_qa_report.json"
    review_markdown_path = target_dir / "project_render_qa_review.md"
    report_json_path.write_text(project_render_qa_report_json(report, indent=2), encoding="utf-8")
    review_markdown_path.write_text(project_render_qa_review_markdown(report), encoding="utf-8")
    return {
        "report_json": report_json_path,
        "review_markdown": review_markdown_path,
    }


def main() -> int:
    """CLI entry point for project render QA aggregation."""

    import argparse

    from .change_planner import load_project_package

    parser = argparse.ArgumentParser(description="Aggregate render QA across a project.")
    parser.add_argument("--project-file", required=True, help="Path to project manifest JSON")
    parser.add_argument("--scenes-dir", required=True, help="Directory containing per-scene exports")
    parser.add_argument("--output-dir", required=True, help="Directory for project QA outputs")
    parser.add_argument("--min-score", type=float, default=0.9, help="Minimum passing score")
    args = parser.parse_args()

    project = load_project_package(args.project_file)
    report = build_project_render_qa_report(project, args.scenes_dir, min_score=args.min_score)
    output_files = write_project_render_qa_report(report, args.output_dir)
    print(
        json.dumps(
            {
                "project_name": report.project_name,
                "average_score": report.average_score,
                "all_scenes_pass": report.all_scenes_pass,
                "output_files": {key: str(value) for key, value in output_files.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
