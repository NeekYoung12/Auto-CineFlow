"""Project-level packaging for multiple storyboard scenes."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .delivery import StoryboardPackage, slugify, write_storyboard_package
from .provider_payloads import write_provider_payloads
from .render_qa import render_manifest_template_json


class ProjectSceneInput(BaseModel):
    """Single scene request inside a batch project."""

    scene_id: str
    description: str
    num_shots: int = Field(default=5, ge=1)
    emotion_override: Optional[str] = None


class ProjectSceneSummary(BaseModel):
    """Quality and output summary for a packaged scene."""

    scene_id: str
    scene_location: str = ""
    analysis_source: str
    quality_score: float = Field(..., ge=0.0, le=1.0)
    passes_gate: bool
    readiness_report: dict[str, bool] = Field(default_factory=dict)
    quality_report: dict[str, Any] = Field(default_factory=dict)
    total_duration_seconds: float = Field(..., ge=0.1)


class ProjectPackage(BaseModel):
    """Project-level package combining multiple storyboard scenes."""

    project_name: str
    generated_at: str
    scene_count: int = Field(..., ge=1)
    total_duration_seconds: float = Field(..., ge=0.1)
    average_quality_score: float = Field(..., ge=0.0, le=1.0)
    all_scenes_ready: bool
    scenes: list[StoryboardPackage] = Field(default_factory=list)
    scene_summaries: list[ProjectSceneSummary] = Field(default_factory=list)


def build_project_package(
    project_name: str,
    scenes: list[StoryboardPackage],
) -> ProjectPackage:
    """Aggregate scene packages into a project-level package."""

    scene_summaries = [
        ProjectSceneSummary(
            scene_id=scene.scene_id,
            scene_location=scene.scene_location,
            analysis_source=scene.analysis_source,
            quality_score=float(scene.quality_report.get("score", 0.0)),
            passes_gate=bool(scene.quality_report.get("passes_gate", False)),
            readiness_report=dict(scene.readiness_report),
            quality_report=dict(scene.quality_report),
            total_duration_seconds=scene.total_duration_seconds,
        )
        for scene in scenes
    ]
    total_duration_seconds = round(sum(scene.total_duration_seconds for scene in scenes), 2) or 0.1
    average_quality_score = round(
        sum(summary.quality_score for summary in scene_summaries) / len(scene_summaries),
        4,
    )

    return ProjectPackage(
        project_name=project_name,
        generated_at=datetime.now(timezone.utc).isoformat(),
        scene_count=len(scenes),
        total_duration_seconds=total_duration_seconds,
        average_quality_score=average_quality_score,
        all_scenes_ready=all(summary.passes_gate for summary in scene_summaries),
        scenes=scenes,
        scene_summaries=scene_summaries,
    )


def project_manifest_json(package: ProjectPackage, indent: int = 2) -> str:
    """Serialise a project package to JSON."""

    return package.model_dump_json(indent=indent)


def project_shotlist_csv(package: ProjectPackage) -> str:
    """Flatten all scene shot lists into one CSV."""

    buffer = StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "scene_id",
            "shot_id",
            "shot_slug",
            "shot_index",
            "beat_type",
            "shot_type",
            "duration_seconds",
            "frame_count",
            "timeline_in",
            "timeline_out",
            "render_seed",
            "continuity_group",
            "reference_shot_id",
            "primary_subjects",
            "scene_location",
            "story_purpose",
        ],
    )
    writer.writeheader()
    for scene in package.scenes:
        for shot in scene.shots:
            writer.writerow(
                {
                    "scene_id": scene.scene_id,
                    "shot_id": shot.shot_id,
                    "shot_slug": shot.shot_slug,
                    "shot_index": shot.shot_index,
                    "beat_type": shot.beat_type,
                    "shot_type": shot.shot_type,
                    "duration_seconds": shot.duration_seconds,
                    "frame_count": shot.frame_count,
                    "timeline_in": shot.timeline_in,
                    "timeline_out": shot.timeline_out,
                    "render_seed": shot.render_seed,
                    "continuity_group": shot.continuity_group,
                    "reference_shot_id": shot.reference_shot_id,
                    "primary_subjects": "|".join(shot.primary_subjects),
                    "scene_location": shot.scene_location,
                    "story_purpose": shot.story_purpose,
                }
            )
    return buffer.getvalue()


def project_review_markdown(package: ProjectPackage) -> str:
    """Export a project-level review markdown document."""

    lines = [
        f"# {package.project_name}",
        "",
        f"- Scenes: `{package.scene_count}`",
        f"- Total Duration: `{package.total_duration_seconds:.2f}s`",
        f"- Average Quality Score: `{package.average_quality_score:.3f}`",
        f"- All Scenes Ready: `{package.all_scenes_ready}`",
        "",
        "## Scene Summary",
        "",
    ]
    for summary in package.scene_summaries:
        lines.extend(
            [
                f"### {summary.scene_id}",
                "",
                f"- Location: `{summary.scene_location or 'unspecified'}`",
                f"- Analysis Source: `{summary.analysis_source}`",
                f"- Quality Score: `{summary.quality_score:.3f}`",
                f"- Passes Gate: `{summary.passes_gate}`",
                f"- Duration: `{summary.total_duration_seconds:.2f}s`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def write_project_package(
    package: ProjectPackage,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write a project package and all scene packages to disk."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = target_dir / "project_manifest.json"
    shotlist_path = target_dir / "project_shotlist.csv"
    review_path = target_dir / "project_review.md"
    scenes_dir = target_dir / "scenes"
    scenes_dir.mkdir(exist_ok=True)

    manifest_path.write_text(project_manifest_json(package, indent=2), encoding="utf-8")
    shotlist_path.write_text(project_shotlist_csv(package), encoding="utf-8", newline="")
    review_path.write_text(project_review_markdown(package), encoding="utf-8")

    for scene in package.scenes:
        scene_dir = scenes_dir / slugify(scene.scene_id, default=scene.scene_id.lower())
        write_storyboard_package(scene, scene_dir)
        write_provider_payloads(scene, scene_dir / "providers")
        (scene_dir / "render_manifest_template.json").write_text(
            render_manifest_template_json(scene, indent=2),
            encoding="utf-8",
        )

    return {
        "manifest": manifest_path,
        "shotlist": shotlist_path,
        "review_markdown": review_path,
        "scenes_dir": scenes_dir,
    }


def load_scene_inputs(path: str | Path) -> list[ProjectSceneInput]:
    """Load scene inputs from a JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        raw_scenes = payload.get("scenes", [])
    else:
        raw_scenes = payload
    return [ProjectSceneInput.model_validate(scene) for scene in raw_scenes]
