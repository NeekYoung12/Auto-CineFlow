"""Asset library indexing for generated scene and project outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .delivery import StoryboardPackage
from .project_delivery import ProjectPackage


class SceneAssetVersion(BaseModel):
    """Indexed scene-level asset version."""

    scene_id: str
    project_name: str
    generated_at: str
    manifest_path: str
    output_dir: str
    analysis_source: str
    detected_emotion: str
    total_duration_seconds: float = Field(..., ge=0.1)
    shot_count: int = Field(..., ge=1)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    render_qa_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    sequence_qa_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    submission_count: int = Field(default=0, ge=0)
    failed_submission_count: int = Field(default=0, ge=0)
    provider_status_summary: list[str] = Field(default_factory=list)
    provider: str = ""
    available_files: list[str] = Field(default_factory=list)


class ProjectAssetVersion(BaseModel):
    """Indexed project-level asset version."""

    project_name: str
    generated_at: str
    manifest_path: str
    output_dir: str
    scene_count: int = Field(..., ge=1)
    total_duration_seconds: float = Field(..., ge=0.1)
    average_quality_score: float = Field(..., ge=0.0, le=1.0)
    average_render_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reuse_count: int = Field(default=0, ge=0)
    rerender_count: int = Field(default=0, ge=0)
    total_failed_submissions: int = Field(default=0, ge=0)
    available_files: list[str] = Field(default_factory=list)


class AssetLibrary(BaseModel):
    """Indexed view of all generated scene and project assets under a root."""

    root_dir: str
    scene_versions: list[SceneAssetVersion] = Field(default_factory=list)
    project_versions: list[ProjectAssetVersion] = Field(default_factory=list)


def _maybe_json(path: Path) -> dict | None:
    """Read a JSON file if it exists."""

    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _scene_files(base_dir: Path) -> list[str]:
    """Return stable relative file names for a scene output directory."""

    names: list[str] = []
    for path in sorted(base_dir.rglob("*")):
        if path.is_file():
            names.append(str(path.relative_to(base_dir)).replace("\\", "/"))
    return names


def index_scene_asset_version(manifest_path: str | Path) -> SceneAssetVersion:
    """Index one scene manifest directory into a scene asset version record."""

    manifest_path = Path(manifest_path)
    package = StoryboardPackage.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent

    render_qa_payload = _maybe_json(base_dir.parent / "qa" / "render_qa_report.json")
    sequence_qa_payload = _maybe_json(base_dir.parent / "sequence_qc" / "sequence_qa_report.json")
    submission_payload = _maybe_json(base_dir.parent / "submission" / "submission_batch.json")

    provider = ""
    submission_count = 0
    failed_submission_count = 0
    provider_status_summary: list[str] = []
    if submission_payload:
        provider = str(submission_payload.get("provider", ""))
        submission_count = int(submission_payload.get("job_count", 0))
        records = submission_payload.get("records", [])
        failed_submission_count = sum(1 for record in records if not record.get("backend_job_id"))
        provider_status_summary = sorted(
            {
                f"{record.get('provider_status_code', 0)}:{record.get('provider_status_message', '')}"
                for record in records
                if record.get("provider_status_code") or record.get("provider_status_message")
            }
        )

    return SceneAssetVersion(
        scene_id=package.scene_id,
        project_name=package.project_name,
        generated_at=package.generated_at,
        manifest_path=str(manifest_path),
        output_dir=str(base_dir.parent),
        analysis_source=package.analysis_source,
        detected_emotion=package.detected_emotion,
        total_duration_seconds=package.total_duration_seconds,
        shot_count=len(package.shots),
        quality_score=float(package.quality_report.get("score", 0.0)),
        render_qa_score=float(render_qa_payload.get("score")) if render_qa_payload and "score" in render_qa_payload else None,
        sequence_qa_score=float(sequence_qa_payload.get("score")) if sequence_qa_payload and "score" in sequence_qa_payload else None,
        submission_count=submission_count,
        failed_submission_count=failed_submission_count,
        provider_status_summary=provider_status_summary,
        provider=provider,
        available_files=_scene_files(base_dir.parent),
    )


def index_project_asset_version(manifest_path: str | Path) -> ProjectAssetVersion:
    """Index one project manifest directory into a project asset version record."""

    manifest_path = Path(manifest_path)
    package = ProjectPackage.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent

    project_render_qa = _maybe_json(base_dir / "project_qa" / "project_render_qa_report.json")
    execution_payload = _maybe_json(base_dir / "execution" / "project_execution_plan.json")
    dashboard_payload = _maybe_json(base_dir / "dashboard" / "project_dashboard.json")

    reuse_count = 0
    rerender_count = 0
    total_failed_submissions = 0
    if dashboard_payload:
        reuse_count = int(dashboard_payload.get("total_reuse_count", 0))
        rerender_count = int(dashboard_payload.get("total_rerender_count", 0))
    elif execution_payload:
        reuse_count = len(execution_payload.get("reuse_manifest", []))
        rerender_count = len(execution_payload.get("rerender_queue", []))

    scenes_dir = base_dir / "scenes"
    for manifest in scenes_dir.glob("*/storyboard_package.json"):
        scene_version = index_scene_asset_version(manifest)
        total_failed_submissions += scene_version.failed_submission_count

    return ProjectAssetVersion(
        project_name=package.project_name,
        generated_at=package.generated_at,
        manifest_path=str(manifest_path),
        output_dir=str(base_dir),
        scene_count=package.scene_count,
        total_duration_seconds=package.total_duration_seconds,
        average_quality_score=package.average_quality_score,
        average_render_score=(
            float(project_render_qa.get("average_score"))
            if project_render_qa and "average_score" in project_render_qa
            else None
        ),
        reuse_count=reuse_count,
        rerender_count=rerender_count,
        total_failed_submissions=total_failed_submissions,
        available_files=_scene_files(base_dir),
    )


def build_asset_library(root_dir: str | Path) -> AssetLibrary:
    """Scan a root output directory and index all scene/project assets."""

    root_dir = Path(root_dir)
    scene_versions: list[SceneAssetVersion] = []
    project_versions: list[ProjectAssetVersion] = []

    for manifest_path in root_dir.rglob("storyboard_package.json"):
        scene_versions.append(index_scene_asset_version(manifest_path))

    for manifest_path in root_dir.rglob("project_manifest.json"):
        project_versions.append(index_project_asset_version(manifest_path))

    scene_versions.sort(key=lambda item: (item.scene_id, item.generated_at))
    project_versions.sort(key=lambda item: (item.project_name, item.generated_at))

    return AssetLibrary(
        root_dir=str(root_dir),
        scene_versions=scene_versions,
        project_versions=project_versions,
    )


def latest_scene_versions(library: AssetLibrary) -> list[SceneAssetVersion]:
    """Return the newest version for each scene ID."""

    latest: dict[str, SceneAssetVersion] = {}
    for item in library.scene_versions:
        latest[item.scene_id] = item
    return sorted(latest.values(), key=lambda item: item.scene_id)


def latest_project_versions(library: AssetLibrary) -> list[ProjectAssetVersion]:
    """Return the newest version for each project name."""

    latest: dict[str, ProjectAssetVersion] = {}
    for item in library.project_versions:
        latest[item.project_name] = item
    return sorted(latest.values(), key=lambda item: item.project_name)


def asset_library_json(library: AssetLibrary, indent: int = 2) -> str:
    """Serialise an asset library to JSON."""

    return library.model_dump_json(indent=indent)


def asset_library_markdown(library: AssetLibrary) -> str:
    """Export a human-readable asset library summary."""

    lines = [
        f"# Asset Library {library.root_dir}",
        "",
        f"- Scene Versions: `{len(library.scene_versions)}`",
        f"- Project Versions: `{len(library.project_versions)}`",
        "",
        "## Latest Scenes",
        "",
    ]
    for scene in latest_scene_versions(library):
        lines.extend(
            [
                f"### {scene.scene_id}",
                "",
                f"- Project: `{scene.project_name}`",
                f"- Generated At: `{scene.generated_at}`",
                f"- Quality: `{scene.quality_score:.3f}`",
                f"- Render QA: `{scene.render_qa_score if scene.render_qa_score is not None else 'n/a'}`",
                f"- Sequence QA: `{scene.sequence_qa_score if scene.sequence_qa_score is not None else 'n/a'}`",
                f"- Failed Submissions: `{scene.failed_submission_count}`",
                "",
            ]
        )

    lines.extend(["", "## Latest Projects", ""])
    for project in latest_project_versions(library):
        lines.extend(
            [
                f"### {project.project_name}",
                "",
                f"- Generated At: `{project.generated_at}`",
                f"- Scenes: `{project.scene_count}`",
                f"- Avg Quality: `{project.average_quality_score:.3f}`",
                f"- Avg Render QA: `{project.average_render_score if project.average_render_score is not None else 'n/a'}`",
                f"- Reuse / Rerender: `{project.reuse_count}/{project.rerender_count}`",
                f"- Failed Submissions: `{project.total_failed_submissions}`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def write_asset_library(library: AssetLibrary, output_dir: str | Path) -> dict[str, Path]:
    """Write asset library outputs to disk."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "asset_library.json"
    markdown_path = output_dir / "asset_library.md"
    json_path.write_text(asset_library_json(library, indent=2), encoding="utf-8")
    markdown_path.write_text(asset_library_markdown(library), encoding="utf-8")
    return {
        "library_json": json_path,
        "library_markdown": markdown_path,
    }


def main() -> int:
    """CLI entry point for asset library indexing."""

    import argparse

    parser = argparse.ArgumentParser(description="Build an asset library from generated outputs.")
    parser.add_argument("--root-dir", default="out", help="Root output directory to scan")
    parser.add_argument("--output-dir", required=True, help="Directory for asset library outputs")
    args = parser.parse_args()

    library = build_asset_library(args.root_dir)
    output_files = write_asset_library(library, args.output_dir)
    print(
        json.dumps(
            {
                "scene_versions": len(library.scene_versions),
                "project_versions": len(library.project_versions),
                "output_files": {key: str(value) for key, value in output_files.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
