"""Incremental rerender planning for storyboard packages and projects."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from .delivery import StoryboardPackage
from .project_delivery import ProjectPackage
from .render_qa import prompt_hash


class ShotChangeType(str, Enum):
    """How a shot changed between two package versions."""

    UNCHANGED = "UNCHANGED"
    NEW_RENDER = "NEW_RENDER"
    RERENDER = "RERENDER"
    REMOVED = "REMOVED"


class ShotChangeRecord(BaseModel):
    """Change record for a single shot."""

    scene_id: str
    shot_id: str
    change_type: ShotChangeType
    reasons: list[str] = Field(default_factory=list)
    continuity_group: str = ""
    previous_reference_shot_id: str = ""
    current_reference_shot_id: str = ""


class SceneChangeSummary(BaseModel):
    """Per-scene summary of shot changes."""

    scene_id: str
    added_shots: int = 0
    rerender_shots: int = 0
    unchanged_shots: int = 0
    removed_shots: int = 0
    passes_incremental_reuse: bool = False


class ProjectChangePlan(BaseModel):
    """Project-level incremental rerender plan."""

    generated_at: str
    previous_project_name: str
    current_project_name: str
    scene_summaries: list[SceneChangeSummary] = Field(default_factory=list)
    shot_changes: list[ShotChangeRecord] = Field(default_factory=list)
    rerender_queue: list[dict] = Field(default_factory=list)


def _scene_map(project: ProjectPackage) -> dict[str, StoryboardPackage]:
    """Index scene packages by scene ID."""

    return {scene.scene_id: scene for scene in project.scenes}


def _render_signature(package: StoryboardPackage, shot_id: str) -> tuple | None:
    """Build a render-critical signature for a shot."""

    shot = next((item for item in package.shots if item.shot_id == shot_id), None)
    render_job = next((job for job in package.render_queue if job.shot_id == shot_id), None)
    if shot is None or render_job is None:
        return None

    return (
        prompt_hash(shot.prompt, shot.negative_prompt),
        shot.render_seed,
        render_job.width,
        render_job.height,
        render_job.frame_count,
        json.dumps(shot.controlnet_points, sort_keys=True),
    )


def compare_storyboard_packages(
    previous: StoryboardPackage | None,
    current: StoryboardPackage,
) -> tuple[list[ShotChangeRecord], SceneChangeSummary, list[dict]]:
    """Compare two scene packages and return shot changes and rerender queue."""

    previous_shots = {shot.shot_id: shot for shot in (previous.shots if previous else [])}
    current_shots = {shot.shot_id: shot for shot in current.shots}
    current_jobs = {job.shot_id: job for job in current.render_queue}

    shot_changes: list[ShotChangeRecord] = []
    rerender_queue: list[dict] = []

    added_shots = rerender_shots = unchanged_shots = removed_shots = 0

    for shot_id, current_shot in current_shots.items():
        previous_shot = previous_shots.get(shot_id)
        if previous_shot is None:
            added_shots += 1
            shot_changes.append(
                ShotChangeRecord(
                    scene_id=current.scene_id,
                    shot_id=shot_id,
                    change_type=ShotChangeType.NEW_RENDER,
                    reasons=["missing_in_previous_package"],
                    continuity_group=current_shot.continuity_group,
                    current_reference_shot_id=current_shot.reference_shot_id,
                )
            )
            rerender_queue.append(current_jobs[shot_id].model_dump(mode="json"))
            continue

        previous_signature = _render_signature(previous, shot_id)
        current_signature = _render_signature(current, shot_id)
        reasons: list[str] = []
        if previous_signature != current_signature:
            if previous_signature and current_signature:
                if previous_signature[0] != current_signature[0]:
                    reasons.append("prompt_changed")
                if previous_signature[1] != current_signature[1]:
                    reasons.append("seed_changed")
                if previous_signature[2:4] != current_signature[2:4]:
                    reasons.append("dimensions_changed")
                if previous_signature[4] != current_signature[4]:
                    reasons.append("frame_count_changed")
                if previous_signature[5] != current_signature[5]:
                    reasons.append("controlnet_points_changed")
            else:
                reasons.append("render_signature_missing")

        if previous_shot.reference_shot_id != current_shot.reference_shot_id:
            reasons.append("reference_chain_changed")

        if reasons:
            rerender_shots += 1
            shot_changes.append(
                ShotChangeRecord(
                    scene_id=current.scene_id,
                    shot_id=shot_id,
                    change_type=ShotChangeType.RERENDER,
                    reasons=reasons,
                    continuity_group=current_shot.continuity_group,
                    previous_reference_shot_id=previous_shot.reference_shot_id,
                    current_reference_shot_id=current_shot.reference_shot_id,
                )
            )
            rerender_queue.append(current_jobs[shot_id].model_dump(mode="json"))
        else:
            unchanged_shots += 1
            shot_changes.append(
                ShotChangeRecord(
                    scene_id=current.scene_id,
                    shot_id=shot_id,
                    change_type=ShotChangeType.UNCHANGED,
                    continuity_group=current_shot.continuity_group,
                    previous_reference_shot_id=previous_shot.reference_shot_id,
                    current_reference_shot_id=current_shot.reference_shot_id,
                )
            )

    for shot_id, previous_shot in previous_shots.items():
        if shot_id not in current_shots:
            removed_shots += 1
            shot_changes.append(
                ShotChangeRecord(
                    scene_id=current.scene_id,
                    shot_id=shot_id,
                    change_type=ShotChangeType.REMOVED,
                    reasons=["missing_in_current_package"],
                    continuity_group=previous_shot.continuity_group,
                    previous_reference_shot_id=previous_shot.reference_shot_id,
                )
            )

    summary = SceneChangeSummary(
        scene_id=current.scene_id,
        added_shots=added_shots,
        rerender_shots=rerender_shots,
        unchanged_shots=unchanged_shots,
        removed_shots=removed_shots,
        passes_incremental_reuse=(added_shots + rerender_shots) == 0,
    )
    return shot_changes, summary, rerender_queue


def build_project_change_plan(
    previous: ProjectPackage,
    current: ProjectPackage,
) -> ProjectChangePlan:
    """Build an incremental rerender plan between two project packages."""

    previous_scenes = _scene_map(previous)
    scene_summaries: list[SceneChangeSummary] = []
    shot_changes: list[ShotChangeRecord] = []
    rerender_queue: list[dict] = []

    for current_scene in current.scenes:
        scene_changes, summary, scene_queue = compare_storyboard_packages(
            previous=previous_scenes.get(current_scene.scene_id),
            current=current_scene,
        )
        scene_summaries.append(summary)
        shot_changes.extend(scene_changes)
        rerender_queue.extend(scene_queue)

    previous_scene_ids = set(previous_scenes)
    current_scene_ids = {scene.scene_id for scene in current.scenes}
    for removed_scene_id in sorted(previous_scene_ids - current_scene_ids):
        removed_scene = previous_scenes[removed_scene_id]
        shot_changes.extend(
            ShotChangeRecord(
                scene_id=removed_scene_id,
                shot_id=shot.shot_id,
                change_type=ShotChangeType.REMOVED,
                reasons=["scene_removed"],
                continuity_group=shot.continuity_group,
                previous_reference_shot_id=shot.reference_shot_id,
            )
            for shot in removed_scene.shots
        )
        scene_summaries.append(
            SceneChangeSummary(
                scene_id=removed_scene_id,
                removed_shots=len(removed_scene.shots),
                passes_incremental_reuse=False,
            )
        )

    return ProjectChangePlan(
        generated_at=datetime.now(timezone.utc).isoformat(),
        previous_project_name=previous.project_name,
        current_project_name=current.project_name,
        scene_summaries=scene_summaries,
        shot_changes=shot_changes,
        rerender_queue=rerender_queue,
    )


def project_change_plan_json(plan: ProjectChangePlan, indent: int = 2) -> str:
    """Serialise a project change plan to JSON."""

    return plan.model_dump_json(indent=indent)


def project_change_review_markdown(plan: ProjectChangePlan) -> str:
    """Export a human-readable change review document."""

    lines = [
        f"# Change Plan {plan.current_project_name}",
        "",
        f"- Previous Project: `{plan.previous_project_name}`",
        f"- Current Project: `{plan.current_project_name}`",
        f"- Rerender Jobs: `{len(plan.rerender_queue)}`",
        "",
        "## Scene Summary",
        "",
    ]
    for summary in plan.scene_summaries:
        lines.extend(
            [
                f"### {summary.scene_id}",
                "",
                f"- Added: `{summary.added_shots}`",
                f"- Rerender: `{summary.rerender_shots}`",
                f"- Unchanged: `{summary.unchanged_shots}`",
                f"- Removed: `{summary.removed_shots}`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def write_project_change_plan(
    plan: ProjectChangePlan,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write a project change plan to disk."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    plan_path = target_dir / "project_change_plan.json"
    review_path = target_dir / "project_change_review.md"
    rerender_path = target_dir / "rerender_queue.json"

    plan_path.write_text(project_change_plan_json(plan, indent=2), encoding="utf-8")
    review_path.write_text(project_change_review_markdown(plan), encoding="utf-8")
    rerender_path.write_text(json.dumps(plan.rerender_queue, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "plan_json": plan_path,
        "review_markdown": review_path,
        "rerender_queue": rerender_path,
    }


def load_project_package(path: str | Path) -> ProjectPackage:
    """Load a project package from disk."""

    return ProjectPackage.model_validate_json(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    """CLI entry point for project diffing."""

    import argparse

    parser = argparse.ArgumentParser(description="Build an incremental rerender plan between two project packages.")
    parser.add_argument("--previous-project", required=True, help="Path to previous project manifest JSON")
    parser.add_argument("--current-project", required=True, help="Path to current project manifest JSON")
    parser.add_argument("--output-dir", required=True, help="Directory for change plan outputs")
    args = parser.parse_args()

    previous = load_project_package(args.previous_project)
    current = load_project_package(args.current_project)
    plan = build_project_change_plan(previous, current)
    output_files = write_project_change_plan(plan, args.output_dir)
    print(
        json.dumps(
            {
                "current_project_name": plan.current_project_name,
                "rerender_jobs": len(plan.rerender_queue),
                "output_files": {key: str(value) for key, value in output_files.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
