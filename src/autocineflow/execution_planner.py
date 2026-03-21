"""Execution planning that combines project diffs with previous render QA."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from .change_planner import ProjectChangePlan, ShotChangeType, build_project_change_plan
from .project_delivery import ProjectPackage
from .render_qa import (
    RenderExpectation,
    RenderQAReport,
    load_render_manifest,
    render_qa_report,
)


class ExecutionDecision(str, Enum):
    """How a shot should be handled in the next production run."""

    REUSE = "REUSE"
    RERENDER = "RERENDER"
    SKIP = "SKIP"


class ShotExecutionRecord(BaseModel):
    """Per-shot execution decision."""

    scene_id: str
    shot_id: str
    decision: ExecutionDecision
    reasons: list[str] = Field(default_factory=list)
    source_output_path: str = ""
    continuity_group: str = ""


class SceneExecutionSummary(BaseModel):
    """Scene-level execution summary."""

    scene_id: str
    reused_shots: int = 0
    rerender_shots: int = 0
    skipped_shots: int = 0


class ProjectExecutionPlan(BaseModel):
    """Project-level execution plan combining reuse and rerender decisions."""

    generated_at: str
    previous_project_name: str
    current_project_name: str
    scene_summaries: list[SceneExecutionSummary] = Field(default_factory=list)
    decisions: list[ShotExecutionRecord] = Field(default_factory=list)
    reuse_manifest: list[dict] = Field(default_factory=list)
    rerender_queue: list[dict] = Field(default_factory=list)
    ordered_rerender_queue: list[dict] = Field(default_factory=list)
    change_plan: ProjectChangePlan


def _render_manifest_by_shot(path: Path) -> dict[str, RenderExpectation]:
    """Load a render manifest and index it by shot ID."""

    if not path.exists():
        return {}
    return {entry.shot_id: entry for entry in load_render_manifest(path)}


def _qa_by_shot(package, manifest_by_shot: dict[str, RenderExpectation]) -> dict[str, float]:
    """Compute QA scores keyed by shot ID from a render manifest."""

    if not manifest_by_shot:
        return {}
    report: RenderQAReport = render_qa_report(package, list(manifest_by_shot.values()), min_score=0.9)
    return {result.shot_id: result.score for result in report.shot_results}


def _scene_dir(scenes_dir: Path, scene_id: str) -> Path:
    """Resolve a scene directory written by the project exporter."""

    candidates = [item for item in scenes_dir.iterdir() if item.is_dir()] if scenes_dir.exists() else []
    for candidate in candidates:
        manifest_path = candidate / "storyboard_package.json"
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if payload.get("scene_id") == scene_id:
                return candidate
    return scenes_dir / scene_id.lower()


def build_project_execution_plan(
    previous: ProjectPackage,
    current: ProjectPackage,
    previous_scenes_dir: str | Path,
) -> ProjectExecutionPlan:
    """Build an execution plan that reuses valid previous renders where possible."""

    previous_scenes_dir = Path(previous_scenes_dir)
    change_plan = build_project_change_plan(previous, current)
    previous_scene_map = {scene.scene_id: scene for scene in previous.scenes}
    current_scene_map = {scene.scene_id: scene for scene in current.scenes}

    decisions: list[ShotExecutionRecord] = []
    reuse_manifest: list[dict] = []
    rerender_queue: list[dict] = []
    scene_summaries: list[SceneExecutionSummary] = []

    for scene_summary in change_plan.scene_summaries:
        scene_id = scene_summary.scene_id
        current_scene = current_scene_map.get(scene_id)
        previous_scene = previous_scene_map.get(scene_id)
        summary = SceneExecutionSummary(scene_id=scene_id)

        scene_changes = [change for change in change_plan.shot_changes if change.scene_id == scene_id]
        scene_dir = _scene_dir(previous_scenes_dir, scene_id)
        manifest_by_shot = _render_manifest_by_shot(scene_dir / "render_manifest_template.json")
        qa_scores = _qa_by_shot(previous_scene, manifest_by_shot) if previous_scene is not None else {}
        current_jobs: dict[str, dict] = {}
        if current_scene is not None:
            current_jobs = {job.shot_id: job.model_dump(mode="json") for job in current_scene.render_queue}

        for change in scene_changes:
            if change.change_type == ShotChangeType.REMOVED:
                summary.skipped_shots += 1
                decisions.append(
                    ShotExecutionRecord(
                        scene_id=scene_id,
                        shot_id=change.shot_id,
                        decision=ExecutionDecision.SKIP,
                        reasons=list(change.reasons or ["removed"]),
                        continuity_group=change.continuity_group,
                    )
                )
                continue

            previous_entry = manifest_by_shot.get(change.shot_id)
            reusable = (
                change.change_type == ShotChangeType.UNCHANGED
                and previous_entry is not None
                and previous_entry.status == "rendered"
                and bool(previous_entry.output_path)
                and qa_scores.get(change.shot_id, 0.0) >= 0.99
            )

            if reusable:
                summary.reused_shots += 1
                reasons = ["unchanged_and_previous_render_passed_qa"]
                decisions.append(
                    ShotExecutionRecord(
                        scene_id=scene_id,
                        shot_id=change.shot_id,
                        decision=ExecutionDecision.REUSE,
                        reasons=reasons,
                        source_output_path=previous_entry.output_path,
                        continuity_group=change.continuity_group,
                    )
                )
                reuse_manifest.append(
                    {
                        "scene_id": scene_id,
                        "shot_id": change.shot_id,
                        "source_output_path": previous_entry.output_path,
                        "continuity_group": change.continuity_group,
                    }
                )
            else:
                summary.rerender_shots += 1
                reasons = list(change.reasons)
                if change.change_type == ShotChangeType.UNCHANGED and previous_entry is None:
                    reasons.append("missing_previous_render_manifest")
                elif change.change_type == ShotChangeType.UNCHANGED and previous_entry is not None:
                    reasons.append("previous_render_failed_or_missing_qa")
                decisions.append(
                    ShotExecutionRecord(
                        scene_id=scene_id,
                        shot_id=change.shot_id,
                        decision=ExecutionDecision.RERENDER,
                        reasons=reasons or ["new_or_changed_shot"],
                        continuity_group=change.continuity_group,
                    )
                )
                if change.shot_id in current_jobs:
                    rerender_queue.append(current_jobs[change.shot_id])

        scene_summaries.append(summary)

    ordered_rerender_queue = order_rerender_queue(rerender_queue)

    return ProjectExecutionPlan(
        generated_at=datetime.now(timezone.utc).isoformat(),
        previous_project_name=previous.project_name,
        current_project_name=current.project_name,
        scene_summaries=scene_summaries,
        decisions=decisions,
        reuse_manifest=reuse_manifest,
        rerender_queue=rerender_queue,
        ordered_rerender_queue=ordered_rerender_queue,
        change_plan=change_plan,
    )


def order_rerender_queue(rerender_queue: list[dict]) -> list[dict]:
    """Topologically order rerender jobs by in-scene reference shot dependencies."""

    jobs_by_shot = {job["shot_id"]: dict(job) for job in rerender_queue}
    dependency_map: dict[str, set[str]] = {}
    for shot_id, job in jobs_by_shot.items():
        metadata = job.get("metadata", {})
        reference_shot_id = metadata.get("reference_shot_id", "")
        deps = {reference_shot_id} if reference_shot_id in jobs_by_shot else set()
        dependency_map[shot_id] = deps

    ordered: list[dict] = []
    available = sorted(shot_id for shot_id, deps in dependency_map.items() if not deps)
    processed: set[str] = set()

    while available:
        shot_id = available.pop(0)
        if shot_id in processed:
            continue
        processed.add(shot_id)
        job = dict(jobs_by_shot[shot_id])
        job["execution_order"] = len(ordered) + 1
        ordered.append(job)

        for candidate, deps in dependency_map.items():
            if shot_id in deps:
                deps.remove(shot_id)
                if not deps and candidate not in processed:
                    available.append(candidate)
        available.sort()

    # Fallback for cycles or malformed references.
    for shot_id in sorted(jobs_by_shot):
        if shot_id in processed:
            continue
        job = dict(jobs_by_shot[shot_id])
        job["execution_order"] = len(ordered) + 1
        ordered.append(job)

    return ordered


def project_execution_plan_json(plan: ProjectExecutionPlan, indent: int = 2) -> str:
    """Serialise a project execution plan to JSON."""

    return plan.model_dump_json(indent=indent)


def project_execution_review_markdown(plan: ProjectExecutionPlan) -> str:
    """Export a human-readable execution review document."""

    lines = [
        f"# Execution Plan {plan.current_project_name}",
        "",
        f"- Previous Project: `{plan.previous_project_name}`",
        f"- Current Project: `{plan.current_project_name}`",
        f"- Reuse Count: `{len(plan.reuse_manifest)}`",
        f"- Rerender Count: `{len(plan.rerender_queue)}`",
        f"- Ordered Rerender Count: `{len(plan.ordered_rerender_queue)}`",
        "",
        "## Scene Summary",
        "",
    ]
    for summary in plan.scene_summaries:
        lines.extend(
            [
                f"### {summary.scene_id}",
                "",
                f"- Reuse: `{summary.reused_shots}`",
                f"- Rerender: `{summary.rerender_shots}`",
                f"- Skip: `{summary.skipped_shots}`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def write_project_execution_plan(
    plan: ProjectExecutionPlan,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write an execution plan to disk."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    plan_path = target_dir / "project_execution_plan.json"
    review_path = target_dir / "project_execution_review.md"
    reuse_path = target_dir / "reuse_manifest.json"
    rerender_path = target_dir / "rerender_queue.json"
    ordered_rerender_path = target_dir / "ordered_rerender_queue.json"

    plan_path.write_text(project_execution_plan_json(plan, indent=2), encoding="utf-8")
    review_path.write_text(project_execution_review_markdown(plan), encoding="utf-8")
    reuse_path.write_text(json.dumps(plan.reuse_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    rerender_path.write_text(json.dumps(plan.rerender_queue, indent=2, ensure_ascii=False), encoding="utf-8")
    ordered_rerender_path.write_text(json.dumps(plan.ordered_rerender_queue, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "plan_json": plan_path,
        "review_markdown": review_path,
        "reuse_manifest": reuse_path,
        "rerender_queue": rerender_path,
        "ordered_rerender_queue": ordered_rerender_path,
    }


def main() -> int:
    """CLI entry point for execution planning."""

    import argparse

    from .change_planner import load_project_package

    parser = argparse.ArgumentParser(description="Build a reuse/rerender execution plan from previous renders.")
    parser.add_argument("--previous-project", required=True, help="Path to previous project manifest JSON")
    parser.add_argument("--current-project", required=True, help="Path to current project manifest JSON")
    parser.add_argument("--previous-scenes-dir", required=True, help="Directory with previous per-scene exports")
    parser.add_argument("--output-dir", required=True, help="Directory for execution plan outputs")
    args = parser.parse_args()

    previous = load_project_package(args.previous_project)
    current = load_project_package(args.current_project)
    plan = build_project_execution_plan(previous, current, args.previous_scenes_dir)
    output_files = write_project_execution_plan(plan, args.output_dir)
    print(
        json.dumps(
            {
                "current_project_name": plan.current_project_name,
                "reuse_count": len(plan.reuse_manifest),
                "rerender_count": len(plan.rerender_queue),
                "ordered_rerender_count": len(plan.ordered_rerender_queue),
                "output_files": {key: str(value) for key, value in output_files.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
