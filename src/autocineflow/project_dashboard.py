"""Project dashboard aggregation for production oversight."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .asset_library import AssetLibrary, latest_scene_versions
from .delivery import slugify
from .execution_planner import ProjectExecutionPlan
from .project_delivery import ProjectPackage
from .project_render_qa import ProjectRenderQAReport


class ProjectDashboardScene(BaseModel):
    """Single scene row in the project dashboard."""

    scene_id: str
    storyboard_quality_score: float = Field(..., ge=0.0, le=1.0)
    storyboard_passes: bool
    keyframe_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    keyframe_passes: Optional[bool] = None
    local_visual_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    local_visual_passes: Optional[bool] = None
    local_visual_status: str = ""
    render_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    render_passes: Optional[bool] = None
    reuse_count: int = Field(default=0, ge=0)
    rerender_count: int = Field(default=0, ge=0)
    overall_status: str


class ProjectDashboard(BaseModel):
    """Unified production dashboard for a project."""

    project_name: str
    generated_at: str
    scene_count: int = Field(..., ge=1)
    storyboard_ready_count: int = Field(..., ge=0)
    keyframe_ready_count: int = Field(..., ge=0)
    local_visual_ready_count: int = Field(..., ge=0)
    render_ready_count: int = Field(..., ge=0)
    total_reuse_count: int = Field(..., ge=0)
    total_rerender_count: int = Field(..., ge=0)
    all_storyboard_ready: bool
    all_keyframes_ready: bool
    all_local_visual_ready: bool
    all_render_ready: bool
    scene_rows: list[ProjectDashboardScene] = Field(default_factory=list)


def build_project_dashboard(
    project: ProjectPackage,
    render_qa: ProjectRenderQAReport | None = None,
    execution_plan: ProjectExecutionPlan | None = None,
    asset_library: AssetLibrary | None = None,
    scenes_dir: str | Path | None = None,
) -> ProjectDashboard:
    """Build a project dashboard from available project reports."""

    render_rows = {row.scene_id: row for row in (render_qa.scene_results if render_qa else [])}
    execution_rows = {row.scene_id: row for row in (execution_plan.scene_summaries if execution_plan else [])}
    keyframe_rows: dict[str, tuple[Optional[float], Optional[bool], str, Optional[float], Optional[bool], str]] = {}
    if asset_library is not None:
        for scene in latest_scene_versions(asset_library):
            keyframe_rows[scene.scene_id] = (
                scene.keyframe_qa_score,
                scene.keyframe_gate_passed,
                scene.keyframe_gate_blocking_reason,
                scene.local_visual_review_score,
                scene.local_visual_review_passed,
                scene.local_visual_review_status,
            )
    elif scenes_dir is not None:
        scenes_path = Path(scenes_dir)
        for scene_summary in project.scene_summaries:
            scene_dir = scenes_path / slugify(scene_summary.scene_id, default=scene_summary.scene_id.lower())
            report_path = scene_dir / "keyframe_qc" / "keyframe_qa_report.json"
            gate_path = scene_dir / "keyframe_qc" / "keyframe_gate_report.json"
            local_visual_path = scene_dir / "keyframe_qc" / "local_vlm" / "local_visual_review_report.json"
            keyframe_score = None
            keyframe_passes = None
            keyframe_blocking_reason = ""
            if gate_path.exists():
                gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
                keyframe_score = float(gate_payload.get("score", 0.0)) if "score" in gate_payload else None
                keyframe_passes = bool(gate_payload.get("passes_gate")) if "passes_gate" in gate_payload else None
                keyframe_blocking_reason = str(gate_payload.get("blocking_reason", "") or "")
            local_visual_score = None
            local_visual_passes = None
            local_visual_status = ""
            if local_visual_path.exists():
                local_payload = json.loads(local_visual_path.read_text(encoding="utf-8"))
                results = local_payload.get("results", []) or []
                ok_scores = [
                    float(result.get("score", 0.0))
                    for result in results
                    if result.get("status") == "ok" and result.get("score") is not None
                ]
                local_visual_score = round(sum(ok_scores) / len(ok_scores), 4) if ok_scores else None
                if local_payload.get("skipped"):
                    local_visual_status = "skipped"
                elif any(result.get("status") == "error" for result in results):
                    local_visual_status = "error"
                elif results:
                    local_visual_status = "reviewed"
                reviewed_results = [result for result in results if result.get("status") == "ok"]
                if reviewed_results:
                    local_visual_passes = not any(
                        result.get("recommendation") in {"repair", "rerender"} or (result.get("issues") or [])
                        for result in reviewed_results
                    )
            if report_path.exists() and keyframe_score is None:
                payload = json.loads(report_path.read_text(encoding="utf-8"))
                keyframe_score = float(payload.get("score", 0.0)) if "score" in payload else None
                keyframe_passes = bool(payload.get("passes_gate")) if "passes_gate" in payload else None
            keyframe_rows[scene_summary.scene_id] = (
                keyframe_score,
                keyframe_passes,
                keyframe_blocking_reason,
                local_visual_score,
                local_visual_passes,
                local_visual_status,
            )

    scene_rows: list[ProjectDashboardScene] = []
    for scene_summary in project.scene_summaries:
        render_row = render_rows.get(scene_summary.scene_id)
        execution_row = execution_rows.get(scene_summary.scene_id)
        keyframe_score, keyframe_passes, keyframe_blocking_reason, local_visual_score, local_visual_passes, local_visual_status = keyframe_rows.get(
            scene_summary.scene_id,
            (None, None, "", None, None, ""),
        )

        storyboard_passes = scene_summary.passes_gate
        render_passes = render_row.passes_gate if render_row else None
        reuse_count = execution_row.reused_shots if execution_row else 0
        rerender_count = execution_row.rerender_shots if execution_row else 0

        if not storyboard_passes:
            overall_status = "storyboard_blocked"
        elif keyframe_passes is False:
            overall_status = f"keyframe_blocked:{keyframe_blocking_reason or 'qa'}"
        elif local_visual_passes is False:
            overall_status = "visual_review_blocked"
        elif render_passes is False:
            overall_status = "render_blocked"
        elif rerender_count > 0:
            overall_status = "rerender_pending"
        elif render_passes is True:
            overall_status = "ready"
        else:
            overall_status = "storyboard_ready"

        scene_rows.append(
            ProjectDashboardScene(
                scene_id=scene_summary.scene_id,
                storyboard_quality_score=scene_summary.quality_score,
                storyboard_passes=storyboard_passes,
                keyframe_score=keyframe_score,
                keyframe_passes=keyframe_passes,
                local_visual_score=local_visual_score,
                local_visual_passes=local_visual_passes,
                local_visual_status=local_visual_status,
                render_score=render_row.score if render_row else None,
                render_passes=render_passes,
                reuse_count=reuse_count,
                rerender_count=rerender_count,
                overall_status=overall_status,
            )
        )

    storyboard_ready_count = sum(row.storyboard_passes for row in scene_rows)
    keyframe_ready_count = sum(row.keyframe_passes is True for row in scene_rows)
    local_visual_ready_count = sum(row.local_visual_passes is True for row in scene_rows)
    render_ready_count = sum(row.render_passes is True for row in scene_rows)
    total_reuse_count = sum(row.reuse_count for row in scene_rows)
    total_rerender_count = sum(row.rerender_count for row in scene_rows)

    return ProjectDashboard(
        project_name=project.project_name,
        generated_at=datetime.now(timezone.utc).isoformat(),
        scene_count=len(scene_rows),
        storyboard_ready_count=storyboard_ready_count,
        keyframe_ready_count=keyframe_ready_count,
        local_visual_ready_count=local_visual_ready_count,
        render_ready_count=render_ready_count,
        total_reuse_count=total_reuse_count,
        total_rerender_count=total_rerender_count,
        all_storyboard_ready=storyboard_ready_count == len(scene_rows),
        all_keyframes_ready=keyframe_ready_count == len(scene_rows) if scene_rows else False,
        all_local_visual_ready=all(row.local_visual_passes is not False for row in scene_rows),
        all_render_ready=render_ready_count == len(scene_rows) if scene_rows else False,
        scene_rows=scene_rows,
    )


def project_dashboard_json(dashboard: ProjectDashboard, indent: int = 2) -> str:
    """Serialise a project dashboard to JSON."""

    return dashboard.model_dump_json(indent=indent)


def project_dashboard_markdown(dashboard: ProjectDashboard) -> str:
    """Export a human-readable project dashboard document."""

    lines = [
        f"# Project Dashboard {dashboard.project_name}",
        "",
        f"- Scene Count: `{dashboard.scene_count}`",
        f"- Storyboard Ready: `{dashboard.storyboard_ready_count}/{dashboard.scene_count}`",
        f"- Keyframe Ready: `{dashboard.keyframe_ready_count}/{dashboard.scene_count}`",
        f"- Local Visual Ready: `{dashboard.local_visual_ready_count}/{dashboard.scene_count}`",
        f"- Render Ready: `{dashboard.render_ready_count}/{dashboard.scene_count}`",
        f"- Reuse Count: `{dashboard.total_reuse_count}`",
        f"- Rerender Count: `{dashboard.total_rerender_count}`",
        "",
        "## Scene Status",
        "",
    ]
    for row in dashboard.scene_rows:
        lines.extend(
            [
                f"### {row.scene_id}",
                "",
                f"- Storyboard Quality: `{row.storyboard_quality_score:.3f}`",
                f"- Storyboard Pass: `{row.storyboard_passes}`",
                f"- Keyframe Score: `{row.keyframe_score if row.keyframe_score is not None else 'n/a'}`",
                f"- Keyframe Pass: `{row.keyframe_passes if row.keyframe_passes is not None else 'n/a'}`",
                f"- Local Visual Score: `{row.local_visual_score if row.local_visual_score is not None else 'n/a'}`",
                f"- Local Visual Pass: `{row.local_visual_passes if row.local_visual_passes is not None else row.local_visual_status or 'n/a'}`",
                f"- Render Score: `{row.render_score if row.render_score is not None else 'n/a'}`",
                f"- Render Pass: `{row.render_passes if row.render_passes is not None else 'n/a'}`",
                f"- Reuse / Rerender: `{row.reuse_count}/{row.rerender_count}`",
                f"- Status: `{row.overall_status}`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def write_project_dashboard(
    dashboard: ProjectDashboard,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write dashboard assets to disk."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    json_path = target_dir / "project_dashboard.json"
    markdown_path = target_dir / "project_dashboard.md"
    json_path.write_text(project_dashboard_json(dashboard, indent=2), encoding="utf-8")
    markdown_path.write_text(project_dashboard_markdown(dashboard), encoding="utf-8")
    return {
        "dashboard_json": json_path,
        "dashboard_markdown": markdown_path,
    }


def main() -> int:
    """CLI entry point for project dashboard generation."""

    import argparse

    from .change_planner import load_project_package

    parser = argparse.ArgumentParser(description="Build a project dashboard from project-level reports.")
    parser.add_argument("--project-file", required=True, help="Path to project manifest JSON")
    parser.add_argument("--render-qa-file", default=None, help="Optional project render QA report JSON")
    parser.add_argument("--execution-plan-file", default=None, help="Optional project execution plan JSON")
    parser.add_argument("--output-dir", required=True, help="Directory for dashboard outputs")
    args = parser.parse_args()

    project = load_project_package(args.project_file)
    render_qa = (
        ProjectRenderQAReport.model_validate_json(Path(args.render_qa_file).read_text(encoding="utf-8"))
        if args.render_qa_file
        else None
    )
    execution_plan = (
        ProjectExecutionPlan.model_validate_json(Path(args.execution_plan_file).read_text(encoding="utf-8"))
        if args.execution_plan_file
        else None
    )
    dashboard = build_project_dashboard(project, render_qa=render_qa, execution_plan=execution_plan)
    output_files = write_project_dashboard(dashboard, args.output_dir)
    print(
        json.dumps(
            {
                "project_name": dashboard.project_name,
                "scene_count": dashboard.scene_count,
                "output_files": {key: str(value) for key, value in output_files.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
