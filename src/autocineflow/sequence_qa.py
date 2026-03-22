"""QA and repair planning for assembled multi-clip scene sequences."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from .sequence_assembly import SequenceAssemblyPlan


class SequenceClipQA(BaseModel):
    """QA result for a single clip referenced by an assembly plan."""

    clip_id: str
    shot_id: str
    asset_exists: bool
    extension_match: bool
    repair_required: bool
    issues: list[str] = Field(default_factory=list)


class SequenceQAReport(BaseModel):
    """Scene-level QA report for an assembled previs sequence."""

    scene_id: str
    clip_count: int = Field(..., ge=0)
    available_clips: int = Field(..., ge=0)
    missing_clips: int = Field(..., ge=0)
    sequence_file_exists: bool
    score: float = Field(..., ge=0.0, le=1.0)
    passes_gate: bool
    clip_results: list[SequenceClipQA] = Field(default_factory=list)
    critical_issues: list[str] = Field(default_factory=list)


class SequenceRepairAction(BaseModel):
    """Repair action for a missing or invalid clip."""

    clip_id: str
    shot_id: str
    action: str
    reasons: list[str] = Field(default_factory=list)


class SequenceRepairPlan(BaseModel):
    """Repair plan for restoring a broken assembled sequence."""

    scene_id: str
    action_count: int = Field(..., ge=0)
    actions: list[SequenceRepairAction] = Field(default_factory=list)


def build_sequence_qa_report(
    plan: SequenceAssemblyPlan,
    assembly_dir: str | Path,
    final_sequence_filename: str | None = None,
) -> SequenceQAReport:
    """Validate clip availability and final sequence presence for an assembly plan."""

    assembly_dir = Path(assembly_dir)
    clip_results: list[SequenceClipQA] = []

    for clip in plan.clips:
        asset_path = assembly_dir / Path(clip.expected_asset_path).name
        asset_exists = asset_path.exists()
        extension_match = asset_path.suffix.lower() in {".mp4", ".mov", ".mkv"} if asset_exists else False
        issues: list[str] = []
        if not asset_exists:
            issues.append("missing_clip_asset")
        if asset_exists and not extension_match:
            issues.append("unexpected_clip_extension")
        clip_results.append(
            SequenceClipQA(
                clip_id=clip.clip_id,
                shot_id=clip.shot_id,
                asset_exists=asset_exists,
                extension_match=extension_match,
                repair_required=bool(issues),
                issues=issues,
            )
        )

    if final_sequence_filename:
        final_sequence_path = assembly_dir / final_sequence_filename
    else:
        final_sequence_path = assembly_dir / f"{plan.scene_id.lower()}_sequence.mp4"
    sequence_file_exists = final_sequence_path.exists()

    available_clips = sum(item.asset_exists for item in clip_results)
    missing_clips = len(clip_results) - available_clips
    score = 0.0
    if clip_results:
        clip_quality = sum(item.asset_exists and item.extension_match for item in clip_results) / len(clip_results)
        sequence_bonus = 0.2 if sequence_file_exists else 0.0
        score = round(min(1.0, clip_quality * 0.8 + sequence_bonus), 4)

    critical_issues = []
    if missing_clips:
        critical_issues.append("missing_clip_assets")
    if not sequence_file_exists:
        critical_issues.append("missing_final_sequence")

    return SequenceQAReport(
        scene_id=plan.scene_id,
        clip_count=len(clip_results),
        available_clips=available_clips,
        missing_clips=missing_clips,
        sequence_file_exists=sequence_file_exists,
        score=score,
        passes_gate=not critical_issues and score >= 0.95,
        clip_results=clip_results,
        critical_issues=critical_issues,
    )


def build_sequence_repair_plan(report: SequenceQAReport) -> SequenceRepairPlan:
    """Turn a failed sequence QA report into a clip-level repair plan."""

    actions: list[SequenceRepairAction] = []
    for clip in report.clip_results:
        if clip.repair_required:
            actions.append(
                SequenceRepairAction(
                    clip_id=clip.clip_id,
                    shot_id=clip.shot_id,
                    action="rerender_clip",
                    reasons=list(clip.issues),
                )
            )

    if "missing_final_sequence" in report.critical_issues:
        actions.append(
            SequenceRepairAction(
                clip_id="FINAL_SEQUENCE",
                shot_id="FINAL_SEQUENCE",
                action="restitch_sequence",
                reasons=["missing_final_sequence"],
            )
        )

    return SequenceRepairPlan(
        scene_id=report.scene_id,
        action_count=len(actions),
        actions=actions,
    )


def sequence_qa_report_json(report: SequenceQAReport, indent: int = 2) -> str:
    """Serialise a sequence QA report to JSON."""

    return report.model_dump_json(indent=indent)


def sequence_repair_plan_json(plan: SequenceRepairPlan, indent: int = 2) -> str:
    """Serialise a sequence repair plan to JSON."""

    return plan.model_dump_json(indent=indent)


def sequence_qa_markdown(report: SequenceQAReport) -> str:
    """Export a human-readable sequence QA report."""

    lines = [
        f"# Sequence QA {report.scene_id}",
        "",
        f"- Score: `{report.score:.3f}`",
        f"- Available / Missing Clips: `{report.available_clips}/{report.missing_clips}`",
        f"- Final Sequence Exists: `{report.sequence_file_exists}`",
        f"- Passes Gate: `{report.passes_gate}`",
        "",
        "## Clip Results",
        "",
    ]
    for clip in report.clip_results:
        lines.extend(
            [
                f"### {clip.clip_id}",
                "",
                f"- Shot: `{clip.shot_id}`",
                f"- Asset Exists: `{clip.asset_exists}`",
                f"- Extension Match: `{clip.extension_match}`",
                f"- Issues: `{', '.join(clip.issues) if clip.issues else 'none'}`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def sequence_repair_markdown(plan: SequenceRepairPlan) -> str:
    """Export a human-readable sequence repair plan."""

    lines = [
        f"# Sequence Repair {plan.scene_id}",
        "",
        f"- Action Count: `{plan.action_count}`",
        "",
        "## Actions",
        "",
    ]
    for action in plan.actions:
        lines.extend(
            [
                f"### {action.clip_id}",
                "",
                f"- Shot: `{action.shot_id}`",
                f"- Action: `{action.action}`",
                f"- Reasons: `{', '.join(action.reasons)}`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def write_sequence_qc_outputs(
    report: SequenceQAReport,
    repair_plan: SequenceRepairPlan,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write sequence QA and repair outputs to disk."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_json_path = output_dir / "sequence_qa_report.json"
    report_md_path = output_dir / "sequence_qa_report.md"
    repair_json_path = output_dir / "sequence_repair_plan.json"
    repair_md_path = output_dir / "sequence_repair_plan.md"

    report_json_path.write_text(sequence_qa_report_json(report, indent=2), encoding="utf-8")
    report_md_path.write_text(sequence_qa_markdown(report), encoding="utf-8")
    repair_json_path.write_text(sequence_repair_plan_json(repair_plan, indent=2), encoding="utf-8")
    repair_md_path.write_text(sequence_repair_markdown(repair_plan), encoding="utf-8")
    return {
        "report_json": report_json_path,
        "report_markdown": report_md_path,
        "repair_json": repair_json_path,
        "repair_markdown": repair_md_path,
    }
