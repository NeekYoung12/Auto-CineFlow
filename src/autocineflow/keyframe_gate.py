"""Combined keyframe gate using heuristic QA and optional visual review."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from .keyframe_qa import KeyframeQAReport
from .local_visual_review import LocalVisualReviewReport


class KeyframeGateResult(BaseModel):
    """Per-shot gate result derived from keyframe QA and visual review."""

    shot_id: str
    keyframe_score: float = Field(..., ge=0.0, le=1.0)
    keyframe_passes: bool
    local_visual_score: float | None = Field(default=None, ge=0.0, le=1.0)
    local_visual_passes: bool | None = None
    local_visual_status: str = ""
    passes_gate: bool
    blocked_by: str = ""
    recommendation: str = ""


class KeyframeGateReport(BaseModel):
    """Scene-level gate result for keyframe admission into the video stage."""

    source_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    passes_gate: bool
    blocking_reason: str = ""
    results: list[KeyframeGateResult] = Field(default_factory=list)


def build_keyframe_gate_report(
    keyframe_report: KeyframeQAReport,
    local_visual_report: LocalVisualReviewReport | None = None,
) -> KeyframeGateReport:
    """Combine heuristic QA and optional visual review into one gate report."""

    visual_by_shot = {}
    if local_visual_report is not None:
        visual_by_shot = {result.shot_id: result for result in local_visual_report.results}

    results: list[KeyframeGateResult] = []
    blocking_reasons: set[str] = set()
    for result in keyframe_report.results:
        visual = visual_by_shot.get(result.shot_id)
        visual_status = visual.status if visual else ""
        visual_passes = None
        local_visual_score = None
        if visual and visual.status == "ok":
            local_visual_score = visual.score
            visual_passes = not (
                visual.recommendation in {"repair", "rerender"}
                or any(issue in {"text_artifact", "face_distortion", "anatomy_issue", "blur_issue"} for issue in visual.issues)
            )
        passes_gate = result.score >= keyframe_report.min_score and not result.issues and (visual_passes is not False)
        blocked_by = ""
        if result.issues:
            blocked_by = "keyframe_qa"
        elif visual_passes is False:
            blocked_by = "local_visual_review"
        if blocked_by:
            blocking_reasons.add(blocked_by)
        results.append(
            KeyframeGateResult(
                shot_id=result.shot_id,
                keyframe_score=result.score,
                keyframe_passes=not bool(result.issues),
                local_visual_score=local_visual_score,
                local_visual_passes=visual_passes,
                local_visual_status=visual_status,
                passes_gate=passes_gate,
                blocked_by=blocked_by,
                recommendation=visual.recommendation if visual else result.recommendation,
            )
        )

    score = 0.0
    if results:
        score = round(sum((item.keyframe_score + (item.local_visual_score or item.keyframe_score)) / 2.0 for item in results) / len(results), 4)

    blocking_reason = ""
    if "keyframe_qa" in blocking_reasons:
        blocking_reason = "keyframe_qa"
    elif "local_visual_review" in blocking_reasons:
        blocking_reason = "local_visual_review"

    return KeyframeGateReport(
        source_id=keyframe_report.source_id,
        score=score,
        passes_gate=all(item.passes_gate for item in results) if results else False,
        blocking_reason=blocking_reason,
        results=results,
    )


def write_keyframe_gate_report(report: KeyframeGateReport, output_dir: str | Path) -> dict[str, Path]:
    """Write keyframe gate outputs to disk."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "keyframe_gate_report.json"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return {"report_json": json_path}
