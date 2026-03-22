"""Optional local visual review using an external Python env and local Qwen-VL weights."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from .config_loader import resolve_local_vlm_settings
from .keyframe_qa import KeyframeQAReport


class LocalVisualReviewResult(BaseModel):
    """Review result for a single keyframe image."""

    shot_id: str
    output_path: str
    status: str
    score: float | None = None
    recommendation: str = ""
    issues: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    reason: str = ""


class LocalVisualReviewReport(BaseModel):
    """Aggregate report for optional local visual review."""

    source_id: str
    enabled: bool
    skipped: bool
    reviewer: str = "qwen3-vl-local"
    results: list[LocalVisualReviewResult] = Field(default_factory=list)


def review_keyframes_with_local_vlm(
    keyframe_report: KeyframeQAReport,
    config_path: str | None = None,
    timeout_seconds: float = 600.0,
) -> LocalVisualReviewReport:
    """Run local visual review for keyframes using an external Python environment."""

    settings = resolve_local_vlm_settings(config_path)
    python_path = Path(settings["python_path"])
    model_path = Path(settings["model_path"])
    worker_path = Path(__file__).with_name("local_vlm_worker.py")

    if not python_path.exists() or not model_path.exists():
        return LocalVisualReviewReport(
            source_id=keyframe_report.source_id,
            enabled=False,
            skipped=True,
            results=[],
        )

    results: list[LocalVisualReviewResult] = []
    for item in keyframe_report.results:
        command = [
            str(python_path),
            str(worker_path),
            "--model-path",
            str(model_path),
            "--image-path",
            item.output_path,
            "--device-preference",
            settings["device_preference"],
            "--min-free-vram-gb",
            settings["min_free_vram_gb"],
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout_seconds)
        stdout = completed.stdout.strip()
        payload = json.loads(stdout or "{}") if stdout else {}
        status = str(payload.get("status", "error"))
        if status == "ok":
            result = payload.get("result", {})
            raw_score = float(result.get("score", 0.0))
            normalized_score = raw_score / 10.0 if raw_score > 1.0 and raw_score <= 10.0 else raw_score
            issues = [
                name
                for name in (
                    "text_artifact",
                    "face_distortion",
                    "anatomy_issue",
                    "blur_issue",
                    "wardrobe_drift",
                    "scene_incoherence",
                )
                if result.get(name)
            ]
            results.append(
                LocalVisualReviewResult(
                    shot_id=item.shot_id,
                    output_path=item.output_path,
                    status="ok",
                    score=normalized_score,
                    recommendation=str(result.get("recommendation", "")),
                    issues=issues,
                    notes=[str(note) for note in result.get("notes", [])],
                )
            )
        else:
            results.append(
                LocalVisualReviewResult(
                    shot_id=item.shot_id,
                    output_path=item.output_path,
                    status=status,
                    recommendation="skip",
                    reason=str(payload.get("reason", "") or completed.stderr[-300:]),
                )
            )

    return LocalVisualReviewReport(
        source_id=keyframe_report.source_id,
        enabled=True,
        skipped=all(result.status == "skipped" for result in results) if results else True,
        results=results,
    )


def write_local_visual_review_report(report: LocalVisualReviewReport, output_dir: str | Path) -> dict[str, Path]:
    """Write local visual review outputs to disk."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "local_visual_review_report.json"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return {"report_json": json_path}
