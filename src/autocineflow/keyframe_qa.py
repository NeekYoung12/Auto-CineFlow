"""Keyframe QA for text artifacts, blur, and production readiness."""

from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat
from pydantic import BaseModel, Field

from .result_ingest import ArtifactDownloadBatch, ArtifactDownloadRecord


class KeyframeQAChecks(BaseModel):
    """Per-frame QA booleans."""

    file_present: bool
    resolution_ok: bool
    sharpness_ok: bool
    text_artifact_risk_low: bool


class KeyframeQAResult(BaseModel):
    """Per-shot keyframe QA result."""

    shot_id: str
    output_path: str = ""
    score: float = Field(..., ge=0.0, le=1.0)
    checks: KeyframeQAChecks
    issues: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    recommendation: str = ""


class KeyframeQAReport(BaseModel):
    """Aggregate keyframe QA report."""

    source_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    min_score: float = Field(..., ge=0.0, le=1.0)
    passes_gate: bool
    results: list[KeyframeQAResult] = Field(default_factory=list)
    critical_issues: list[str] = Field(default_factory=list)


def analyze_keyframe(
    image_path: str | Path,
    min_resolution_short_side: int = 1024,
    min_sharpness_score: float = 12.0,
    max_text_artifact_score: float = 0.24,
) -> KeyframeQAResult:
    """Analyze one keyframe image using local heuristics."""

    path = Path(image_path)
    if not path.exists():
        checks = KeyframeQAChecks(
            file_present=False,
            resolution_ok=False,
            sharpness_ok=False,
            text_artifact_risk_low=False,
        )
        return KeyframeQAResult(
            shot_id=path.stem,
            output_path=str(path),
            score=0.0,
            checks=checks,
            issues=["missing_file"],
            metrics={},
            recommendation="rerender_keyframe",
        )

    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        short_side = min(width, height)
        gray = image.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_mean = float(ImageStat.Stat(edges).mean[0])
        edge_std = math.sqrt(float(ImageStat.Stat(edges).var[0]))
        sharpness_score = round((edge_mean * 0.7) + (edge_std * 0.3), 4)
        text_artifact_score = round(_text_artifact_score(image), 4)

    resolution_ok = short_side >= min_resolution_short_side
    sharpness_ok = sharpness_score >= min_sharpness_score
    text_artifact_risk_low = text_artifact_score <= max_text_artifact_score

    issues: list[str] = []
    if not resolution_ok:
        issues.append("low_resolution_keyframe")
    if not sharpness_ok:
        issues.append("soft_or_blurry_keyframe")
    if not text_artifact_risk_low:
        issues.append("text_artifact_risk")

    checks = KeyframeQAChecks(
        file_present=True,
        resolution_ok=resolution_ok,
        sharpness_ok=sharpness_ok,
        text_artifact_risk_low=text_artifact_risk_low,
    )
    weights = {
        "file_present": 0.2,
        "resolution_ok": 0.2,
        "sharpness_ok": 0.3,
        "text_artifact_risk_low": 0.3,
    }
    score = round(
        sum(getattr(checks, key) * weight for key, weight in weights.items()),
        4,
    )
    recommendation = "approve"
    if "text_artifact_risk" in issues:
        recommendation = "repair_text_artifacts"
    elif "soft_or_blurry_keyframe" in issues:
        recommendation = "rebuild_keyframe"
    elif "low_resolution_keyframe" in issues:
        recommendation = "upscale_or_rebuild"

    return KeyframeQAResult(
        shot_id=path.stem,
        output_path=str(path),
        score=score,
        checks=checks,
        issues=issues,
        metrics={
            "width": float(width),
            "height": float(height),
            "sharpness_score": sharpness_score,
            "text_artifact_score": text_artifact_score,
        },
        recommendation=recommendation,
    )


def keyframe_qa_report(
    downloads: ArtifactDownloadBatch,
    min_score: float = 0.75,
) -> KeyframeQAReport:
    """Evaluate all downloaded PNG/JPG keyframes in a download batch."""

    results: list[KeyframeQAResult] = []
    for record in downloads.records:
        if not record.downloaded:
            continue
        if not record.output_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            continue
        result = analyze_keyframe(record.output_path)
        if result.shot_id != record.shot_id:
            result = result.model_copy(update={"shot_id": record.shot_id})
        results.append(result)

    score = round(sum(result.score for result in results) / len(results), 4) if results else 0.0
    critical_issues = sorted({issue for result in results for issue in result.issues})
    return KeyframeQAReport(
        source_id=downloads.source_id,
        score=score,
        min_score=min_score,
        passes_gate=bool(results) and score >= min_score and not critical_issues,
        results=results,
        critical_issues=critical_issues,
    )


def select_best_keyframe_downloads(*batches: ArtifactDownloadBatch) -> ArtifactDownloadBatch:
    """Select the highest-scoring downloaded keyframe per shot across multiple batches."""

    best_by_shot: dict[str, tuple[ArtifactDownloadRecord, float]] = {}
    source_id = ""
    for batch in batches:
        source_id = source_id or batch.source_id
        report = keyframe_qa_report(batch)
        report_by_shot = {result.shot_id: result for result in report.results}
        for record in batch.records:
            if not record.downloaded or not record.output_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                continue
            result = report_by_shot.get(record.shot_id)
            score = result.score if result else 0.0
            current = best_by_shot.get(record.shot_id)
            if current is None or score > current[1]:
                best_by_shot[record.shot_id] = (record, score)

    return ArtifactDownloadBatch(
        source_id=source_id,
        records=[item[0] for item in best_by_shot.values()],
    )


def build_keyframe_repair_jobs(
    keyframe_jobs: list,
    qa_report: KeyframeQAReport,
) -> list:
    """Build repair-rebuild jobs by strengthening prompts based on QA issues."""

    failing = {result.shot_id: result for result in qa_report.results if not result.checks.text_artifact_risk_low or not result.checks.sharpness_ok}
    repair_jobs = []
    for job in keyframe_jobs:
        result = failing.get(job.shot_id)
        if result is None:
            continue
        payload = dict(job.payload)
        inputs = dict(payload.get("workflow_inputs", {}))
        positive = str(inputs.get("positive_prompt", "") or "")
        negative = str(inputs.get("negative_prompt", "") or "")

        if "text_artifact_risk" in result.issues:
            positive = f"{positive}, clean blank surfaces, no posters, no labels, no visible text, no signage"
            negative = _append_unique_negative(negative, ["text", "letters", "subtitles", "garbled typography", "distorted text", "signage text"])
        if "soft_or_blurry_keyframe" in result.issues:
            positive = f"{positive}, tack sharp focus, crisp micro detail, clean facial features, clear clothing seams"
            negative = _append_unique_negative(negative, ["blurry", "soft focus", "low resolution", "fuzzy face", "smudged details"])

        inputs["positive_prompt"] = positive
        inputs["negative_prompt"] = negative
        inputs["steps"] = max(int(inputs.get("steps", 30) or 30), 50)
        payload["workflow_inputs"] = inputs
        repair_jobs.append(job.model_copy(update={"job_id": f"{job.job_id}_FIX", "payload": payload}))
    return repair_jobs


def keyframe_qa_report_json(report: KeyframeQAReport, indent: int = 2) -> str:
    """Serialise a keyframe QA report."""

    return report.model_dump_json(indent=indent)


def keyframe_qa_markdown(report: KeyframeQAReport) -> str:
    """Human-readable keyframe QA report."""

    lines = [
        f"# Keyframe QA {report.source_id}",
        "",
        f"- Score: `{report.score:.3f}`",
        f"- Passes Gate: `{report.passes_gate}`",
        f"- Critical Issues: `{', '.join(report.critical_issues) if report.critical_issues else 'none'}`",
        "",
    ]
    for result in report.results:
        lines.extend(
            [
                f"## {result.shot_id}",
                "",
                f"- Score: `{result.score:.3f}`",
                f"- Recommendation: `{result.recommendation}`",
                f"- Issues: `{', '.join(result.issues) if result.issues else 'none'}`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def write_keyframe_qa_report(report: KeyframeQAReport, output_dir: str | Path) -> dict[str, Path]:
    """Write keyframe QA outputs to disk."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "keyframe_qa_report.json"
    markdown_path = output_dir / "keyframe_qa_report.md"
    json_path.write_text(keyframe_qa_report_json(report, indent=2), encoding="utf-8")
    markdown_path.write_text(keyframe_qa_markdown(report), encoding="utf-8")
    return {"report_json": json_path, "report_markdown": markdown_path}


def _append_unique_negative(base: str, extras: list[str]) -> str:
    parts = [part.strip() for part in base.split(",") if part.strip()]
    lowered = {part.lower() for part in parts}
    for extra in extras:
        if extra.lower() not in lowered:
            parts.append(extra)
            lowered.add(extra.lower())
    return ", ".join(parts)


def _text_artifact_score(image: Image.Image) -> float:
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    width, height = gray.size
    tiles_x = 10
    tiles_y = 10
    scores: list[float] = []

    for tile_y in range(tiles_y):
        for tile_x in range(tiles_x):
            left = int(width * tile_x / tiles_x)
            right = int(width * (tile_x + 1) / tiles_x)
            top = int(height * tile_y / tiles_y)
            bottom = int(height * (tile_y + 1) / tiles_y)
            gray_tile = gray.crop((left, top, right, bottom))
            edge_tile = edges.crop((left, top, right, bottom))
            score = max(_binary_text_patch_score(gray_tile), _edge_text_patch_score(edge_tile))
            scores.append(score)

    scores.sort(reverse=True)
    if not scores or scores[0] < 0.03:
        return 0.0
    return min(1.0, sum(scores[:6]) / 1.2)


def _binary_text_patch_score(tile: Image.Image) -> float:
    tile = tile.resize((48, 48))
    pixels = list(tile.tobytes())
    mean_value = sum(pixels) / len(pixels)
    threshold = max(12, min(180, int(mean_value * 0.72)))
    binary = [1 if value < threshold else 0 for value in pixels]
    fill_ratio = sum(binary) / len(binary)
    if fill_ratio < 0.003 or fill_ratio > 0.35:
        return 0.0
    variance = ImageStat.Stat(tile).var[0]
    transition_density = _transition_density(binary, *tile.size)
    if variance < 40:
        return 0.0
    return min(1.0, (transition_density * 2.5) + min(1.0, variance / 3000.0) * 0.35)


def _edge_text_patch_score(tile: Image.Image) -> float:
    tile = tile.resize((48, 48))
    pixels = list(tile.tobytes())
    binary = [1 if value > 48 else 0 for value in pixels]
    fill_ratio = sum(binary) / len(binary)
    if fill_ratio < 0.015 or fill_ratio > 0.30:
        return 0.0

    transition_density = _transition_density(binary, *tile.size)
    variance = ImageStat.Stat(tile).var[0]
    variance_score = min(1.0, variance / 5000.0)
    return min(1.0, (transition_density * 2.4) + variance_score * 0.25)


def _transition_density(binary: list[int], width: int, height: int) -> float:
    transitions = 0
    possible = 0
    for y in range(height):
        row = binary[y * width:(y + 1) * width]
        transitions += sum(1 for i in range(width - 1) if row[i] != row[i + 1])
        possible += width - 1
    for x in range(width):
        col = [binary[y * width + x] for y in range(height)]
        transitions += sum(1 for i in range(height - 1) if col[i] != col[i + 1])
        possible += height - 1
    return transitions / max(1, possible)
