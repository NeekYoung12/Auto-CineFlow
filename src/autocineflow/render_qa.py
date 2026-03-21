"""Render output manifest generation and QA scoring."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .delivery import StoryboardPackage


class RenderExpectation(BaseModel):
    """Expected render artifact entry derived from a storyboard package."""

    shot_id: str
    expected_filename: str
    expected_seed: int = Field(..., ge=0)
    expected_width: int = Field(..., ge=1)
    expected_height: int = Field(..., ge=1)
    expected_prompt_hash: str
    continuity_group: str
    reference_shot_id: str = ""
    output_path: str = ""
    status: str = "pending"
    actual_seed: Optional[int] = Field(default=None, ge=0)
    actual_width: Optional[int] = Field(default=None, ge=1)
    actual_height: Optional[int] = Field(default=None, ge=1)
    actual_prompt_hash: Optional[str] = None
    provider: str = "generic"
    notes: list[str] = Field(default_factory=list)


class ShotQAChecks(BaseModel):
    """Per-shot QA booleans."""

    rendered: bool
    naming_match: bool
    seed_match: bool
    dimensions_match: bool
    prompt_hash_match: bool
    continuity_reference_resolved: bool


class ShotQAResult(BaseModel):
    """Per-shot QA result."""

    shot_id: str
    checks: ShotQAChecks
    score: float = Field(..., ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)


class RenderQAReport(BaseModel):
    """Scene-level render QA report."""

    scene_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    min_score: float = Field(..., ge=0.0, le=1.0)
    passes_gate: bool
    coverage_rate: float = Field(..., ge=0.0, le=1.0)
    shot_results: list[ShotQAResult] = Field(default_factory=list)
    summary: dict[str, float] = Field(default_factory=dict)
    critical_issues: list[str] = Field(default_factory=list)


def prompt_hash(positive_prompt: str, negative_prompt: str) -> str:
    """Build a stable prompt hash for render QA."""

    return sha256(f"{positive_prompt}||{negative_prompt}".encode("utf-8")).hexdigest()


def build_render_manifest_template(package: StoryboardPackage) -> list[RenderExpectation]:
    """Create an expected render manifest from a storyboard package."""

    manifest: list[RenderExpectation] = []
    for shot in package.shots:
        manifest.append(
            RenderExpectation(
                shot_id=shot.shot_id,
                expected_filename=f"{shot.shot_id}.png",
                expected_seed=shot.render_seed,
                expected_width=package.render_preset.width,
                expected_height=package.render_preset.height,
                expected_prompt_hash=prompt_hash(shot.prompt, shot.negative_prompt),
                continuity_group=shot.continuity_group,
                reference_shot_id=shot.reference_shot_id,
            )
        )
    return manifest


def render_manifest_template_json(package: StoryboardPackage, indent: int = 2) -> str:
    """Serialise the expected render manifest to JSON."""

    return json.dumps(
        [entry.model_dump(mode="json") for entry in build_render_manifest_template(package)],
        indent=indent,
        ensure_ascii=False,
    )


def load_render_manifest(path: str | Path) -> list[RenderExpectation]:
    """Load a render manifest from disk."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [RenderExpectation.model_validate(item) for item in payload]


def render_qa_report(
    package: StoryboardPackage,
    manifest: list[RenderExpectation],
    min_score: float = 0.9,
) -> RenderQAReport:
    """Evaluate render outputs against expected shot metadata."""

    manifest_by_shot = {item.shot_id: item for item in manifest}
    shot_results: list[ShotQAResult] = []

    for shot in package.shots:
        item = manifest_by_shot.get(shot.shot_id)
        issues: list[str] = []
        if item is None:
            checks = ShotQAChecks(
                rendered=False,
                naming_match=False,
                seed_match=False,
                dimensions_match=False,
                prompt_hash_match=False,
                continuity_reference_resolved=False,
            )
            shot_results.append(
                ShotQAResult(
                    shot_id=shot.shot_id,
                    checks=checks,
                    score=0.0,
                    issues=["missing_manifest_entry"],
                )
            )
            continue

        rendered = item.status == "rendered" and bool(item.output_path)
        naming_match = Path(item.output_path).name == item.expected_filename if item.output_path else False
        seed_match = item.actual_seed == item.expected_seed if item.actual_seed is not None else False
        dimensions_match = (
            item.actual_width == item.expected_width and item.actual_height == item.expected_height
            if item.actual_width is not None and item.actual_height is not None
            else False
        )
        prompt_match = item.actual_prompt_hash == item.expected_prompt_hash if item.actual_prompt_hash else False
        continuity_resolved = True
        if item.reference_shot_id:
            reference = manifest_by_shot.get(item.reference_shot_id)
            continuity_resolved = bool(reference and reference.status == "rendered" and reference.output_path)

        checks = ShotQAChecks(
            rendered=rendered,
            naming_match=naming_match,
            seed_match=seed_match,
            dimensions_match=dimensions_match,
            prompt_hash_match=prompt_match,
            continuity_reference_resolved=continuity_resolved,
        )

        if not rendered:
            issues.append("not_rendered")
        if rendered and not naming_match:
            issues.append("filename_mismatch")
        if rendered and not seed_match:
            issues.append("seed_mismatch")
        if rendered and not dimensions_match:
            issues.append("dimension_mismatch")
        if rendered and not prompt_match:
            issues.append("prompt_hash_mismatch")
        if rendered and not continuity_resolved:
            issues.append("missing_reference_render")

        weights = {
            "rendered": 0.35,
            "naming_match": 0.1,
            "seed_match": 0.15,
            "dimensions_match": 0.15,
            "prompt_hash_match": 0.15,
            "continuity_reference_resolved": 0.1,
        }
        score = round(
            sum(
                getattr(checks, key) * weight
                for key, weight in weights.items()
            ),
            4,
        )
        shot_results.append(ShotQAResult(shot_id=shot.shot_id, checks=checks, score=score, issues=issues))

    if not shot_results:
        return RenderQAReport(
            scene_id=package.scene_id,
            score=0.0,
            min_score=min_score,
            passes_gate=False,
            coverage_rate=0.0,
            shot_results=[],
            summary={},
            critical_issues=["no_shots"],
        )

    summary = {
        "rendered_rate": round(sum(result.checks.rendered for result in shot_results) / len(shot_results), 4),
        "seed_match_rate": round(sum(result.checks.seed_match for result in shot_results) / len(shot_results), 4),
        "dimension_match_rate": round(sum(result.checks.dimensions_match for result in shot_results) / len(shot_results), 4),
        "prompt_match_rate": round(sum(result.checks.prompt_hash_match for result in shot_results) / len(shot_results), 4),
        "continuity_reference_rate": round(
            sum(result.checks.continuity_reference_resolved for result in shot_results) / len(shot_results),
            4,
        ),
    }
    score = round(sum(result.score for result in shot_results) / len(shot_results), 4)
    critical_issues = sorted(
        {
            issue
            for result in shot_results
            for issue in result.issues
            if issue in {"not_rendered", "seed_mismatch", "dimension_mismatch", "prompt_hash_mismatch"}
        }
    )
    return RenderQAReport(
        scene_id=package.scene_id,
        score=score,
        min_score=min_score,
        passes_gate=score >= min_score and not critical_issues,
        coverage_rate=summary["rendered_rate"],
        shot_results=shot_results,
        summary=summary,
        critical_issues=critical_issues,
    )


def render_qa_report_json(report: RenderQAReport, indent: int = 2) -> str:
    """Serialise a render QA report to JSON."""

    return report.model_dump_json(indent=indent)


def render_qa_review_markdown(report: RenderQAReport) -> str:
    """Export a human-readable render QA review document."""

    lines = [
        f"# Render QA {report.scene_id}",
        "",
        f"- Score: `{report.score:.3f}`",
        f"- Min Score: `{report.min_score:.3f}`",
        f"- Passes Gate: `{report.passes_gate}`",
        f"- Coverage: `{report.coverage_rate:.3f}`",
        "",
        "## Shot Results",
        "",
    ]
    for result in report.shot_results:
        lines.extend(
            [
                f"### {result.shot_id}",
                "",
                f"- Score: `{result.score:.3f}`",
                f"- Issues: `{', '.join(result.issues) if result.issues else 'none'}`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def write_render_qa_report(
    report: RenderQAReport,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write render QA outputs to disk."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    report_json_path = target_dir / "render_qa_report.json"
    review_markdown_path = target_dir / "render_qa_review.md"
    report_json_path.write_text(render_qa_report_json(report, indent=2), encoding="utf-8")
    review_markdown_path.write_text(render_qa_review_markdown(report), encoding="utf-8")
    return {
        "report_json": report_json_path,
        "review_markdown": review_markdown_path,
    }


def main() -> int:
    """CLI entry point for render QA."""

    import argparse

    from .delivery import StoryboardPackage

    parser = argparse.ArgumentParser(description="Evaluate render outputs against storyboard expectations.")
    parser.add_argument("--package-file", required=True, help="Path to storyboard_package.json")
    parser.add_argument("--manifest-file", required=True, help="Path to render manifest JSON")
    parser.add_argument("--output-dir", required=True, help="Directory for QA outputs")
    parser.add_argument("--min-score", type=float, default=0.9, help="Minimum passing score")
    args = parser.parse_args()

    package = StoryboardPackage.model_validate_json(Path(args.package_file).read_text(encoding="utf-8"))
    manifest = load_render_manifest(args.manifest_file)
    report = render_qa_report(package, manifest, min_score=args.min_score)
    output_files = write_render_qa_report(report, args.output_dir)
    print(
        json.dumps(
            {
                "scene_id": report.scene_id,
                "score": report.score,
                "passes_gate": report.passes_gate,
                "output_files": {key: str(value) for key, value in output_files.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
