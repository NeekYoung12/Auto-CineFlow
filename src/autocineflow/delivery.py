"""Production delivery builders for storyboard manifests and render queues."""

from __future__ import annotations

import csv
import json
import re
from hashlib import sha256
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .models import BeatType, SceneContext, ShotBlock, ShotType


class RenderPreset(BaseModel):
    """Generic render preset shared across exported shots."""

    width: int = Field(default=1536, ge=64)
    height: int = Field(default=864, ge=64)
    fps: int = Field(default=24, ge=1)
    sampler: str = Field(default="dpmpp_2m")
    steps: int = Field(default=30, ge=1)
    cfg_scale: float = Field(default=6.5, ge=0.0)
    seed_base: int = Field(default=420000, ge=0)


class CharacterBibleEntry(BaseModel):
    """Reusable continuity entry for each tracked character."""

    char_id: str
    visual_anchor: str
    continuity_tag: str
    preferred_facing: str
    default_seed: int = Field(..., ge=0)


class DeliveryShot(BaseModel):
    """Production-facing shot entry with timing and prompt metadata."""

    shot_id: str
    shot_slug: str
    scene_id: str
    shot_index: int
    beat_type: str
    shot_type: str
    duration_seconds: float = Field(..., ge=0.1)
    fps: int = Field(..., ge=1)
    frame_count: int = Field(..., ge=1)
    timeline_in: str
    timeline_out: str
    focal_length_mm: int = Field(..., ge=1)
    render_seed: int = Field(..., ge=0)
    continuity_group: str
    reference_shot_id: str = ""
    axis_side: str
    primary_subjects: list[str] = Field(default_factory=list)
    scene_location: str = ""
    scene_tags: list[str] = Field(default_factory=list)
    prompt: str
    negative_prompt: str
    controlnet_points: list[dict[str, float | str]] = Field(default_factory=list)
    story_purpose: str = ""
    staging_note: str = ""
    nose_room_note: str = ""


class RenderJob(BaseModel):
    """Generic render queue item for downstream image/video generation."""

    job_id: str
    shot_id: str
    provider: str = "generic_sd_video"
    width: int
    height: int
    fps: int
    frame_count: int
    duration_seconds: float
    render_seed: int = Field(..., ge=0)
    prompt: str
    negative_prompt: str
    controlnet_points: list[dict[str, float | str]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StoryboardPackage(BaseModel):
    """Packaged production artifact containing manifest and render jobs."""

    project_name: str
    scene_id: str
    scene_location: str = ""
    detected_emotion: str = "neutral"
    analysis_source: str = "rule_based"
    generated_at: str
    total_duration_seconds: float = Field(..., ge=0.1)
    render_preset: RenderPreset
    readiness_report: dict[str, bool] = Field(default_factory=dict)
    character_bible: list[CharacterBibleEntry] = Field(default_factory=list)
    shots: list[DeliveryShot] = Field(default_factory=list)
    render_queue: list[RenderJob] = Field(default_factory=list)


_BASE_DURATIONS: dict[ShotType, float] = {
    ShotType.MASTER_SHOT: 4.0,
    ShotType.MEDIUM_SHOT: 3.0,
    ShotType.MCU: 2.6,
    ShotType.CLOSE_UP: 2.3,
    ShotType.OVER_SHOULDER: 3.1,
}

_BEAT_DURATION_BONUS: dict[BeatType, float] = {
    BeatType.ESTABLISH: 0.9,
    BeatType.RELATION: 0.4,
    BeatType.BUILD: 0.2,
    BeatType.ESCALATION: 0.3,
    BeatType.REACTION: 0.4,
    BeatType.RESOLUTION: 0.5,
}


def slugify(value: str, default: str = "scene") -> str:
    """Convert free-form text into a stable ASCII slug."""

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or default


def stable_seed(*parts: str | int) -> int:
    """Build a deterministic seed from stable identifiers."""

    payload = "::".join(str(part) for part in parts)
    digest = sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def estimate_shot_duration(shot: ShotBlock) -> float:
    """Estimate editorial duration for a shot."""

    base = _BASE_DURATIONS[shot.framing.shot_type]
    motion_bonus = shot.motion_instruction.intensity * 0.8
    beat_bonus = _BEAT_DURATION_BONUS.get(shot.beat_type, 0.0) if shot.beat_type else 0.0
    prompt_bonus = 0.15 if shot.framing.shot_type == ShotType.CLOSE_UP and shot.story_purpose else 0.0
    duration = round(base + motion_bonus + beat_bonus + prompt_bonus, 2)
    return max(duration, 1.0)


def seconds_to_timecode(total_seconds: float, fps: int) -> str:
    """Convert seconds to a non-drop-frame SMPTE timecode."""

    total_frames = round(total_seconds * fps)
    frames = total_frames % fps
    total_seconds_int = total_frames // fps
    seconds = total_seconds_int % 60
    total_minutes = total_seconds_int // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"


def build_character_bible(
    context: SceneContext,
    render_preset: RenderPreset,
) -> list[CharacterBibleEntry]:
    """Create a stable character continuity list for the scene."""

    return [
        CharacterBibleEntry(
            char_id=character.char_id,
            visual_anchor=character.visual_anchor,
            continuity_tag=slugify(f"{context.scene_id}-{character.char_id}-{character.visual_anchor}"),
            preferred_facing=character.facing.value,
            default_seed=render_preset.seed_base + stable_seed(context.scene_id, character.char_id),
        )
        for character in context.characters
    ]


def infer_reference_shot_id(existing_shots: list[DeliveryShot], shot: ShotBlock) -> str:
    """Link a shot to the most recent shot covering the same primary subject."""

    subject_set = set(shot.framing.subjects)
    if not subject_set:
        return existing_shots[-1].shot_id if existing_shots else ""

    for previous in reversed(existing_shots):
        if subject_set.intersection(previous.primary_subjects):
            return previous.shot_id
    return existing_shots[-1].shot_id if existing_shots else ""


def build_delivery_shot(
    context: SceneContext,
    shot: ShotBlock,
    render_preset: RenderPreset,
    start_seconds: float,
    existing_shots: list[DeliveryShot] | None = None,
) -> DeliveryShot:
    """Build a single delivery shot from a pipeline shot block."""

    shot_code = f"{context.scene_id}_SH{shot.shot_index + 1:03d}"
    shot_slug = slugify(
        f"{shot_code}-{shot.framing.shot_type.value}-{shot.story_purpose or shot.beat_type or 'shot'}",
        default=shot_code.lower(),
    )
    duration_seconds = estimate_shot_duration(shot)
    frame_count = max(1, round(duration_seconds * render_preset.fps))
    timeline_in = seconds_to_timecode(start_seconds, render_preset.fps)
    timeline_out = seconds_to_timecode(start_seconds + duration_seconds, render_preset.fps)
    continuity_group = slugify(
        "-".join(shot.framing.subjects) if shot.framing.subjects else f"{context.scene_id}-ensemble",
        default=f"{context.scene_id.lower()}-ensemble",
    )
    render_seed = render_preset.seed_base + stable_seed(context.scene_id, shot_code, continuity_group)
    reference_shot_id = infer_reference_shot_id(existing_shots or [], shot)

    return DeliveryShot(
        shot_id=shot_code,
        shot_slug=shot_slug,
        scene_id=context.scene_id,
        shot_index=shot.shot_index,
        beat_type=shot.beat_type.value if shot.beat_type else "",
        shot_type=shot.framing.shot_type.value,
        duration_seconds=duration_seconds,
        fps=render_preset.fps,
        frame_count=frame_count,
        timeline_in=timeline_in,
        timeline_out=timeline_out,
        focal_length_mm=shot.framing.focal_length_mm,
        render_seed=render_seed,
        continuity_group=continuity_group,
        reference_shot_id=reference_shot_id,
        axis_side=shot.camera_angle.axis_side.value,
        primary_subjects=list(shot.framing.subjects),
        scene_location=shot.scene_location or context.scene_location,
        scene_tags=list(shot.scene_tags or context.scene_tags),
        prompt=shot.sd_prompt,
        negative_prompt=shot.negative_prompt,
        controlnet_points=list(shot.controlnet_points),
        story_purpose=shot.story_purpose,
        staging_note=shot.composition.staging if shot.composition else "",
        nose_room_note=shot.composition.nose_room if shot.composition else "",
    )


def build_render_job(delivery_shot: DeliveryShot, render_preset: RenderPreset) -> RenderJob:
    """Build a generic render queue entry from a delivery shot."""

    return RenderJob(
        job_id=f"JOB_{delivery_shot.shot_id}",
        shot_id=delivery_shot.shot_id,
        width=render_preset.width,
        height=render_preset.height,
        fps=render_preset.fps,
        frame_count=delivery_shot.frame_count,
        duration_seconds=delivery_shot.duration_seconds,
        render_seed=delivery_shot.render_seed,
        prompt=delivery_shot.prompt,
        negative_prompt=delivery_shot.negative_prompt,
        controlnet_points=list(delivery_shot.controlnet_points),
        metadata={
            "scene_id": delivery_shot.scene_id,
            "shot_index": delivery_shot.shot_index,
            "shot_slug": delivery_shot.shot_slug,
            "beat_type": delivery_shot.beat_type,
            "shot_type": delivery_shot.shot_type,
            "primary_subjects": list(delivery_shot.primary_subjects),
            "reference_shot_id": delivery_shot.reference_shot_id,
            "continuity_group": delivery_shot.continuity_group,
            "axis_side": delivery_shot.axis_side,
            "focal_length_mm": delivery_shot.focal_length_mm,
            "scene_location": delivery_shot.scene_location,
            "scene_tags": list(delivery_shot.scene_tags),
        },
    )


def build_storyboard_package(
    context: SceneContext,
    project_name: str = "Auto-CineFlow Project",
    render_preset: RenderPreset | None = None,
    readiness_report: dict[str, bool] | None = None,
    generated_at: str | None = None,
) -> StoryboardPackage:
    """Create a production-ready delivery package from a scene context."""

    render_preset = render_preset or RenderPreset()
    character_bible = build_character_bible(context, render_preset)
    delivery_shots: list[DeliveryShot] = []
    elapsed_seconds = 0.0
    for shot in context.shot_blocks:
        delivery_shot = build_delivery_shot(
            context,
            shot,
            render_preset,
            start_seconds=elapsed_seconds,
            existing_shots=delivery_shots,
        )
        delivery_shots.append(delivery_shot)
        elapsed_seconds += delivery_shot.duration_seconds
    render_queue = [build_render_job(shot, render_preset) for shot in delivery_shots]
    total_duration_seconds = round(sum(shot.duration_seconds for shot in delivery_shots), 2) or 0.1

    return StoryboardPackage(
        project_name=project_name,
        scene_id=context.scene_id,
        scene_location=context.scene_location,
        detected_emotion=context.detected_emotion,
        analysis_source=context.analysis_source,
        generated_at=generated_at or datetime.now(timezone.utc).isoformat(),
        total_duration_seconds=total_duration_seconds,
        render_preset=render_preset,
        readiness_report=readiness_report or {},
        character_bible=character_bible,
        shots=delivery_shots,
        render_queue=render_queue,
    )


def package_to_json(package: StoryboardPackage, indent: int = 2) -> str:
    """Serialise a storyboard package to JSON."""

    return package.model_dump_json(indent=indent)


def shotlist_to_csv(package: StoryboardPackage) -> str:
    """Export the packaged shot list to CSV."""

    buffer = StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "shot_id",
            "shot_slug",
            "shot_index",
            "beat_type",
            "shot_type",
            "duration_seconds",
            "frame_count",
            "timeline_in",
            "timeline_out",
            "focal_length_mm",
            "render_seed",
            "continuity_group",
            "reference_shot_id",
            "axis_side",
            "primary_subjects",
            "scene_location",
            "scene_tags",
            "story_purpose",
            "staging_note",
            "nose_room_note",
        ],
    )
    writer.writeheader()
    for shot in package.shots:
        writer.writerow(
            {
                "shot_id": shot.shot_id,
                "shot_slug": shot.shot_slug,
                "shot_index": shot.shot_index,
                "beat_type": shot.beat_type,
                "shot_type": shot.shot_type,
                "duration_seconds": shot.duration_seconds,
                "frame_count": shot.frame_count,
                "timeline_in": shot.timeline_in,
                "timeline_out": shot.timeline_out,
                "focal_length_mm": shot.focal_length_mm,
                "render_seed": shot.render_seed,
                "continuity_group": shot.continuity_group,
                "reference_shot_id": shot.reference_shot_id,
                "axis_side": shot.axis_side,
                "primary_subjects": "|".join(shot.primary_subjects),
                "scene_location": shot.scene_location,
                "scene_tags": "|".join(shot.scene_tags),
                "story_purpose": shot.story_purpose,
                "staging_note": shot.staging_note,
                "nose_room_note": shot.nose_room_note,
            }
        )
    return buffer.getvalue()


def render_queue_to_json(package: StoryboardPackage, indent: int = 2) -> str:
    """Serialise just the render queue to JSON."""

    return json.dumps(
        [job.model_dump(mode="json") for job in package.render_queue],
        indent=indent,
        ensure_ascii=False,
    )


def character_bible_to_json(package: StoryboardPackage, indent: int = 2) -> str:
    """Serialise the scene character bible to JSON."""

    return json.dumps(
        [entry.model_dump(mode="json") for entry in package.character_bible],
        indent=indent,
        ensure_ascii=False,
    )


def storyboard_review_markdown(package: StoryboardPackage) -> str:
    """Export a human-readable markdown review document for editorial approval."""

    lines = [
        f"# {package.project_name}",
        "",
        f"- Scene ID: `{package.scene_id}`",
        f"- Location: `{package.scene_location or 'unspecified'}`",
        f"- Emotion: `{package.detected_emotion}`",
        f"- Analysis Source: `{package.analysis_source}`",
        f"- Total Duration: `{package.total_duration_seconds:.2f}s`",
        "",
        "## Character Bible",
        "",
    ]
    for character in package.character_bible:
        lines.append(
            f"- `{character.char_id}`: {character.visual_anchor} | continuity `{character.continuity_tag}` | seed `{character.default_seed}`"
        )

    lines.extend(["", "## Shot List", ""])
    for shot in package.shots:
        lines.extend(
            [
                f"### {shot.shot_id} · {shot.shot_type}",
                "",
                f"- Timeline: `{shot.timeline_in}` -> `{shot.timeline_out}`",
                f"- Duration: `{shot.duration_seconds:.2f}s` ({shot.frame_count} frames @ {shot.fps} fps)",
                f"- Subjects: `{', '.join(shot.primary_subjects) or 'ensemble'}`",
                f"- Continuity: group `{shot.continuity_group}` | seed `{shot.render_seed}` | reference `{shot.reference_shot_id or 'none'}`",
                f"- Purpose: {shot.story_purpose or 'n/a'}",
                f"- Staging: {shot.staging_note or 'n/a'}",
                f"- Nose Room: {shot.nose_room_note or 'n/a'}",
                "",
                "**Prompt**",
                "",
                shot.prompt,
                "",
                "**Negative Prompt**",
                "",
                shot.negative_prompt,
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def edl_text(package: StoryboardPackage) -> str:
    """Export a simple CMX-style EDL for editorial review."""

    lines = [f"TITLE: {package.project_name} {package.scene_id}", "FCM: NON-DROP FRAME"]
    for index, shot in enumerate(package.shots, start=1):
        lines.append(
            f"{index:03d}  AX       V     C        "
            f"{shot.timeline_in} {shot.timeline_out} {shot.timeline_in} {shot.timeline_out}"
        )
        lines.append(f"* FROM CLIP NAME: {shot.shot_id}")
        lines.append(f"* COMMENT: {shot.shot_type} | {shot.story_purpose}")
    return "\n".join(lines) + "\n"


def write_storyboard_package(
    package: StoryboardPackage,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write manifest, shot list and render queue to disk."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = target_dir / "storyboard_package.json"
    shotlist_path = target_dir / "shotlist.csv"
    render_queue_path = target_dir / "render_queue.json"
    character_bible_path = target_dir / "character_bible.json"
    edl_path = target_dir / "timeline.edl"
    review_markdown_path = target_dir / "storyboard_review.md"

    manifest_path.write_text(package_to_json(package, indent=2), encoding="utf-8")
    shotlist_path.write_text(shotlist_to_csv(package), encoding="utf-8", newline="")
    render_queue_path.write_text(render_queue_to_json(package, indent=2), encoding="utf-8")
    character_bible_path.write_text(character_bible_to_json(package, indent=2), encoding="utf-8")
    edl_path.write_text(edl_text(package), encoding="utf-8")
    review_markdown_path.write_text(storyboard_review_markdown(package), encoding="utf-8")

    return {
        "manifest": manifest_path,
        "shotlist": shotlist_path,
        "render_queue": render_queue_path,
        "character_bible": character_bible_path,
        "edl": edl_path,
        "review_markdown": review_markdown_path,
    }


def main() -> int:
    """Package a single scene description from the command line."""

    import argparse

    from .pipeline import CineFlowPipeline

    parser = argparse.ArgumentParser(description="Package a scene into delivery assets.")
    parser.add_argument("--description", required=True, help="Scene description")
    parser.add_argument("--scene-id", default="SCENE_01", help="Scene identifier")
    parser.add_argument("--project-name", default="Auto-CineFlow Project", help="Project name")
    parser.add_argument("--output-dir", required=True, help="Directory for exported assets")
    parser.add_argument("--config-path", default=None, help="Path to OpenAI-compatible config")
    parser.add_argument("--num-shots", type=int, default=5, help="Number of generated shots")
    parser.add_argument("--offline", action="store_true", help="Disable LLM analysis")
    parser.add_argument("--model", default="gpt-5", help="LLM model name")
    args = parser.parse_args()

    pipeline = CineFlowPipeline(config_path=args.config_path, model=args.model)
    context = pipeline.run(
        description=args.description,
        num_shots=args.num_shots,
        scene_id=args.scene_id,
        use_llm=not args.offline,
    )
    package = pipeline.build_storyboard_package(context, project_name=args.project_name)
    files = pipeline.write_delivery_package(package, args.output_dir)
    print(
        json.dumps(
            {
                "scene_id": context.scene_id,
                "analysis_source": context.analysis_source,
                "output_files": {key: str(value) for key, value in files.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
