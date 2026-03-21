"""Production delivery builders for storyboard manifests and render queues."""

from __future__ import annotations

import csv
import json
import re
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


def build_delivery_shot(
    context: SceneContext,
    shot: ShotBlock,
    render_preset: RenderPreset,
    start_seconds: float,
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
    delivery_shots: list[DeliveryShot] = []
    elapsed_seconds = 0.0
    for shot in context.shot_blocks:
        delivery_shot = build_delivery_shot(context, shot, render_preset, start_seconds=elapsed_seconds)
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
    edl_path = target_dir / "timeline.edl"

    manifest_path.write_text(package_to_json(package, indent=2), encoding="utf-8")
    shotlist_path.write_text(shotlist_to_csv(package), encoding="utf-8", newline="")
    render_queue_path.write_text(render_queue_to_json(package, indent=2), encoding="utf-8")
    edl_path.write_text(edl_text(package), encoding="utf-8")

    return {
        "manifest": manifest_path,
        "shotlist": shotlist_path,
        "render_queue": render_queue_path,
        "edl": edl_path,
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
