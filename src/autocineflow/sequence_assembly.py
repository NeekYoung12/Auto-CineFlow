"""Sequence assembly planning for stitching many short clips into a long previs."""

from __future__ import annotations

import json
import math
from pathlib import Path

from pydantic import BaseModel, Field

from .delivery import StoryboardPackage


class SequenceClip(BaseModel):
    """Single clip in an assembled scene sequence."""

    clip_id: str
    shot_id: str
    timeline_start_seconds: float = Field(..., ge=0.0)
    source_duration_seconds: float = Field(..., ge=0.1)
    editorial_duration_seconds: float = Field(..., ge=0.1)
    expected_asset_path: str
    transition: str = "CUT"


class SequenceAssemblyPlan(BaseModel):
    """Plan for concatenating many short clip renders into one scene sequence."""

    scene_id: str
    clip_count: int = Field(..., ge=1)
    target_duration_seconds: float = Field(..., ge=0.1)
    editorial_duration_seconds: float = Field(..., ge=0.1)
    clips: list[SequenceClip] = Field(default_factory=list)


def recommended_shot_count(
    target_duration_seconds: float,
    clip_duration_seconds: float = 4.0,
    min_shots: int = 8,
    max_shots: int = 24,
) -> int:
    """Recommend a finer-grained shot count for long-form assembled previs."""

    estimated = math.ceil(target_duration_seconds / max(clip_duration_seconds, 0.1))
    return max(min_shots, min(max_shots, estimated))


def build_sequence_assembly_plan(
    package: StoryboardPackage,
    artifacts_dir: str = "artifacts",
) -> SequenceAssemblyPlan:
    """Build a clip-by-clip assembly plan from package video segments."""

    clips: list[SequenceClip] = []
    timeline_start = 0.0
    for segment in package.video_segments:
        clip = SequenceClip(
            clip_id=segment.segment_id,
            shot_id=segment.shot_id,
            timeline_start_seconds=round(timeline_start, 2),
            source_duration_seconds=segment.generation_duration_seconds,
            editorial_duration_seconds=min(segment.generation_duration_seconds, segment.covered_timeline_seconds),
            expected_asset_path=str(Path(artifacts_dir) / f"{segment.shot_id}.mp4"),
        )
        clips.append(clip)
        timeline_start += clip.editorial_duration_seconds

    return SequenceAssemblyPlan(
        scene_id=package.scene_id,
        clip_count=len(clips),
        target_duration_seconds=round(sum(clip.source_duration_seconds for clip in clips), 2),
        editorial_duration_seconds=round(sum(clip.editorial_duration_seconds for clip in clips), 2),
        clips=clips,
    )


def sequence_assembly_json(plan: SequenceAssemblyPlan, indent: int = 2) -> str:
    """Serialise a sequence assembly plan to JSON."""

    return plan.model_dump_json(indent=indent)


def ffmpeg_concat_manifest(plan: SequenceAssemblyPlan) -> str:
    """Build an FFmpeg concat manifest."""

    return "".join(f"file '{clip.expected_asset_path.replace('\\', '/')}'\n" for clip in plan.clips)


def ffmpeg_concat_script(plan: SequenceAssemblyPlan) -> str:
    """Build a Windows-friendly FFmpeg concat script."""

    output_name = f"{plan.scene_id.lower()}_sequence.mp4"
    return (
        "@echo off\n"
        "setlocal\n"
        "ffmpeg -y -f concat -safe 0 -i ffmpeg_concat_manifest.txt -c copy "
        f"{output_name}\n"
        "endlocal\n"
    )


def write_sequence_assembly_plan(
    plan: SequenceAssemblyPlan,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write sequence assembly planning assets to disk."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "sequence_assembly.json"
    manifest_path = output_dir / "ffmpeg_concat_manifest.txt"
    script_path = output_dir / "concat_with_ffmpeg.bat"

    json_path.write_text(sequence_assembly_json(plan, indent=2), encoding="utf-8")
    manifest_path.write_text(ffmpeg_concat_manifest(plan), encoding="utf-8")
    script_path.write_text(ffmpeg_concat_script(plan), encoding="utf-8")
    return {
        "assembly_json": json_path,
        "concat_manifest": manifest_path,
        "concat_script": script_path,
    }
