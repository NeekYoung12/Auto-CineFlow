"""Local FFmpeg-based video enhancement for previs clips."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from .result_ingest import ArtifactDownloadBatch


class VideoMetadata(BaseModel):
    """Minimal probed metadata for a local video file."""

    path: str
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    codec_name: str = ""


class VideoEnhancePlanItem(BaseModel):
    """One planned local enhancement pass."""

    shot_id: str
    input_path: str
    output_path: str
    preset: str
    target_width: int = Field(..., ge=1)
    filter_chain: str
    metadata: VideoMetadata


class VideoEnhanceResult(BaseModel):
    """Result of running one enhancement pass."""

    shot_id: str
    input_path: str
    output_path: str
    enhanced: bool
    preset: str
    command: list[str] = Field(default_factory=list)
    message: str = ""


class VideoEnhanceReport(BaseModel):
    """Aggregate report for local enhancement passes."""

    source_id: str
    preset: str
    items: list[VideoEnhancePlanItem] = Field(default_factory=list)
    results: list[VideoEnhanceResult] = Field(default_factory=list)


def probe_video_metadata(path: str | Path, ffprobe_bin: str = "ffprobe") -> VideoMetadata:
    """Probe a video file using ffprobe."""

    if shutil.which(ffprobe_bin) is None:
        raise FileNotFoundError(f"FFprobe binary not found: {ffprobe_bin}")

    input_path = Path(path)
    completed = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name:format=duration",
            "-of",
            "json",
            str(input_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "ffprobe failed")

    payload = json.loads(completed.stdout or "{}")
    stream = (payload.get("streams") or [{}])[0]
    format_payload = payload.get("format", {}) or {}
    return VideoMetadata(
        path=str(input_path),
        width=int(stream.get("width", 0) or 0),
        height=int(stream.get("height", 0) or 0),
        duration_seconds=float(format_payload.get("duration", 0.0) or 0.0),
        codec_name=str(stream.get("codec_name", "") or ""),
    )


def build_video_enhance_plan(
    downloads: ArtifactDownloadBatch,
    output_dir: str | Path,
    preset: str = "production_hd",
    ffprobe_bin: str = "ffprobe",
) -> list[VideoEnhancePlanItem]:
    """Build enhancement plans for downloaded MP4 clips."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan: list[VideoEnhancePlanItem] = []

    for record in downloads.records:
        if not record.downloaded or not record.output_path.lower().endswith(".mp4"):
            continue
        metadata = probe_video_metadata(record.output_path, ffprobe_bin=ffprobe_bin)
        target_width = _target_width_for_metadata(metadata)
        output_path = output_dir / f"{record.shot_id}.mp4"
        plan.append(
            VideoEnhancePlanItem(
                shot_id=record.shot_id,
                input_path=record.output_path,
                output_path=str(output_path),
                preset=preset,
                target_width=target_width,
                filter_chain=_filter_chain_for_metadata(metadata, target_width=target_width, preset=preset),
                metadata=metadata,
            )
        )
    return plan


def enhance_video_with_ffmpeg(
    item: VideoEnhancePlanItem,
    ffmpeg_bin: str = "ffmpeg",
) -> VideoEnhanceResult:
    """Run one FFmpeg enhancement pass."""

    if shutil.which(ffmpeg_bin) is None:
        raise FileNotFoundError(f"FFmpeg binary not found: {ffmpeg_bin}")

    output_path = Path(item.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        item.input_path,
        "-vf",
        item.filter_chain,
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "17",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        item.output_path,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    enhanced = completed.returncode == 0 and output_path.exists()
    message = (completed.stderr or completed.stdout)[-1200:].strip()
    return VideoEnhanceResult(
        shot_id=item.shot_id,
        input_path=item.input_path,
        output_path=item.output_path if enhanced else "",
        enhanced=enhanced,
        preset=item.preset,
        command=command,
        message=message,
    )


def enhance_videos(
    downloads: ArtifactDownloadBatch,
    output_dir: str | Path,
    preset: str = "production_hd",
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> VideoEnhanceReport:
    """Enhance all downloaded MP4 clips and return a structured report."""

    items = build_video_enhance_plan(downloads, output_dir, preset=preset, ffprobe_bin=ffprobe_bin)
    results = [enhance_video_with_ffmpeg(item, ffmpeg_bin=ffmpeg_bin) for item in items]
    return VideoEnhanceReport(
        source_id=downloads.source_id,
        preset=preset,
        items=items,
        results=results,
    )


def video_enhance_report_json(report: VideoEnhanceReport, indent: int = 2) -> str:
    """Serialise a video enhancement report."""

    return report.model_dump_json(indent=indent)


def video_enhance_report_markdown(report: VideoEnhanceReport) -> str:
    """Human-readable enhancement report."""

    lines = [
        f"# Video Enhance {report.source_id}",
        "",
        f"- Preset: `{report.preset}`",
        f"- Planned Clips: `{len(report.items)}`",
        f"- Enhanced Clips: `{sum(result.enhanced for result in report.results)}`",
        "",
    ]
    for result in report.results:
        lines.extend(
            [
                f"## {result.shot_id}",
                "",
                f"- Enhanced: `{result.enhanced}`",
                f"- Output: `{result.output_path or 'n/a'}`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def write_video_enhance_report(report: VideoEnhanceReport, output_dir: str | Path) -> dict[str, Path]:
    """Write enhancement report assets to disk."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "video_enhance_report.json"
    markdown_path = output_dir / "video_enhance_report.md"
    json_path.write_text(video_enhance_report_json(report, indent=2), encoding="utf-8")
    markdown_path.write_text(video_enhance_report_markdown(report), encoding="utf-8")
    return {
        "report_json": json_path,
        "report_markdown": markdown_path,
    }


def _target_width_for_metadata(metadata: VideoMetadata) -> int:
    if metadata.width <= 1024:
        return 1920
    if metadata.width <= 1280:
        return 2048
    return max(1920, metadata.width)


def _filter_chain_for_metadata(metadata: VideoMetadata, target_width: int, preset: str) -> str:
    scale_filter = f"scale={target_width}:-2:flags=lanczos"
    if preset == "production_hd":
        return ",".join(
            [
                "hqdn3d=1.2:1.2:6:6",
                scale_filter,
                "unsharp=5:5:0.65:3:3:0.25",
                "eq=contrast=1.03:saturation=1.02",
            ]
        )
    return ",".join([scale_filter, "unsharp=5:5:0.45:3:3:0.15"])
