"""Provider-oriented payload bundles derived from delivery packages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .delivery import StoryboardPackage
from .runninghub_workflows import (
    build_runninghub_video_bundle,
    runninghub_workflow_suite,
    sanitize_runninghub_prompt,
    strengthen_runninghub_negative_prompt,
)


def _remote_urls(paths: list[str]) -> list[str]:
    return [path for path in paths if path.startswith(("http://", "https://"))]


def _local_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if not path.startswith(("http://", "https://"))]


def automatic1111_bundle(package: StoryboardPackage) -> list[dict[str, Any]]:
    """Build Automatic1111-style txt2img payloads for each render job."""

    jobs: list[dict[str, Any]] = []
    for render_job in package.render_queue:
        character_refs = list(render_job.metadata.get("character_reference_images", []))
        scene_refs = list(render_job.metadata.get("scene_reference_images", []))
        faceid_profile_ids = dict(render_job.metadata.get("faceid_profile_ids", {}))
        alwayson_scripts: dict[str, Any] = {
            "controlnet": {
                "args": [
                    {
                        "char_id": point["char_id"],
                        "x": point["x"],
                        "y": point["y"],
                    }
                    for point in render_job.controlnet_points
                ]
            }
        }
        if character_refs or faceid_profile_ids:
            alwayson_scripts["ip_adapter_faceid"] = {
                "args": [
                    {
                        "char_id": char_id,
                        "faceid_profile_id": profile_id,
                        "reference_images": character_refs,
                        "weight": 0.85,
                    }
                    for char_id, profile_id in faceid_profile_ids.items()
                ]
                or [{"reference_images": character_refs, "weight": 0.85}]
            }
        if scene_refs:
            alwayson_scripts["scene_reference"] = {
                "args": [{"reference_images": scene_refs, "weight": 0.65}]
            }
        jobs.append(
            {
                "job_id": render_job.job_id,
                "shot_id": render_job.shot_id,
                "prompt": render_job.prompt,
                "negative_prompt": render_job.negative_prompt,
                "seed": render_job.render_seed,
                "sampler_name": package.render_preset.sampler,
                "steps": package.render_preset.steps,
                "cfg_scale": package.render_preset.cfg_scale,
                "width": render_job.width,
                "height": render_job.height,
                "batch_size": 1,
                "n_iter": 1,
                "alwayson_scripts": alwayson_scripts,
                "metadata": dict(render_job.metadata),
            }
        )
    return jobs


def comfyui_bundle(package: StoryboardPackage) -> list[dict[str, Any]]:
    """Build ComfyUI-oriented prompt bundles for each render job."""

    jobs: list[dict[str, Any]] = []
    for render_job in package.render_queue:
        jobs.append(
            {
                "job_id": render_job.job_id,
                "shot_id": render_job.shot_id,
                "workflow": {
                    "positive_prompt": render_job.prompt,
                    "negative_prompt": render_job.negative_prompt,
                    "seed": render_job.render_seed,
                    "steps": package.render_preset.steps,
                    "cfg": package.render_preset.cfg_scale,
                    "sampler_name": package.render_preset.sampler,
                    "width": render_job.width,
                    "height": render_job.height,
                    "controlnet_points": list(render_job.controlnet_points),
                    "character_reference_images": list(render_job.metadata.get("character_reference_images", [])),
                    "scene_reference_images": list(render_job.metadata.get("scene_reference_images", [])),
                    "faceid_profile_ids": dict(render_job.metadata.get("faceid_profile_ids", {})),
                    "selected_character_views": dict(render_job.metadata.get("selected_character_views", {})),
                },
                "metadata": dict(render_job.metadata),
            }
        )
    return jobs


def runninghub_faceid_bundle(package: StoryboardPackage) -> list[dict[str, Any]]:
    """Build RunningHub-ready workflow bundles emphasizing FaceID consistency."""

    jobs: list[dict[str, Any]] = []
    for render_job in package.render_queue:
        character_refs = list(render_job.metadata.get("character_reference_images", []))
        scene_refs = list(render_job.metadata.get("scene_reference_images", []))
        jobs.append(
            {
                "job_id": render_job.job_id,
                "shot_id": render_job.shot_id,
                "workflow_key": "rh_shot_keyframe_faceid_v1",
                "workflow_id_env": "RUNNINGHUB_WORKFLOW_RH_SHOT_KEYFRAME_FACEID_V1",
                "workflow_family": "runninghub_comfyui_faceid",
                "workflow_inputs": {
                    "positive_prompt": sanitize_runninghub_prompt(render_job.prompt),
                    "negative_prompt": strengthen_runninghub_negative_prompt(render_job.negative_prompt),
                    "seed": render_job.render_seed,
                    "width": render_job.width,
                    "height": render_job.height,
                    "steps": package.render_preset.steps,
                    "cfg": package.render_preset.cfg_scale,
                    "sampler_name": package.render_preset.sampler,
                    "controlnet_points": list(render_job.controlnet_points),
                    "character_reference_images": character_refs,
                    "scene_reference_images": scene_refs,
                    "faceid_profile_ids": dict(render_job.metadata.get("faceid_profile_ids", {})),
                    "selected_character_views": dict(render_job.metadata.get("selected_character_views", {})),
                },
                "metadata": dict(render_job.metadata),
            }
        )
    return jobs


def volcengine_seedream_bundle(package: StoryboardPackage) -> list[dict[str, Any]]:
    """Build Seedream-ready reference bundles for Volcengine ARK/LAS image generation."""

    jobs: list[dict[str, Any]] = []
    for render_job in package.render_queue:
        character_refs = list(render_job.metadata.get("character_reference_images", []))
        scene_refs = list(render_job.metadata.get("scene_reference_images", []))
        all_refs = character_refs + scene_refs
        remote_refs = _remote_urls(all_refs)[:10]
        request = {
            "model": "doubao-seedream-4-0-250828",
            "prompt": render_job.prompt,
            "negative_prompt": render_job.negative_prompt,
            "seed": render_job.render_seed,
            "size": f"{render_job.width}x{render_job.height}",
            "response_format": "url",
            "watermark": True,
        }
        if remote_refs:
            request["image"] = remote_refs
        jobs.append(
            {
                "job_id": render_job.job_id,
                "shot_id": render_job.shot_id,
                "provider": "volcengine_seedream_reference",
                "request": request,
                "metadata": {
                    **dict(render_job.metadata),
                    "reference_image_urls": remote_refs,
                    "local_reference_images": _local_paths(all_refs),
                },
            }
        )
    return jobs


def automatic1111_bundle_json(package: StoryboardPackage, indent: int = 2) -> str:
    """Serialise Automatic1111 bundles to JSON."""

    return json.dumps(automatic1111_bundle(package), indent=indent, ensure_ascii=False)


def comfyui_bundle_json(package: StoryboardPackage, indent: int = 2) -> str:
    """Serialise ComfyUI bundles to JSON."""

    return json.dumps(comfyui_bundle(package), indent=indent, ensure_ascii=False)


def runninghub_faceid_bundle_json(package: StoryboardPackage, indent: int = 2) -> str:
    """Serialise RunningHub FaceID bundles to JSON."""

    return json.dumps(runninghub_faceid_bundle(package), indent=indent, ensure_ascii=False)


def runninghub_video_bundle_json(package: StoryboardPackage, mode: str = "auto", indent: int = 2) -> str:
    """Serialise RunningHub video workflow plans to JSON."""

    return json.dumps(
        [item.model_dump(mode="json") for item in build_runninghub_video_bundle(package, mode=mode)],
        indent=indent,
        ensure_ascii=False,
    )


def runninghub_workflow_suite_json(indent: int = 2) -> str:
    """Serialise the curated RunningHub workflow suite."""

    return runninghub_workflow_suite().model_dump_json(indent=indent)


def volcengine_seedream_bundle_json(package: StoryboardPackage, indent: int = 2) -> str:
    """Serialise Volcengine Seedream bundles to JSON."""

    return json.dumps(volcengine_seedream_bundle(package), indent=indent, ensure_ascii=False)


def write_provider_payloads(package: StoryboardPackage, output_dir: str | Path) -> dict[str, Path]:
    """Write provider-oriented payload bundles to disk."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    automatic1111_path = target_dir / "automatic1111_txt2img.json"
    comfyui_path = target_dir / "comfyui_prompt_bundle.json"
    runninghub_path = target_dir / "runninghub_faceid_bundle.json"
    runninghub_workflow_suite_path = target_dir / "runninghub_workflow_suite.json"
    runninghub_video_auto_path = target_dir / "runninghub_video_auto_bundle.json"
    runninghub_video_quality_path = target_dir / "runninghub_video_quality_bundle.json"
    runninghub_video_fast_path = target_dir / "runninghub_video_fast_bundle.json"
    volcengine_path = target_dir / "volcengine_seedream_bundle.json"

    automatic1111_path.write_text(automatic1111_bundle_json(package, indent=2), encoding="utf-8")
    comfyui_path.write_text(comfyui_bundle_json(package, indent=2), encoding="utf-8")
    runninghub_path.write_text(runninghub_faceid_bundle_json(package, indent=2), encoding="utf-8")
    runninghub_workflow_suite_path.write_text(runninghub_workflow_suite_json(indent=2), encoding="utf-8")
    runninghub_video_auto_path.write_text(runninghub_video_bundle_json(package, mode="auto", indent=2), encoding="utf-8")
    runninghub_video_quality_path.write_text(runninghub_video_bundle_json(package, mode="quality", indent=2), encoding="utf-8")
    runninghub_video_fast_path.write_text(runninghub_video_bundle_json(package, mode="fast", indent=2), encoding="utf-8")
    volcengine_path.write_text(volcengine_seedream_bundle_json(package, indent=2), encoding="utf-8")

    return {
        "automatic1111": automatic1111_path,
        "comfyui": comfyui_path,
        "runninghub_faceid": runninghub_path,
        "runninghub_workflow_suite": runninghub_workflow_suite_path,
        "runninghub_video_auto": runninghub_video_auto_path,
        "runninghub_video_quality": runninghub_video_quality_path,
        "runninghub_video_fast": runninghub_video_fast_path,
        "volcengine_seedream": volcengine_path,
    }
