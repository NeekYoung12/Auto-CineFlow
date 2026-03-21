"""Provider-oriented payload bundles derived from delivery packages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .delivery import StoryboardPackage


def automatic1111_bundle(package: StoryboardPackage) -> list[dict[str, Any]]:
    """Build Automatic1111-style txt2img payloads for each render job."""

    jobs: list[dict[str, Any]] = []
    for render_job in package.render_queue:
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
                "alwayson_scripts": {
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
                },
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
                },
                "metadata": dict(render_job.metadata),
            }
        )
    return jobs


def automatic1111_bundle_json(package: StoryboardPackage, indent: int = 2) -> str:
    """Serialise Automatic1111 bundles to JSON."""

    return json.dumps(automatic1111_bundle(package), indent=indent, ensure_ascii=False)


def comfyui_bundle_json(package: StoryboardPackage, indent: int = 2) -> str:
    """Serialise ComfyUI bundles to JSON."""

    return json.dumps(comfyui_bundle(package), indent=indent, ensure_ascii=False)


def write_provider_payloads(package: StoryboardPackage, output_dir: str | Path) -> dict[str, Path]:
    """Write provider-oriented payload bundles to disk."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    automatic1111_path = target_dir / "automatic1111_txt2img.json"
    comfyui_path = target_dir / "comfyui_prompt_bundle.json"

    automatic1111_path.write_text(automatic1111_bundle_json(package, indent=2), encoding="utf-8")
    comfyui_path.write_text(comfyui_bundle_json(package, indent=2), encoding="utf-8")

    return {
        "automatic1111": automatic1111_path,
        "comfyui": comfyui_path,
    }
