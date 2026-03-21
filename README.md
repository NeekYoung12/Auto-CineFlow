# Auto-CineFlow

Auto-CineFlow is an LLM-driven cinematic storyboard generator for 1-2 character scenes. It converts scene descriptions into structured shot plans, render-ready prompts, geometry-safe blocking, and downstream delivery assets for image, video, and editorial workflows.

## Core Capabilities

- Stable character state with immutable `CHAR_A` / `CHAR_B` visual anchors
- 180-degree axis protection with scene-space camera placement
- Shot planning across `MASTER_SHOT`, `MEDIUM_SHOT`, `MCU`, `CLOSE_UP`, and `OVER_SHOULDER`
- Semantic mapping from scene emotion to lighting, motion, and escalation
- Narrative beat planning across `ESTABLISH`, `RELATION`, `BUILD`, `ESCALATION`, `REACTION`, and `RESOLUTION`
- English and Chinese scene parsing with dialogue extraction
- Prompt assembly for Stable Diffusion-style image/video workflows
- Delivery packaging to manifest JSON, shot list CSV, render queue JSON, and editorial EDL
- Acceptance and production-readiness reports

## Architecture

```text
[Parser Filter]  ->  [Director Filter]  ->  [Geometry Filter]  ->  [Formatter Filter]  ->  [Delivery Layer]
script_analyzer       director_logic         spatial_solver          prompt_builder         delivery
```

## Setup With uv

```bash
python -m uv venv .venv
python -m uv sync --extra dev
```

LLM-backed parsing can resolve credentials from:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- an explicit `config_path`
- a sibling workspace config file such as `../config/conf`

The loader normalizes common gateway URLs to an OpenAI-compatible `/v1` endpoint automatically.

## Quick Start

```python
from autocineflow.pipeline import CineFlowPipeline

pipeline = CineFlowPipeline()

ctx = pipeline.run(
    description=(
        "A man in black coat faces a woman in red dress across a tavern table. "
        "He suddenly slams the glass down and erupts in anger."
    ),
    num_shots=5,
    use_llm=False,
)

print(pipeline.to_json(ctx, indent=2))
print(pipeline.controlnet_coords(ctx))
print(pipeline.acceptance_report(ctx))
print(pipeline.production_readiness_report(ctx))
```

## Delivery Package

```python
package = pipeline.build_storyboard_package(ctx, project_name="Feature Previs")

print(pipeline.storyboard_package_json(package, indent=2))
print(pipeline.shotlist_csv(package))
print(pipeline.render_queue_json(package, indent=2))
print(pipeline.edl_text(package))

pipeline.write_delivery_package(package, "out/feature_previs_scene_01")
```

Each packaged shot includes:

- stable `shot_id` such as `SCENE_01_SH001`
- `timeline_in` / `timeline_out` SMPTE-style timecodes
- editorial duration and frame count
- render prompt and negative prompt
- ControlNet-compatible points
- staging and nose-room notes

## Validation

```python
report = pipeline.acceptance_report(ctx)
assert report["logic_axis_consistency"]
assert report["logic_gaze_direction"]
assert report["data_required_fields"]
assert report["data_visual_anchor_consistency"]
assert report["interface_prompt_quality"]
assert report["interface_controlnet_coords"]

production = pipeline.production_readiness_report(ctx)
assert all(production.values())
```

## Run Tests

```bash
python -m uv run pytest
```

## Run Online Production Evaluation

```bash
python -m uv run python -m autocineflow.production_eval --config-path D:\Codex\workspace\config\conf --fail-on-readiness
```

## Package A Scene From The CLI

```bash
python -m uv run python -m autocineflow.delivery ^
  --description "A detective in a rain-soaked trench coat faces a wounded informant in a neon alley." ^
  --scene-id SCENE_07 ^
  --project-name "Feature Previs" ^
  --output-dir out\scene_07 ^
  --config-path D:\Codex\workspace\config\conf
```

This writes:

- `storyboard_package.json`
- `shotlist.csv`
- `render_queue.json`
- `timeline.edl`
