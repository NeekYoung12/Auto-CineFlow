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
- Provider bundles for common generation stacks such as Automatic1111-style and ComfyUI-oriented payloads
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
- deterministic `render_seed` for reproducible generation
- `reference_shot_id` and `continuity_group` for continuity chaining
- render prompt and negative prompt
- ControlNet-compatible points
- staging and nose-room notes

The package also includes a `character_bible` with per-character continuity tags and default seeds.
For human review, the exporter also writes a `storyboard_review.md` document.
Provider-oriented files are also written for downstream generators.
It also writes a `render_manifest_template.json` file for downstream render tracking and QA.

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
- `character_bible.json`
- `timeline.edl`
- `storyboard_review.md`
- `providers/automatic1111_txt2img.json`
- `providers/comfyui_prompt_bundle.json`
- `render_manifest_template.json`

## Render QA

After a renderer fills the manifest with actual output metadata, run:

```bash
python -m uv run python -m autocineflow.render_qa ^
  --package-file out\scene_10\storyboard_package.json ^
  --manifest-file out\scene_10\render_manifest_template.json ^
  --output-dir out\scene_10\qa
```

This writes:

- `render_qa_report.json`
- `render_qa_review.md`

## Project Render QA

To aggregate render QA across all scene folders in a project:

```bash
python -m uv run python -m autocineflow.project_render_qa ^
  --project-file out\feature_previs\project_manifest.json ^
  --scenes-dir out\feature_previs\scenes ^
  --output-dir out\feature_previs\project_qa
```

This writes:

- `project_render_qa_report.json`
- `project_render_qa_review.md`

## Project Batch Packaging

```bash
python -m uv run python -m autocineflow.project_batch ^
  --input-file scenes.json ^
  --project-name "Feature Previs" ^
  --output-dir out\feature_previs ^
  --config-path D:\Codex\workspace\config\conf
```

The batch input is a JSON array or an object with a `scenes` array:

```json
[
  {
    "scene_id": "SCENE_A",
    "description": "A man in black coat faces a woman in red dress across a tavern table.",
    "num_shots": 5,
    "emotion_override": "tense"
  }
]
```

The batch exporter writes:

- `project_manifest.json`
- `project_shotlist.csv`
- `project_review.md`
- `scenes/<scene-id>/...` with per-scene delivery assets

## Incremental Rerender Planning

When you have a previous and current project manifest, you can generate a rerender diff:

```bash
python -m uv run python -m autocineflow.change_planner ^
  --previous-project out\feature_previs_v1\project_manifest.json ^
  --current-project out\feature_previs_v2\project_manifest.json ^
  --output-dir out\feature_previs_diff
```

This writes:

- `project_change_plan.json`
- `project_change_review.md`
- `rerender_queue.json`

## Reuse-Aware Execution Planning

If you also have previous scene export folders with completed render manifests, you can build a reuse-aware execution plan:

```bash
python -m uv run python -m autocineflow.execution_planner ^
  --previous-project out\feature_previs_v1\project_manifest.json ^
  --current-project out\feature_previs_v2\project_manifest.json ^
  --previous-scenes-dir out\feature_previs_v1\scenes ^
  --output-dir out\feature_previs_execution
```

This writes:

- `project_execution_plan.json`
- `project_execution_review.md`
- `reuse_manifest.json`
- `rerender_queue.json`
- `ordered_rerender_queue.json`

## Project Dashboard

To generate one top-level dashboard from project metadata, project render QA, and execution planning:

```bash
python -m uv run python -m autocineflow.project_dashboard ^
  --project-file out\feature_previs\project_manifest.json ^
  --render-qa-file out\feature_previs\project_qa\project_render_qa_report.json ^
  --execution-plan-file out\feature_previs_execution\project_execution_plan.json ^
  --output-dir out\feature_previs_dashboard
```

This writes:

- `project_dashboard.json`
- `project_dashboard.md`

## Asset Library

To index all generated scene and project outputs under an `out/` root:

```bash
python -m uv run python -m autocineflow.asset_library ^
  --root-dir out ^
  --output-dir out\asset_library
```

This writes:

- `asset_library.json`
- `asset_library.md`

## Task Submission

To turn a scene package into submit-ready render tasks and queue them to a local spool:

```bash
python -m uv run python -m autocineflow.submission ^
  --package-file out\scene_10\storyboard_package.json ^
  --provider automatic1111 ^
  --backend filesystem ^
  --spool-dir out\submission_spool ^
  --output-dir out\submission_records
```

For project execution plans you can submit only the ordered rerender queue:

```bash
python -m uv run python -m autocineflow.submission ^
  --execution-plan-file out\feature_previs_execution\project_execution_plan.json ^
  --backend dry_run ^
  --output-dir out\submission_preview
```

This writes:

- `submission_batch.json`
- `submission_batch.md`

You can also submit directly to MiniMax image generation if your config file contains a valid media API key:

```bash
python -m uv run python -m autocineflow.submission ^
  --package-file out\scene_10\storyboard_package.json ^
  --provider minimax_image ^
  --backend minimax_api ^
  --config-path D:\Codex\workspace\config\conf ^
  --output-dir out\submission_records ^
  --spool-dir ""
```

MiniMax text-to-video is also supported:

```bash
python -m uv run python -m autocineflow.submission ^
  --package-file out\scene_10\storyboard_package.json ^
  --provider minimax_video ^
  --backend minimax_api ^
  --config-path D:\Codex\workspace\config\conf ^
  --output-dir out\submission_records
```

## End-to-End Scene Runner

To run one scene through generation, packaging, MiniMax submission, artifact download, and render QA:

```bash
python -m uv run python -m autocineflow.scene_runner ^
  --description "A detective faces a wounded informant in a neon alley at night." ^
  --scene-id SCENE_20 ^
  --output-dir out\scene_20_run ^
  --config-path D:\Codex\workspace\config\conf ^
  --provider minimax_image ^
  --backend minimax_api ^
  --target-duration-seconds 40 ^
  --clip-duration-seconds 4 ^
  --job-limit 1
```

For MiniMax text-to-video, swap `--provider minimax_video`.

If a long scene run only downloads part of its clips on the first pass, you can resume it without resubmitting jobs:

```bash
python -m uv run python -m autocineflow.scene_resume ^
  --run-dir out\scene_40s_full_v2 ^
  --config-path D:\Codex\workspace\config\conf
```

To rerender only the failed clips from `sequence_repair_plan.json`:

```bash
python -m uv run python -m autocineflow.scene_repair ^
  --run-dir out\scene_40s_full_v2 ^
  --config-path D:\Codex\workspace\config\conf
```

## Submission Monitoring

For filesystem-backed queues you can monitor batch progress:

```bash
python -m uv run python -m autocineflow.submission_monitor ^
  --batch-file out\submission_records\submission_batch.json ^
  --spool-dir out\submission_spool ^
  --output-dir out\submission_monitor
```

This writes:

- `submission_monitor_report.json`
- `submission_monitor_report.md`

## Recovery Planning

To turn provider failures into actions such as pause, retry, or manual fix:

```bash
python -m uv run python -m autocineflow.recovery_policy ^
  --batch-file out\scene_40s_full_v2\repair_submission\submission_batch.json ^
  --output-dir out\scene_40s_full_v2\recovery
```

This writes:

- `recovery_plan.json`
- `recovery_plan.md`
