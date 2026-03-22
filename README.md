# Auto-CineFlow

Auto-CineFlow is a previs-oriented cinematic production pipeline for 1-2 character scenes. It starts from a short scene description and produces:

- structured storyboard shots
- character and scene consistency packages
- provider-specific payload bundles
- real image/video generation runs
- QA, repair, and delivery artifacts

The current codebase is no longer only a storyboard generator. It is a working previs production system with real integrations for:

- MiniMax image generation
- MiniMax video generation
- Volcengine Seedream image generation
- RunningHub keyframe generation
- RunningHub video generation
- RunningHub AI post-enhancement
- local FFmpeg video enhancement

## Docs Index

- [DESIGN.md](./DESIGN.md): current product and technical design baseline
- [CURRENT_ARCHITECTURE.md](./docs/CURRENT_ARCHITECTURE.md): module map and pipeline graph
- [INTEGRATION_API.md](./docs/INTEGRATION_API.md): strict Python and CLI integration contracts for other modules
- [NEXT_STEPS.md](./docs/NEXT_STEPS.md): prioritized continuation roadmap for future development
- [PRODUCTION_RUNBOOK.md](./docs/PRODUCTION_RUNBOOK.md): operational commands, config, and recovery usage

## Primary Integration Interface

If another module needs to call into Auto-CineFlow, the primary supported interface is:

- `autocineflow.CineFlowPipeline`

The strict contract for what the pipeline currently supports is documented in:

- [INTEGRATION_API.md](./docs/INTEGRATION_API.md)

## Current Pipeline

At a high level, one scene can flow through these stages:

1. `script_analyzer` parses the scene into characters, emotion, dialogue, location, and tags.
2. `director_logic` and `spatial_solver` generate shot sequencing, lensing, geometry, and blocking.
3. `prompt_builder` assembles render prompts and negative prompts.
4. `consistency` and `reference_rag` retrieve character and scene references from local libraries.
5. `delivery` packages the scene into production assets.
6. `submission` sends jobs to the selected backend.
7. `result_ingest` downloads outputs and syncs render manifests.
8. `keyframe_qa`, optional `local_visual_review`, and `keyframe_gate` decide whether keyframes can continue into video.
9. `video_enhance` and optional RunningHub post-enhance improve finished clips.
10. `sequence_assembly` and `sequence_qa` assemble and validate longer previs sequences.

## Core Modules

- `script_analyzer.py`
- `director_logic.py`
- `spatial_solver.py`
- `prompt_builder.py`
- `consistency.py`
- `reference_rag.py`
- `delivery.py`
- `submission.py`
- `runninghub_backend.py`
- `result_ingest.py`
- `keyframe_qa.py`
- `keyframe_gate.py`
- `local_visual_review.py`
- `video_enhance.py`
- `sequence_assembly.py`
- `sequence_qa.py`
- `asset_library.py`
- `project_dashboard.py`

## Provider Support

### Real backends

- `minimax_image` + `minimax_api`
- `minimax_video` + `minimax_api`
- `volcengine_seedream` + `volcengine_ark`
- `runninghub_faceid` + `runninghub_api`
- `runninghub_video_auto` + `runninghub_api`
- `runninghub_video_quality` + `runninghub_api`
- `runninghub_video_fast` + `runninghub_api`

### Provider payload exports

Every delivery package also exports offline payload bundles for:

- Automatic1111
- ComfyUI
- RunningHub keyframe workflows
- RunningHub video workflows
- Volcengine Seedream

## Setup

```bash
python -m uv venv .venv
python -m uv sync --extra dev
```

Run tests:

```bash
python -m uv run pytest
```

## Config

The pipeline resolves credentials from `config/conf` or explicit `--config-path`.

Current supported config sections:

- `MiniMax:`
- `Image or Video Generation:`
- `Kimi-Code:`
- `OpenRouter:`
- `OPENAI RELATED:`
- `RunningHUB:`
- `volcengine:`
- `Local Visual Review:`

### RunningHub requirements

For full RunningHub automation, both of these are required:

1. workflow IDs in `RunningHUB:`
2. exported API-format workflow JSON files in `RUNNINGHUB_API_FORMAT_DIR`

Expected standard filenames:

- `rh_char_identity_forge_v1.json`
- `rh_char_sheet_multiview_v1.json`
- `rh_scene_set_forge_v1.json`
- `rh_shot_keyframe_faceid_v1.json`
- `rh_shot_relight_match_v1.json`
- `rh_shot_repair_inpaint_v1.json`
- `rh_shot_i2v_wan22_full_v1.json`
- `rh_shot_i2v_wan21_hq_v1.json`
- `rh_shot_i2v_framepack_fast_v1.json`
- `rh_video_post_enhance_v1.json`

### Local visual review

Default local visual review runtime:

- Python:
  `C:\Users\neekyoung12\Documents\ComfyUI\.venv\Scripts\python.exe`
- Model:
  `D:\ComfyUI_ws\ComfyUI_Models\models\LLM\Qwen-VL\Huihui-Qwen3-VL-4B-Instruct-abliterated`

This review stage is intentionally optional and low-priority:

- it is disabled unless explicitly requested
- it runs in an external Python process
- it skips rather than blocking when GPU memory is too tight

## Common Commands

### Scene delivery only

```bash
python -m uv run python -m autocineflow.delivery ^
  --description "A detective in a rain-soaked trench coat faces a wounded informant in a neon alley." ^
  --scene-id SCENE_07 ^
  --project-name "Feature Previs" ^
  --output-dir out\scene_07 ^
  --config-path D:\Codex\workspace\config\conf
```

### End-to-end scene run

```bash
python -m uv run python -m autocineflow.scene_runner ^
  --description "A detective faces a wounded informant in a neon alley at night." ^
  --scene-id SCENE_20 ^
  --output-dir out\scene_20_run ^
  --config-path D:\Codex\workspace\config\conf ^
  --provider runninghub_video_auto ^
  --backend runninghub_api
```

### Keyframe-only repair

```bash
python -m uv run python -m autocineflow.scene_keyframe_repair ^
  --run-dir out\scene_runninghub_rebuild_prod ^
  --config-path D:\Codex\workspace\config\conf ^
  --enable-local-vlm-review ^
  --continue-video
```

### Standalone keyframe visual review

```bash
python -m uv run python -m autocineflow.local_visual_review ^
  --artifacts-dir out\scene_runninghub_rebuild_prod\rebuild_keyframes\artifacts ^
  --source-id SCENE_RUNNINGHUB_REBUILD_PROD ^
  --config-path D:\Codex\workspace\config\conf ^
  --output-dir out\scene_runninghub_rebuild_prod\local_visual_review
```

### Project dashboard

```bash
python -m uv run python -m autocineflow.project_dashboard ^
  --project-file out\feature_previs\project_manifest.json ^
  --render-qa-file out\feature_previs\project_qa\project_render_qa_report.json ^
  --execution-plan-file out\feature_previs_execution\project_execution_plan.json ^
  --output-dir out\feature_previs_dashboard
```

### Provider probe

```bash
python -m uv run python -m autocineflow.provider_probe ^
  --config-path D:\Codex\workspace\config\conf ^
  --output-dir out\provider_probe
```

## Current Quality Strategy

The current production strategy for RunningHub scenes is:

- prefer external reference assets over previously generated `out/` frames
- generate bootstrap keyframes first
- rebuild keyframes with stronger prompts before video
- run heuristic keyframe QA
- optionally run local visual review with local Qwen-VL
- build a unified keyframe gate
- block video generation when the gate fails unless explicitly overridden
- run video generation
- optionally run RunningHub AI post-enhancement
- run local FFmpeg enhancement as fallback/final polish

## Current Limitations

- local visual review is optional and not yet enabled by default
- keyframe gate currently uses heuristic QA plus optional local visual review, but not yet a mandatory multimodal judge on every run
- some RunningHub post-enhancement jobs can still fail under remote GPU memory pressure
- project-level resumability is stronger for scene/video artifacts than for keyframe gate iterations

## Development Notes

When continuing development, treat these as the current priorities:

1. keep keyframe quality ahead of video spend
2. avoid feeding generated artifacts back into reference retrieval unless explicitly intended
3. prefer structured gate outputs over ad hoc status checks
4. keep provider integrations resumable and diagnosable
5. preserve a clean split between:
   - scene planning
   - keyframe generation
   - video generation
   - post-enhancement
   - QA and repair
