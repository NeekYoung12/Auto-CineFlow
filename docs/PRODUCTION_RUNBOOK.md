# Production Runbook

## Purpose

This runbook lists the practical commands and recovery actions most useful during day-to-day previs production.

## Primary Commands

### Build a scene package

```bash
python -m uv run python -m autocineflow.delivery ^
  --description "A detective in a rain-soaked trench coat faces a wounded informant in a neon alley." ^
  --scene-id SCENE_07 ^
  --project-name "Feature Previs" ^
  --output-dir out\scene_07 ^
  --config-path D:\Codex\workspace\config\conf
```

### Run a full RunningHub scene

```bash
python -m uv run python -m autocineflow.scene_runner ^
  --description "A detective faces a wounded informant in a neon alley at night." ^
  --scene-id SCENE_RUN ^
  --output-dir out\scene_run ^
  --config-path D:\Codex\workspace\config\conf ^
  --provider runninghub_video_auto ^
  --backend runninghub_api
```

### Resume an existing scene

```bash
python -m uv run python -m autocineflow.scene_resume ^
  --run-dir out\scene_run ^
  --config-path D:\Codex\workspace\config\conf
```

### Repair only failed video clips

```bash
python -m uv run python -m autocineflow.scene_repair ^
  --run-dir out\scene_run ^
  --config-path D:\Codex\workspace\config\conf
```

### Repair only keyframe stages

```bash
python -m uv run python -m autocineflow.scene_keyframe_repair ^
  --run-dir out\scene_run ^
  --config-path D:\Codex\workspace\config\conf ^
  --enable-local-vlm-review ^
  --continue-video
```

### Review existing keyframes with local Qwen-VL

```bash
python -m uv run python -m autocineflow.local_visual_review ^
  --artifacts-dir out\scene_run\rebuild_keyframes\artifacts ^
  --source-id SCENE_RUN ^
  --config-path D:\Codex\workspace\config\conf ^
  --output-dir out\scene_run\local_visual_review
```

## Output Directories

### Key scene outputs

- `delivery/`
- `submission/`
- `downloads/`
- `qa/`
- `sequence_qc/`
- `recovery/`

### RunningHub keyframe outputs

- `bootstrap_keyframes/`
- `rebuild_keyframes/`
- `repair_keyframes/`
- `keyframe_qc/`

### Video enhancement outputs

- `video_post_enhance/`
- `video_post_enhance_retry/`
- `enhance/`
- `enhanced_artifacts/`

## Recovery Guidance

### When keyframe gate blocks video

Use:

```bash
python -m uv run python -m autocineflow.scene_keyframe_repair ^
  --run-dir out\scene_run ^
  --config-path D:\Codex\workspace\config\conf ^
  --enable-local-vlm-review ^
  --continue-video
```

### When video generation succeeded but looks weak

Check:

- `video_post_enhance/`
- `enhance/`
- `keyframe_qc/`
- `delivery/providers/runninghub_video_auto_bundle.json`

### When RunningHub post-enhance fails

The current code automatically retries that stage with more conservative memory settings. If it still fails, the local FFmpeg enhancement remains the final fallback.

## Key Config Sections

### RunningHub

- `API_KEY`
- `RUNNINGHUB_BASE_URL`
- `RUNNINGHUB_API_FORMAT_DIR`
- `RUNNINGHUB_WORKFLOW_*`

### Volcengine

- `ARK_API_KEY`
- optional `VOLCENGINE_ARK_BASE_URL`

### Local Visual Review

- `PYTHON_PATH`
- `MODEL_PATH`
- `DEVICE_PREFERENCE`
- `MIN_FREE_VRAM_GB`

## Operational Rule

Do not spend video budget on a scene that has not passed the keyframe gate unless you intentionally override the gate for debugging. The current production path assumes keyframe quality is upstream of video quality.
