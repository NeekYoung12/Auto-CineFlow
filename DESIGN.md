# Auto-CineFlow Design

## Purpose

Auto-CineFlow is a previs production pipeline for short narrative scenes with one or two primary characters. The system is designed to produce industrially usable storyboard, keyframe, video, and QA assets rather than only prompt text.

## Design Goals

### 1. Cinematic correctness

- maintain a stable 180-degree axis unless a scene reset is intentional
- keep screen direction and eyelines coherent
- preserve character visual anchors through the full scene
- generate shot progressions that read like film coverage, not random image prompts

### 2. Production continuity

- assign deterministic shot IDs and render seeds
- export continuity metadata such as `reference_shot_id` and `continuity_group`
- keep character identity, wardrobe, makeup, and face cues stable through local reference retrieval and downstream workflow guidance

### 3. Real backend execution

The system must be able to move from planning to actual provider execution. Current real backends:

- MiniMax image
- MiniMax video
- Volcengine Seedream image
- RunningHub keyframe generation
- RunningHub video generation
- RunningHub AI video post-enhancement

### 4. Quality gates before scale

The pipeline should not spend video generation budget on low-quality keyframes. Keyframe quality is therefore treated as a gate, not just a report.

## Current System Architecture

### Stage A: Scene analysis

- `script_analyzer.py`
- `director_logic.py`
- `spatial_solver.py`
- `prompt_builder.py`

Inputs:

- scene description
- optional LLM configuration

Outputs:

- `SceneContext`
- beat plan
- shot blocks
- render prompts

### Stage B: Consistency and delivery

- `reference_rag.py`
- `consistency.py`
- `delivery.py`
- `provider_payloads.py`

Inputs:

- `SceneContext`
- local reference roots

Outputs:

- `StoryboardPackage`
- consistency package
- provider payload bundles
- render manifest template
- sequence assembly plan

### Stage C: Execution

- `submission.py`
- `runninghub_backend.py`
- `result_ingest.py`

Responsibilities:

- submit jobs to providers
- upload local inputs when required
- poll provider outputs
- download final assets
- sync render manifests

### Stage D: Quality control

- `keyframe_qa.py`
- `local_visual_review.py`
- `keyframe_gate.py`
- `render_qa.py`
- `sequence_qa.py`
- `video_enhance.py`

Responsibilities:

- evaluate keyframe quality before video generation
- optionally run local VLM image review
- merge QA and visual review into a single keyframe gate
- evaluate render correctness
- evaluate assembled sequence correctness
- apply local video enhancement when needed

### Stage E: Oversight and project control

- `asset_library.py`
- `project_dashboard.py`
- `project_render_qa.py`
- `change_planner.py`
- `execution_planner.py`

Responsibilities:

- aggregate multi-scene outputs
- surface blocking states
- track reuse vs rerender
- preserve historical run visibility

## Key Data Objects

### `SceneContext`

The in-memory planning object containing:

- parsed characters
- emotion
- dialogue
- location
- tags
- shot blocks

### `StoryboardPackage`

The main delivery package containing:

- shot list
- render queue
- video segments
- character bible
- consistency package
- readiness and quality reports

### `ConsistencyPackage`

The character/scene reference package containing:

- character candidates
- scene candidates
- fusion plans
- multiview prompts
- face-compatible descriptors
- shot-level reference bundles

### `KeyframeGateReport`

The unified decision object that decides whether a scene may continue into video generation.

Inputs:

- heuristic keyframe QA
- optional local visual review

Outputs:

- per-shot pass/block
- blocking reason
- recommendation

## RunningHub Design

### Keyframe path

1. `runninghub_faceid` bootstrap keyframe
2. rebuild keyframe with stronger prompts
3. optional repair-rebuild when keyframe QA finds problems
4. optional local visual review
5. unified keyframe gate

### Video path

1. choose video workflow bundle:
   - `runninghub_video_auto`
   - `runninghub_video_quality`
   - `runninghub_video_fast`
2. inject selected keyframe as the first-frame candidate
3. run video generation
4. optional RunningHub AI post-enhancement
5. local FFmpeg enhancement fallback/final polish

### RunningHub workflow naming

Expected workflow keys:

- `rh_char_identity_forge_v1`
- `rh_char_sheet_multiview_v1`
- `rh_scene_set_forge_v1`
- `rh_shot_keyframe_faceid_v1`
- `rh_shot_relight_match_v1`
- `rh_shot_repair_inpaint_v1`
- `rh_shot_i2v_wan22_full_v1`
- `rh_shot_i2v_wan21_hq_v1`
- `rh_shot_i2v_framepack_fast_v1`
- `rh_video_post_enhance_v1`

## Local Visual Review Design

The local visual review stage is deliberately optional.

Reasons:

- it depends on an external Python environment
- it may compete for GPU memory with other active processes
- it should yield rather than block when resources are tight

Current strategy:

- load Qwen-VL from a local model directory
- run the worker in an external Python process
- skip when runtime paths are missing or the worker decides GPU memory is too tight
- feed the result into `KeyframeGateReport` when available

## Current Acceptance Standard

### Scene-level minimum

- storyboard package builds successfully
- consistency package builds successfully
- provider payloads export correctly
- keyframe gate is computed before RunningHub video generation

### RunningHub scene minimum

- bootstrap keyframe can be generated
- keyframe rebuild can be generated
- keyframe gate can be written
- video generation is blocked if keyframes fail the gate

### Project-level minimum

- project dashboard shows storyboard, keyframe, and render states separately
- asset library records enough metadata for later recovery and comparison

## Immediate Next Priorities

1. push local visual review from optional signal toward stronger automated repair decisions
2. improve scene-level resumption so keyframe repair and video continuation become one continuous recovery loop
3. reduce textual artifacts further by combining better masks, local review, and targeted inpaint repair
4. make project dashboards reflect the newest gate and repair outcome automatically
