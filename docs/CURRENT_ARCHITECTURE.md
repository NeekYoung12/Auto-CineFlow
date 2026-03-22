# Current Architecture

## Scope

This document describes the current implementation state of Auto-CineFlow as of the latest working branch. It is intended for future development handoff.

## Top-Level Flow

```text
scene text
  -> analysis
  -> shot planning
  -> prompt assembly
  -> consistency retrieval
  -> package + payload export
  -> provider submission
  -> keyframe QA / review / gate
  -> video generation
  -> post-enhancement
  -> sequence QA
  -> project oversight
```

## Module Map

### Planning

- `script_analyzer.py`
- `director_logic.py`
- `spatial_solver.py`
- `prompt_builder.py`

### Consistency and references

- `reference_rag.py`
- `consistency_models.py`
- `consistency.py`

### Delivery and exports

- `delivery.py`
- `provider_payloads.py`
- `project_delivery.py`

### Provider execution

- `submission.py`
- `runninghub_backend.py`
- `result_ingest.py`
- `provider_probe.py`

### Quality control

- `keyframe_qa.py`
- `local_visual_review.py`
- `keyframe_gate.py`
- `render_qa.py`
- `sequence_qa.py`
- `video_enhance.py`

### Recovery and repair

- `recovery_policy.py`
- `scene_resume.py`
- `scene_repair.py`
- `scene_keyframe_repair.py`

### Oversight

- `asset_library.py`
- `project_render_qa.py`
- `project_dashboard.py`
- `change_planner.py`
- `execution_planner.py`

## RunningHub Production Path

### Image/keyframe lane

1. Build `runninghub_faceid` jobs
2. Generate bootstrap keyframes
3. Run keyframe QA
4. Rebuild keyframes with stronger prompts
5. Optionally run repair-focused keyframe reruns
6. Optionally run local visual review
7. Build one `keyframe_gate_report`
8. Block or allow video generation

### Video lane

1. Pick workflow bundle:
   - `runninghub_video_auto`
   - `runninghub_video_quality`
   - `runninghub_video_fast`
2. Inject the selected best keyframe into the request contract
3. Submit RunningHub video jobs
4. Download MP4 outputs
5. Optionally run RunningHub AI post-enhance
6. Optionally run local FFmpeg enhancement
7. Build sequence QA / repair

## Gate Hierarchy

The system now distinguishes these states:

- `storyboard_blocked`
- `keyframe_blocked:keyframe_qa`
- `keyframe_blocked:local_visual_review`
- `visual_review_blocked`
- `render_blocked`
- `rerender_pending`
- `ready`

Project-level tooling should prefer gate reports over ad hoc directory checks.

## Important Runtime Assumptions

### RunningHub

- `workflowId` values are stored in `config/conf`
- API export JSON files are stored in `RUNNINGHUB_API_FORMAT_DIR`
- local uploads are converted into RunningHub `fileName` values before submission

### Local visual review

- runs in an external Python environment
- can use the ComfyUI Python environment
- should skip rather than block under resource pressure

## Current Weak Spots

- local visual review is not yet mandatory
- text artifact masks are still heuristic
- some provider post-enhance workflows remain sensitive to remote GPU memory pressure
- project-level automatic replay of keyframe repair is still being expanded

## Recommended Next Development Order

1. strengthen keyframe repair masks for text artifacts
2. let resume/repair commands consume the newest gate report automatically
3. add project-level keyframe repair orchestration
4. make provider probe and dashboards show local visual review readiness and recent skips
