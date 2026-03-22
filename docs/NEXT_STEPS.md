# Next Steps

## Purpose

This document lists the current highest-value follow-up work for Auto-CineFlow. It is meant to guide continued development after the current milestone.

## Current State Summary

The system already supports:

- scene analysis and storyboard planning
- consistency retrieval and character/scene reference packaging
- real RunningHub image and video generation
- keyframe QA
- optional local visual review
- unified keyframe gating
- AI and local post-enhancement
- sequence QA
- project dashboard and asset library visibility

The next work should improve automation, quality recovery, and project-scale operability.

## Priority 1: Make keyframe repair part of the normal resume flow

### Goal

When a scene is blocked by keyframe quality, the system should not require manual command selection. Resume and repair flows should detect the gate state and automatically choose the correct recovery lane.

### Tasks

- teach `scene_resume` to read `keyframe_gate_report.json`
- if the scene is blocked before video generation, automatically route to keyframe repair instead of video resume
- if keyframes pass after repair, automatically continue into the configured video provider

### Expected result

One command can recover either:

- keyframe-blocked scenes
- partially rendered video scenes
- fully rendered but not yet assembled scenes

## Priority 2: Improve text artifact repair with targeted masks

### Goal

Current keyframe repair is prompt-driven. The next step is to make text and signage repair spatially targeted.

### Tasks

- extend `keyframe_qa` to emit approximate bounding boxes or hot regions for text-like artifacts
- build masks from those hot regions
- feed masks into `rh_shot_repair_inpaint_v1`
- compare repaired outputs against previous keyframes and keep the best one

### Expected result

More reliable removal of:

- fake signage
- malformed Chinese/English letters
- subtitle-like artifacts on props and walls

## Priority 3: Project-level automated recovery orchestration

### Goal

Move from scene-level recovery tools to project-level recovery plans that can decide:

- which scenes are storyboard-ready
- which scenes are keyframe-blocked
- which scenes can continue directly to video
- which scenes need post-enhancement reruns

### Tasks

- add project-level keyframe gate aggregation
- add project-level keyframe repair plan writer
- extend execution planning so it understands gate-blocked scenes
- teach dashboard and asset library to expose project-wide blocked reasons clearly

## Priority 4: Stronger local visual review policy

### Goal

The local VLM review is currently optional and low-priority. It should become more operationally useful without turning into a hard dependency.

### Tasks

- add explicit score normalization and confidence handling
- persist skip reasons more clearly in dashboard/asset library
- support CPU fallback for low-priority offline review runs
- allow sampling only selected keyframes instead of reviewing every frame

## Priority 5: Better documentation and integration consistency

### Goal

Keep the external contract stable and handoff-friendly.

### Tasks

- keep `INTEGRATION_API.md` aligned with `CineFlowPipeline`
- keep `CURRENT_ARCHITECTURE.md` aligned with the real module map
- keep `PRODUCTION_RUNBOOK.md` aligned with the latest operational commands
- document every new blocking state and recovery lane when introduced

## Suggested Development Order

1. unify `scene_resume` and `scene_keyframe_repair`
2. add targeted text masks to keyframe repair
3. aggregate keyframe gate state at project level
4. improve local visual review ergonomics and reporting
5. continue tightening docs as the public integration surface evolves
