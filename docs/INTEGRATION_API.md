# Integration API

## Purpose

This document defines the current integration surface for other modules or services that need to call into Auto-CineFlow.

The primary Python integration boundary is:

- `autocineflow.CineFlowPipeline`

Other code should prefer this class over reaching into lower-level modules directly unless there is a clear reason to work at a lower layer.

## Stable Entry Points

### Primary import surface

```python
from autocineflow import (
    CineFlowPipeline,
    SceneContext,
    StoryboardPackage,
    ProjectPackage,
    SubmissionProvider,
    SubmissionBackend,
    SubmissionTarget,
)
```

### Core enums

#### `SubmissionProvider`

Allowed values:

- `generic`
- `automatic1111`
- `comfyui`
- `minimax_image`
- `minimax_video`
- `runninghub_faceid`
- `runninghub_video_auto`
- `runninghub_video_quality`
- `runninghub_video_fast`
- `volcengine_seedream`

#### `SubmissionBackend`

Allowed values:

- `filesystem`
- `webhook`
- `dry_run`
- `minimax_api`
- `volcengine_ark`
- `runninghub_api`

## `CineFlowPipeline`

### Constructor

```python
CineFlowPipeline(
    api_key: str | None = None,
    model: str = "gpt-5",
    base_url: str | None = None,
    config_path: str | None = None,
)
```

Behavior:

- resolves LLM settings from explicit args, env, or `config/conf`
- stores provider config path for later execution and QA steps

## Scene Planning API

### `run`

```python
run(
    description: str,
    num_shots: int = 5,
    scene_id: str = "SCENE_01",
    use_llm: bool = True,
    emotion_override: str | None = None,
) -> SceneContext
```

Use this when another module needs the raw planned scene state without delivery/export.

### `run_with_quality_gate`

```python
run_with_quality_gate(
    description: str,
    num_shots: int = 5,
    scene_id: str = "SCENE_01",
    use_llm: bool = True,
    emotion_override: str | None = None,
    min_score: float = 0.85,
    max_attempts: int = 3,
) -> tuple[SceneContext, dict[str, object]]
```

Use this when upstream callers want automatic retry/fallback behavior before a scene is accepted.

### Validation helpers

These operate on `SceneContext`:

- `acceptance_report(context) -> dict[str, bool]`
- `production_readiness_report(context) -> dict[str, bool]`
- `quality_report(context, min_score=0.85) -> dict[str, object]`
- `controlnet_coords(context, shot_index=None) -> list[dict]`

## Delivery API

### `build_storyboard_package`

```python
build_storyboard_package(
    context: SceneContext,
    project_name: str = "Auto-CineFlow Project",
    render_preset: RenderPreset | None = None,
    generated_at: str | None = None,
    reference_roots: list[str | Path] | None = None,
    character_reference_top_k: int = 3,
    scene_reference_top_k: int = 4,
    reference_library: ReferenceLibrary | None = None,
) -> StoryboardPackage
```

This is the main scene-level delivery interface.

### Scene package serializers

- `storyboard_package_json(package, indent=2) -> str`
- `shotlist_csv(package) -> str`
- `render_queue_json(package, indent=2) -> str`
- `character_bible_json(package, indent=2) -> str`
- `video_plan_json(package, indent=2) -> str`
- `edl_text(package) -> str`
- `storyboard_review_markdown(package) -> str`

### Provider bundle serializers

- `automatic1111_bundle_json(package, indent=2) -> str`
- `comfyui_bundle_json(package, indent=2) -> str`
- `runninghub_faceid_bundle_json(package, indent=2) -> str`
- `runninghub_video_bundle_json(package, mode="auto", indent=2) -> str`
- `runninghub_workflow_suite_json(indent=2) -> str`
- `volcengine_seedream_bundle_json(package, indent=2) -> str`

### Disk writer

```python
write_delivery_package(
    package: StoryboardPackage,
    output_dir: str | Path,
) -> dict[str, Path]
```

This writes all scene delivery assets and returns a file map.

## Consistency API

### `default_reference_roots`

```python
default_reference_roots() -> list[Path]
```

### `build_consistency_package`

```python
build_consistency_package(
    context: SceneContext,
    package: StoryboardPackage,
    reference_roots: list[str | Path] | None = None,
    reference_library: ReferenceLibrary | None = None,
    character_top_k: int = 3,
    scene_top_k: int = 4,
    allow_generated_scene_refs: bool = False,
) -> ConsistencyPackage
```

### `attach_consistency_package`

```python
attach_consistency_package(
    package: StoryboardPackage,
    consistency: ConsistencyPackage,
) -> StoryboardPackage
```

## Submission API

### Build jobs

```python
build_submission_jobs_from_package(
    package: StoryboardPackage,
    provider: SubmissionProvider = SubmissionProvider.GENERIC,
) -> list[SubmissionJob]
```

```python
build_submission_jobs_from_execution_plan(
    plan: ProjectExecutionPlan,
    provider: SubmissionProvider = SubmissionProvider.GENERIC,
) -> list[SubmissionJob]
```

### Submit jobs

```python
submit_jobs(
    jobs: list[SubmissionJob],
    target: SubmissionTarget,
    source_type: str,
    source_id: str,
) -> SubmissionBatch
```

`SubmissionTarget`:

```python
SubmissionTarget(
    backend: SubmissionBackend,
    spool_dir: str = "",
    webhook_url: str = "",
    headers: dict[str, str] = {},
    timeout_seconds: float = 30.0,
    config_path: str = "",
)
```

### Submission batch helpers

- `submission_batch_json(batch, indent=2) -> str`
- `submission_batch_markdown(batch) -> str`
- `merge_submission_batches(*batches) -> SubmissionBatch`
- `write_submission_batch(batch, output_dir) -> dict[str, Path]`

## Artifact and QA API

### Download artifacts

```python
download_submission_artifacts(
    batch: SubmissionBatch,
    output_dir: str | Path,
    config_path: str | None = None,
    timeout_seconds: float = 900.0,
    poll_interval_seconds: float = 10.0,
    skip_existing: bool = True,
) -> ArtifactDownloadBatch
```

### Manifest sync

```python
update_render_manifest_from_downloads(
    manifest_path: str | Path,
    downloads: ArtifactDownloadBatch,
) -> list[RenderExpectation]
```

### Render QA

- `render_manifest_template_json(package, indent=2) -> str`
- `render_qa_report(package, manifest, min_score=0.9) -> RenderQAReport`
- `render_qa_report_json(report, indent=2) -> str`
- `render_qa_review_markdown(report) -> str`
- `write_render_qa_report(report, output_dir) -> dict[str, Path]`

### Keyframe QA and gate

- `keyframe_qa_report(downloads, min_score=0.75) -> KeyframeQAReport`
- `keyframe_qa_report_json(report, indent=2) -> str`
- `keyframe_qa_markdown(report) -> str`
- `write_keyframe_qa_report(report, output_dir) -> dict[str, Path]`
- `build_keyframe_gate_report(keyframe_report, local_visual_report=None) -> KeyframeGateReport`
- `write_keyframe_gate_report(report, output_dir) -> dict[str, Path]`
- `select_best_keyframe_downloads(*batches) -> ArtifactDownloadBatch`
- `build_keyframe_repair_jobs(keyframe_jobs, report) -> list[SubmissionJob]`

### Local visual review

```python
review_keyframes_with_local_vlm(
    keyframe_report: KeyframeQAReport,
    config_path: str | None = None,
    timeout_seconds: float = 600.0,
) -> LocalVisualReviewReport
```

```python
write_local_visual_review_report(
    report: LocalVisualReviewReport,
    output_dir: str | Path,
) -> dict[str, Path]
```

### Video enhancement

- `enhance_videos(downloads, output_dir, preset="production_hd", ffmpeg_bin="ffmpeg", ffprobe_bin="ffprobe") -> VideoEnhanceReport`
- `video_enhance_report_json(report, indent=2) -> str`
- `video_enhance_report_markdown(report) -> str`
- `write_video_enhance_report(report, output_dir) -> dict[str, Path]`

### Sequence assembly and QA

- `recommended_video_shot_count(target_duration_seconds, clip_duration_seconds=4.0, min_shots=8, max_shots=24) -> int`
- `build_sequence_assembly_plan(package, artifacts_dir="artifacts") -> SequenceAssemblyPlan`
- `sequence_assembly_json(plan, indent=2) -> str`
- `write_sequence_assembly_plan(plan, output_dir) -> dict[str, Path]`
- `assemble_sequence_with_ffmpeg(plan, assembly_dir, output_path=None, ffmpeg_bin="ffmpeg") -> SequenceAssemblyResult`
- `build_sequence_qa_report(plan, assembly_dir, final_sequence_filename=None) -> SequenceQAReport`
- `build_sequence_repair_plan(report) -> SequenceRepairPlan`
- `write_sequence_qc_outputs(report, repair_plan, output_dir) -> dict[str, Path]`

## Project API

### Build project package

```python
build_project_package(
    scene_inputs: list[ProjectSceneInput],
    project_name: str = "Auto-CineFlow Project",
    use_llm: bool = True,
    min_score: float = 0.85,
    max_attempts: int = 3,
    reference_roots: list[str | Path] | None = None,
    character_reference_top_k: int = 3,
    scene_reference_top_k: int = 4,
) -> ProjectPackage
```

### Project serializers and writers

- `project_manifest_json(package, indent=2) -> str`
- `project_shotlist_csv(package) -> str`
- `project_review_markdown(package) -> str`
- `write_project_package(package, output_dir) -> dict[str, Path]`

### Project diff/execution

- `build_project_change_plan(previous, current) -> ProjectChangePlan`
- `project_change_plan_json(plan, indent=2) -> str`
- `write_project_change_plan(plan, output_dir) -> dict[str, Path]`
- `build_project_execution_plan(previous, current, previous_scenes_dir) -> ProjectExecutionPlan`
- `project_execution_plan_json(plan, indent=2) -> str`
- `write_project_execution_plan(plan, output_dir) -> dict[str, Path]`

### Project QA and dashboard

- `build_project_render_qa_report(project, scenes_dir, min_score=0.9) -> ProjectRenderQAReport`
- `project_render_qa_report_json(report, indent=2) -> str`
- `write_project_render_qa_report(report, output_dir) -> dict[str, Path]`
- `build_project_dashboard(project, render_qa=None, execution_plan=None, asset_library=None, scenes_dir=None) -> ProjectDashboard`
- `project_dashboard_json(dashboard, indent=2) -> str`
- `write_project_dashboard(dashboard, output_dir) -> dict[str, Path]`

## Asset and Oversight API

- `build_asset_library(root_dir) -> AssetLibrary`
- `asset_library_json(library, indent=2) -> str`
- `asset_library_markdown(library) -> str`
- `latest_scene_versions(library) -> list[SceneAssetVersion]`
- `latest_project_versions(library) -> list[ProjectAssetVersion]`
- `write_asset_library(library, output_dir) -> dict[str, Path]`

## Recovery API

- `build_recovery_plan(batch) -> RecoveryPlan`
- `recovery_plan_json(plan, indent=2) -> str`
- `recovery_plan_markdown(plan) -> str`
- `write_recovery_plan(plan, output_dir) -> dict[str, Path]`

## CLI Contracts

The following CLIs are intended for external orchestration and operational use:

- `autocineflow.delivery`
- `autocineflow.project_batch`
- `autocineflow.submission`
- `autocineflow.scene_runner`
- `autocineflow.scene_resume`
- `autocineflow.scene_repair`
- `autocineflow.scene_keyframe_repair`
- `autocineflow.local_visual_review`
- `autocineflow.project_dashboard`
- `autocineflow.asset_library`
- `autocineflow.provider_probe`

## Integration Guidance

### Preferred usage patterns

- Use `run` or `run_with_quality_gate` for planning only.
- Use `build_storyboard_package` + `write_delivery_package` for scene-level production handoff.
- Use `build_submission_jobs_from_package` + `submit_jobs` for provider execution.
- Use `download_submission_artifacts` + QA APIs for downstream validation.
- Use project-level APIs for multi-scene orchestration instead of hand-rolling loops.

### Avoid

- writing directly into internal JSON artifacts without rerunning the relevant builder
- bypassing `CineFlowPipeline` to call deep provider modules unless you are extending provider behavior itself
- treating raw generated keyframes as video inputs without a gate decision

## Compatibility Note

The documented API reflects the current branch state. Lower-level internal helper functions are not considered stable integration contracts unless explicitly listed in this document.
