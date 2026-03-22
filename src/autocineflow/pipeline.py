"""Main pipeline: Pipe-and-Filter architecture for Auto-CineFlow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from .delivery import (
    RenderPreset,
    StoryboardPackage,
    build_storyboard_package,
    character_bible_to_json,
    edl_text,
    package_to_json,
    render_queue_to_json,
    shotlist_to_csv,
    storyboard_review_markdown,
    video_plan_to_json,
    write_storyboard_package,
)
from .change_planner import (
    ProjectChangePlan,
    build_project_change_plan,
    project_change_plan_json,
    project_change_review_markdown,
    write_project_change_plan,
)
from .execution_planner import (
    ProjectExecutionPlan,
    build_project_execution_plan,
    project_execution_plan_json,
    project_execution_review_markdown,
    write_project_execution_plan,
)
from .director_logic import build_shot, plan_scene_beats
from .models import SceneContext
from .project_delivery import (
    ProjectPackage,
    ProjectSceneInput,
    build_project_package,
    project_manifest_json,
    project_review_markdown,
    project_shotlist_csv,
    write_project_package,
)
from .project_render_qa import (
    ProjectRenderQAReport,
    build_project_render_qa_report,
    project_render_qa_report_json,
    project_render_qa_review_markdown,
    write_project_render_qa_report,
)
from .project_dashboard import (
    ProjectDashboard,
    build_project_dashboard,
    project_dashboard_json,
    project_dashboard_markdown,
    write_project_dashboard,
)
from .provider_payloads import (
    automatic1111_bundle_json,
    comfyui_bundle_json,
    write_provider_payloads,
)
from .render_qa import (
    RenderExpectation,
    RenderQAReport,
    build_render_manifest_template,
    render_manifest_template_json,
    render_qa_report,
    render_qa_report_json,
    render_qa_review_markdown,
    write_render_qa_report,
)
from .submission import (
    SubmissionBackend,
    SubmissionBatch,
    SubmissionJob,
    SubmissionProvider,
    SubmissionTarget,
    build_submission_jobs_from_execution_plan,
    build_submission_jobs_from_package,
    submission_batch_json,
    submission_batch_markdown,
    submit_jobs,
    write_submission_batch,
)
from .submission_monitor import (
    SubmissionMonitorReport,
    monitor_filesystem_submission_batch,
    submission_monitor_markdown,
    submission_monitor_report_json,
    write_submission_monitor_report,
)
from .sequence_assembly import (
    SequenceAssemblyPlan,
    build_sequence_assembly_plan,
    recommended_shot_count,
    sequence_assembly_json,
    write_sequence_assembly_plan,
)
from .result_ingest import (
    ArtifactDownloadBatch,
    artifact_download_batch_json,
    download_submission_artifacts,
    update_render_manifest_from_downloads,
    write_artifact_download_batch,
)
from .prompt_builder import attach_prompts
from .script_analyzer import analyse_script
from .spatial_solver import positions_to_controlnet

logger = logging.getLogger(__name__)


class CineFlowPipeline:
    """End-to-end pipeline that converts a scene description into a ShotBlock sequence."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5",
        base_url: str | None = None,
        config_path: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.config_path = config_path

    def run(
        self,
        description: str,
        num_shots: int = 5,
        scene_id: str = "SCENE_01",
        use_llm: bool = True,
        emotion_override: Optional[str] = None,
    ) -> SceneContext:
        """Run the full pipeline and return a SceneContext with shot_blocks populated."""

        logger.info("Parser Filter: analysing script")
        context = analyse_script(
            description,
            scene_id=scene_id,
            use_llm=use_llm,
            api_key=self.api_key,
            model=self.model,
            base_url=self.base_url,
            config_path=self.config_path,
        )

        if emotion_override:
            context = context.model_copy(update={"detected_emotion": emotion_override.lower()})

        beats = plan_scene_beats(context, num_shots)
        context = context.model_copy(update={"beats": beats})

        logger.info(
            "Detected %d character(s), emotion=%s, analysis_source=%s",
            len(context.characters),
            context.detected_emotion,
            context.analysis_source,
        )

        for shot_idx in range(num_shots):
            logger.info("Director+Geometry Filter: building shot %d", shot_idx)
            shot_block, context = build_shot(context, shot_idx)
            shot_block = attach_prompts(shot_block)
            updated_shots = list(context.shot_blocks[:-1]) + [shot_block]
            context = context.model_copy(update={"shot_blocks": updated_shots})

        logger.info("Pipeline complete. %d shots generated.", num_shots)
        return context

    def to_json(self, context: SceneContext, indent: int = 2) -> str:
        """Serialise the full SceneContext to a JSON string."""

        return context.model_dump_json(indent=indent)

    def controlnet_coords(self, context: SceneContext, shot_index: int | None = None) -> list[dict]:
        """Return ControlNet coordinate dicts for a selected shot or the final shot."""

        if not context.shot_blocks:
            return []

        if shot_index is None:
            selected_shot = context.shot_blocks[-1]
        else:
            selected_shot = context.shot_blocks[shot_index]

        return positions_to_controlnet(selected_shot.characters)

    def validate_axis_consistency(self, context: SceneContext) -> bool:
        """Verify that all shots in the context share the same axis_side."""

        sides = {shot.camera_angle.axis_side for shot in context.shot_blocks}
        return len(sides) <= 1

    def validate_visual_anchor_consistency(self, context: SceneContext) -> bool:
        """Verify that each character's visual_anchor is identical across all shots."""

        anchor_registry: dict[str, str] = {}
        for shot in context.shot_blocks:
            for char in shot.characters:
                if char.char_id in anchor_registry and anchor_registry[char.char_id] != char.visual_anchor:
                    return False
                anchor_registry.setdefault(char.char_id, char.visual_anchor)
        return True

    def validate_gaze_logic(self, context: SceneContext) -> bool:
        """Validate left-right screen direction across all shots."""

        for shot in context.shot_blocks:
            for char in shot.characters:
                if char.pos.x < 0.5 and char.facing.value == "LEFT":
                    return False
                if char.pos.x > 0.5 and char.facing.value == "RIGHT":
                    return False
        return True

    def validate_data_completeness(self, context: SceneContext) -> bool:
        """Ensure every shot contains the required protocol objects."""

        for shot in context.shot_blocks:
            if not all([shot.framing, shot.camera_angle, shot.lighting, shot.motion_instruction]):
                return False
        return True

    def validate_prompt_quality(self, context: SceneContext) -> bool:
        """Check that each prompt is populated and contains shot-defining tokens."""

        for shot in context.shot_blocks:
            prompt = shot.sd_prompt.lower()
            if not prompt:
                return False
            if str(shot.framing.focal_length_mm).lower() not in prompt:
                return False
            if not any(keyword in prompt for keyword in ("shot", "cinematic", "close-up", "wide", "shoulder")):
                return False
        return True

    def validate_controlnet_range(self, context: SceneContext) -> bool:
        """Ensure ControlNet coordinates stay inside the expected 0-1 canvas."""

        for shot in context.shot_blocks:
            for coord in shot.controlnet_points:
                if not (0.0 <= float(coord["x"]) <= 1.0 and 0.0 <= float(coord["y"]) <= 1.0):
                    return False
        return True

    def validate_anchor_specificity(self, context: SceneContext) -> bool:
        """Ensure production anchors are not generic placeholders."""

        generic = {"person a", "person b", "person 1", "person 2"}
        for char in context.characters:
            if char.visual_anchor.strip().lower() in generic:
                return False
        return True

    def validate_subject_coverage(self, context: SceneContext) -> bool:
        """Ensure both characters receive focus in a two-character scene."""

        if len(context.characters) < 2:
            return True

        focused_subjects = {
            subject
            for shot in context.shot_blocks
            for subject in shot.framing.subjects
            if subject in {"CHAR_A", "CHAR_B"}
        }
        return {"CHAR_A", "CHAR_B"}.issubset(focused_subjects)

    def validate_cinematic_progression(self, context: SceneContext) -> bool:
        """Check that the shot sequence escalates in a production-usable way."""

        if not context.shot_blocks:
            return False

        if len(context.characters) >= 2 and context.shot_blocks[0].framing.shot_type.value != "MASTER_SHOT":
            return False

        if len(context.shot_blocks) >= 5:
            distinct_beats = {shot.beat_type for shot in context.shot_blocks if shot.beat_type is not None}
            if len(distinct_beats) < 3:
                return False

        if context.detected_emotion in {"angry", "furious", "scared", "tense", "romantic", "sad"}:
            late_shots = context.shot_blocks[-2:]
            if not any(shot.framing.shot_type.value in {"MCU", "CLOSE_UP"} for shot in late_shots):
                return False

        return True

    def validate_dialogue_carryover(self, context: SceneContext) -> bool:
        """If the source contains quotes, extracted dialogue should not be empty."""

        if '"' not in context.description and "“" not in context.description and "「" not in context.description:
            return True
        return bool(context.dialogue)

    def acceptance_report(self, context: SceneContext) -> dict[str, bool]:
        """Return a grouped acceptance report matching DESIGN.md."""

        return {
            "logic_axis_consistency": self.validate_axis_consistency(context),
            "logic_gaze_direction": self.validate_gaze_logic(context),
            "data_required_fields": self.validate_data_completeness(context),
            "data_visual_anchor_consistency": self.validate_visual_anchor_consistency(context),
            "interface_prompt_quality": self.validate_prompt_quality(context),
            "interface_controlnet_coords": self.validate_controlnet_range(context),
        }

    def production_readiness_report(self, context: SceneContext) -> dict[str, bool]:
        """Return stricter quality checks for industrial production use."""

        return {
            "llm_analysis_active": context.analysis_source == "llm",
            "anchor_specificity": self.validate_anchor_specificity(context),
            "subject_coverage": self.validate_subject_coverage(context),
            "cinematic_progression": self.validate_cinematic_progression(context),
            "dialogue_carryover": self.validate_dialogue_carryover(context),
            **self.acceptance_report(context),
        }

    def quality_report(
        self,
        context: SceneContext,
        min_score: float = 0.85,
    ) -> dict[str, Any]:
        """Return weighted quality metrics and gate decision for a scene."""

        metrics: dict[str, float] = {
            "axis_consistency": 1.0 if self.validate_axis_consistency(context) else 0.0,
            "gaze_logic": 1.0 if self.validate_gaze_logic(context) else 0.0,
            "anchor_specificity": 1.0 if self.validate_anchor_specificity(context) else 0.0,
            "subject_coverage": 1.0 if self.validate_subject_coverage(context) else 0.0,
            "cinematic_progression": 1.0 if self.validate_cinematic_progression(context) else 0.0,
            "prompt_quality": 1.0 if self.validate_prompt_quality(context) else 0.0,
            "controlnet_range": 1.0 if self.validate_controlnet_range(context) else 0.0,
            "dialogue_carryover": 1.0 if self.validate_dialogue_carryover(context) else 0.0,
            "beat_diversity": min(
                1.0,
                len({shot.beat_type for shot in context.shot_blocks if shot.beat_type is not None}) / 4.0,
            ),
            "shot_type_diversity": min(
                1.0,
                len({shot.framing.shot_type for shot in context.shot_blocks}) / 3.0,
            ),
        }
        weights = {
            "axis_consistency": 0.16,
            "gaze_logic": 0.12,
            "anchor_specificity": 0.12,
            "subject_coverage": 0.08,
            "cinematic_progression": 0.12,
            "prompt_quality": 0.10,
            "controlnet_range": 0.08,
            "dialogue_carryover": 0.06,
            "beat_diversity": 0.08,
            "shot_type_diversity": 0.08,
        }
        score = round(sum(metrics[name] * weights[name] for name in weights), 4)
        issues = [
            name
            for name, value in metrics.items()
            if value < 1.0 and name in {"axis_consistency", "gaze_logic", "anchor_specificity", "cinematic_progression"}
        ]

        return {
            "score": score,
            "min_score": min_score,
            "passes_gate": score >= min_score and not issues,
            "metrics": metrics,
            "critical_issues": issues,
            "analysis_source": context.analysis_source,
        }

    def run_with_quality_gate(
        self,
        description: str,
        num_shots: int = 5,
        scene_id: str = "SCENE_01",
        use_llm: bool = True,
        emotion_override: Optional[str] = None,
        min_score: float = 0.85,
        max_attempts: int = 3,
    ) -> tuple[SceneContext, dict[str, Any]]:
        """Run the pipeline with retry strategies until the scene passes a quality gate."""

        strategies: list[tuple[bool, int, str]] = [(use_llm, num_shots, "requested")]
        if use_llm:
            strategies.append((False, num_shots, "rule_based_fallback"))
        if num_shots < 5:
            strategies.append((use_llm, 5, "expanded_shot_count"))
        if use_llm and num_shots < 5:
            strategies.append((False, 5, "fallback_expanded_shot_count"))

        seen: set[tuple[bool, int, str]] = set()
        ordered_strategies = []
        for strategy in strategies:
            if strategy not in seen:
                ordered_strategies.append(strategy)
                seen.add(strategy)

        best_context: SceneContext | None = None
        best_report: dict[str, Any] | None = None
        for attempt_index, (attempt_use_llm, attempt_num_shots, strategy_name) in enumerate(
            ordered_strategies[:max_attempts],
            start=1,
        ):
            context = self.run(
                description=description,
                num_shots=attempt_num_shots,
                scene_id=scene_id,
                use_llm=attempt_use_llm,
                emotion_override=emotion_override,
            )
            report = self.quality_report(context, min_score=min_score)
            report = {
                **report,
                "attempt": attempt_index,
                "strategy": strategy_name,
            }

            if best_report is None or report["score"] > best_report["score"]:
                best_context = context
                best_report = report

            if report["passes_gate"]:
                return context, report

        assert best_context is not None and best_report is not None
        return best_context, best_report

    def build_storyboard_package(
        self,
        context: SceneContext,
        project_name: str = "Auto-CineFlow Project",
        render_preset: RenderPreset | None = None,
        generated_at: str | None = None,
    ) -> StoryboardPackage:
        """Build a production delivery package for the scene."""

        return build_storyboard_package(
            context=context,
            project_name=project_name,
            render_preset=render_preset,
            readiness_report=self.production_readiness_report(context),
            quality_report=self.quality_report(context),
            generated_at=generated_at,
        )

    def storyboard_package_json(
        self,
        package: StoryboardPackage,
        indent: int = 2,
    ) -> str:
        """Serialise a delivery package to JSON."""

        return package_to_json(package, indent=indent)

    def shotlist_csv(self, package: StoryboardPackage) -> str:
        """Export the packaged shot list to CSV."""

        return shotlist_to_csv(package)

    def render_queue_json(self, package: StoryboardPackage, indent: int = 2) -> str:
        """Serialise the packaged render queue to JSON."""

        return render_queue_to_json(package, indent=indent)

    def character_bible_json(self, package: StoryboardPackage, indent: int = 2) -> str:
        """Serialise the packaged character bible to JSON."""

        return character_bible_to_json(package, indent=indent)

    def video_plan_json(self, package: StoryboardPackage, indent: int = 2) -> str:
        """Serialise the packaged video generation plan to JSON."""

        return video_plan_to_json(package, indent=indent)

    def edl_text(self, package: StoryboardPackage) -> str:
        """Export a simple editorial decision list."""

        return edl_text(package)

    def storyboard_review_markdown(self, package: StoryboardPackage) -> str:
        """Export a human-readable storyboard review document."""

        return storyboard_review_markdown(package)

    def automatic1111_bundle_json(self, package: StoryboardPackage, indent: int = 2) -> str:
        """Serialise Automatic1111-style payloads."""

        return automatic1111_bundle_json(package, indent=indent)

    def comfyui_bundle_json(self, package: StoryboardPackage, indent: int = 2) -> str:
        """Serialise ComfyUI-oriented bundles."""

        return comfyui_bundle_json(package, indent=indent)

    def render_manifest_template_json(self, package: StoryboardPackage, indent: int = 2) -> str:
        """Serialise the expected render manifest template."""

        return render_manifest_template_json(package, indent=indent)

    def render_qa_report(
        self,
        package: StoryboardPackage,
        manifest: list[RenderExpectation],
        min_score: float = 0.9,
    ) -> RenderQAReport:
        """Evaluate a render manifest against storyboard expectations."""

        return render_qa_report(package, manifest, min_score=min_score)

    def render_qa_report_json(self, report: RenderQAReport, indent: int = 2) -> str:
        """Serialise a render QA report."""

        return render_qa_report_json(report, indent=indent)

    def render_qa_review_markdown(self, report: RenderQAReport) -> str:
        """Export a human-readable render QA review document."""

        return render_qa_review_markdown(report)

    def write_delivery_package(
        self,
        package: StoryboardPackage,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Write packaged delivery assets to disk."""

        files = write_storyboard_package(package, output_dir)
        provider_files = write_provider_payloads(package, Path(output_dir) / "providers")
        render_manifest_path = Path(output_dir) / "render_manifest_template.json"
        render_manifest_path.write_text(
            self.render_manifest_template_json(package, indent=2),
            encoding="utf-8",
        )
        assembly_files = self.write_sequence_assembly_plan(
            self.build_sequence_assembly_plan(package),
            Path(output_dir) / "assembly",
        )
        return {
            **files,
            **provider_files,
            "render_manifest_template": render_manifest_path,
            **assembly_files,
        }

    def write_render_qa_report(
        self,
        report: RenderQAReport,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Write render QA outputs to disk."""

        return write_render_qa_report(report, output_dir)

    def build_project_package(
        self,
        scene_inputs: list[ProjectSceneInput],
        project_name: str = "Auto-CineFlow Project",
        use_llm: bool = True,
        min_score: float = 0.85,
        max_attempts: int = 3,
    ) -> ProjectPackage:
        """Build a project package from multiple scene requests."""

        scene_packages: list[StoryboardPackage] = []
        for scene in scene_inputs:
            context, _ = self.run_with_quality_gate(
                description=scene.description,
                num_shots=scene.num_shots,
                scene_id=scene.scene_id,
                use_llm=use_llm,
                emotion_override=scene.emotion_override,
                min_score=min_score,
                max_attempts=max_attempts,
            )
            scene_packages.append(self.build_storyboard_package(context, project_name=project_name))

        return build_project_package(project_name=project_name, scenes=scene_packages)

    def project_manifest_json(self, package: ProjectPackage, indent: int = 2) -> str:
        """Serialise a project package to JSON."""

        return project_manifest_json(package, indent=indent)

    def project_shotlist_csv(self, package: ProjectPackage) -> str:
        """Export a flattened multi-scene shot list."""

        return project_shotlist_csv(package)

    def project_review_markdown(self, package: ProjectPackage) -> str:
        """Export a project-level review markdown document."""

        return project_review_markdown(package)

    def write_project_package(
        self,
        package: ProjectPackage,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Write a full project package to disk."""

        return write_project_package(package, output_dir)

    def build_project_change_plan(
        self,
        previous: ProjectPackage,
        current: ProjectPackage,
    ) -> ProjectChangePlan:
        """Build an incremental rerender plan between two project packages."""

        return build_project_change_plan(previous, current)

    def project_change_plan_json(self, plan: ProjectChangePlan, indent: int = 2) -> str:
        """Serialise a project change plan."""

        return project_change_plan_json(plan, indent=indent)

    def project_change_review_markdown(self, plan: ProjectChangePlan) -> str:
        """Export a human-readable project change review document."""

        return project_change_review_markdown(plan)

    def write_project_change_plan(
        self,
        plan: ProjectChangePlan,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Write a project change plan to disk."""

        return write_project_change_plan(plan, output_dir)

    def build_project_execution_plan(
        self,
        previous: ProjectPackage,
        current: ProjectPackage,
        previous_scenes_dir: str | Path,
    ) -> ProjectExecutionPlan:
        """Build a reuse/rerender execution plan from previous renders."""

        return build_project_execution_plan(previous, current, previous_scenes_dir)

    def project_execution_plan_json(self, plan: ProjectExecutionPlan, indent: int = 2) -> str:
        """Serialise a project execution plan."""

        return project_execution_plan_json(plan, indent=indent)

    def project_execution_review_markdown(self, plan: ProjectExecutionPlan) -> str:
        """Export a human-readable execution review document."""

        return project_execution_review_markdown(plan)

    def write_project_execution_plan(
        self,
        plan: ProjectExecutionPlan,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Write a project execution plan to disk."""

        return write_project_execution_plan(plan, output_dir)

    def build_project_render_qa_report(
        self,
        project: ProjectPackage,
        scenes_dir: str | Path,
        min_score: float = 0.9,
    ) -> ProjectRenderQAReport:
        """Aggregate render QA across a project."""

        return build_project_render_qa_report(project, scenes_dir, min_score=min_score)

    def project_render_qa_report_json(self, report: ProjectRenderQAReport, indent: int = 2) -> str:
        """Serialise a project render QA report."""

        return project_render_qa_report_json(report, indent=indent)

    def project_render_qa_review_markdown(self, report: ProjectRenderQAReport) -> str:
        """Export a human-readable project render QA report."""

        return project_render_qa_review_markdown(report)

    def write_project_render_qa_report(
        self,
        report: ProjectRenderQAReport,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Write a project render QA report to disk."""

        return write_project_render_qa_report(report, output_dir)

    def build_project_dashboard(
        self,
        project: ProjectPackage,
        render_qa: ProjectRenderQAReport | None = None,
        execution_plan: ProjectExecutionPlan | None = None,
    ) -> ProjectDashboard:
        """Build a unified project dashboard."""

        return build_project_dashboard(project, render_qa=render_qa, execution_plan=execution_plan)

    def project_dashboard_json(self, dashboard: ProjectDashboard, indent: int = 2) -> str:
        """Serialise a project dashboard."""

        return project_dashboard_json(dashboard, indent=indent)

    def project_dashboard_markdown(self, dashboard: ProjectDashboard) -> str:
        """Export a human-readable project dashboard."""

        return project_dashboard_markdown(dashboard)

    def write_project_dashboard(
        self,
        dashboard: ProjectDashboard,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Write project dashboard assets to disk."""

        return write_project_dashboard(dashboard, output_dir)

    def build_submission_jobs_from_package(
        self,
        package: StoryboardPackage,
        provider: SubmissionProvider = SubmissionProvider.GENERIC,
    ) -> list[SubmissionJob]:
        """Build submit-ready jobs from a storyboard package."""

        return build_submission_jobs_from_package(package, provider=provider)

    def build_submission_jobs_from_execution_plan(
        self,
        plan: ProjectExecutionPlan,
        provider: SubmissionProvider = SubmissionProvider.GENERIC,
    ) -> list[SubmissionJob]:
        """Build submit-ready jobs from an execution plan."""

        return build_submission_jobs_from_execution_plan(plan, provider=provider)

    def submit_jobs(
        self,
        jobs: list[SubmissionJob],
        target: SubmissionTarget,
        source_type: str,
        source_id: str,
    ) -> SubmissionBatch:
        """Submit jobs using the selected backend."""

        return submit_jobs(jobs, target, source_type=source_type, source_id=source_id)

    def submission_batch_json(self, batch: SubmissionBatch, indent: int = 2) -> str:
        """Serialise a submission batch."""

        return submission_batch_json(batch, indent=indent)

    def submission_batch_markdown(self, batch: SubmissionBatch) -> str:
        """Export a human-readable submission batch."""

        return submission_batch_markdown(batch)

    def write_submission_batch(
        self,
        batch: SubmissionBatch,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Write submission batch outputs to disk."""

        return write_submission_batch(batch, output_dir)

    def monitor_filesystem_submission_batch(
        self,
        batch: SubmissionBatch,
        spool_dir: str | Path,
    ) -> SubmissionMonitorReport:
        """Inspect a filesystem spool for the current state of a submission batch."""

        return monitor_filesystem_submission_batch(batch, spool_dir)

    def submission_monitor_report_json(self, report: SubmissionMonitorReport, indent: int = 2) -> str:
        """Serialise a submission monitor report."""

        return submission_monitor_report_json(report, indent=indent)

    def submission_monitor_markdown(self, report: SubmissionMonitorReport) -> str:
        """Export a human-readable submission monitor report."""

        return submission_monitor_markdown(report)

    def write_submission_monitor_report(
        self,
        report: SubmissionMonitorReport,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Write submission monitor outputs to disk."""

        return write_submission_monitor_report(report, output_dir)

    def recommended_video_shot_count(
        self,
        target_duration_seconds: float,
        clip_duration_seconds: float = 4.0,
        min_shots: int = 8,
        max_shots: int = 24,
    ) -> int:
        """Recommend a finer-grained shot count for assembled long-form previs."""

        return recommended_shot_count(
            target_duration_seconds=target_duration_seconds,
            clip_duration_seconds=clip_duration_seconds,
            min_shots=min_shots,
            max_shots=max_shots,
        )

    def download_submission_artifacts(
        self,
        batch: SubmissionBatch,
        output_dir: str | Path,
        config_path: str | None = None,
        timeout_seconds: float = 900.0,
        poll_interval_seconds: float = 10.0,
    ) -> ArtifactDownloadBatch:
        """Download URL-based artifacts referenced by a submission batch."""

        return download_submission_artifacts(
            batch,
            output_dir,
            config_path=config_path,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    def artifact_download_batch_json(self, batch: ArtifactDownloadBatch, indent: int = 2) -> str:
        """Serialise an artifact download batch."""

        return artifact_download_batch_json(batch, indent=indent)

    def write_artifact_download_batch(
        self,
        batch: ArtifactDownloadBatch,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Write artifact download metadata to disk."""

        return write_artifact_download_batch(batch, output_dir)

    def update_render_manifest_from_downloads(
        self,
        manifest_path: str | Path,
        downloads: ArtifactDownloadBatch,
    ):
        """Sync successful artifact downloads back into a render manifest."""

        return update_render_manifest_from_downloads(manifest_path, downloads)

    def build_sequence_assembly_plan(
        self,
        package: StoryboardPackage,
        artifacts_dir: str = "artifacts",
    ) -> SequenceAssemblyPlan:
        """Build a clip-by-clip sequence assembly plan."""

        return build_sequence_assembly_plan(package, artifacts_dir=artifacts_dir)

    def sequence_assembly_json(self, plan: SequenceAssemblyPlan, indent: int = 2) -> str:
        """Serialise a sequence assembly plan."""

        return sequence_assembly_json(plan, indent=indent)

    def write_sequence_assembly_plan(
        self,
        plan: SequenceAssemblyPlan,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Write sequence assembly planning assets to disk."""

        return write_sequence_assembly_plan(plan, output_dir)
