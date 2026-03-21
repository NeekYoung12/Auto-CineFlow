"""Main pipeline: Pipe-and-Filter architecture for Auto-CineFlow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .delivery import (
    RenderPreset,
    StoryboardPackage,
    build_storyboard_package,
    character_bible_to_json,
    edl_text,
    package_to_json,
    render_queue_to_json,
    shotlist_to_csv,
    write_storyboard_package,
)
from .director_logic import build_shot, plan_scene_beats
from .models import SceneContext
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

    def edl_text(self, package: StoryboardPackage) -> str:
        """Export a simple editorial decision list."""

        return edl_text(package)

    def write_delivery_package(
        self,
        package: StoryboardPackage,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Write packaged delivery assets to disk."""

        return write_storyboard_package(package, output_dir)
