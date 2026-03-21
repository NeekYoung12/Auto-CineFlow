"""Main pipeline: Pipe-and-Filter architecture for Auto-CineFlow.

Filters (in order):
  1. Parser Filter   — script_analyzer.analyse_script()
  2. Director Filter — director_logic.build_shot()  (repeated N times)
  3. Geometry Filter — spatial_solver (called inside director_logic)
  4. Formatter Filter— prompt_builder.attach_prompts()

Usage::

    from autocineflow.pipeline import CineFlowPipeline

    pipeline = CineFlowPipeline()
    result = pipeline.run(
        description="Two people sit across from each other in a tavern, tension rising.",
        num_shots=5,
        use_llm=False,
    )
    for shot in result.shot_blocks:
        print(shot.model_dump_json(indent=2))
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from .director_logic import build_shot
from .models import SceneContext, ShotBlock
from .prompt_builder import attach_prompts
from .script_analyzer import analyse_script
from .spatial_solver import positions_to_controlnet

logger = logging.getLogger(__name__)


class CineFlowPipeline:
    """End-to-end pipeline that converts a scene description into a ShotBlock sequence."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
    ) -> None:
        self.api_key = api_key
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        description: str,
        num_shots: int = 5,
        scene_id: str = "SCENE_01",
        use_llm: bool = True,
        emotion_override: Optional[str] = None,
    ) -> SceneContext:
        """Run the full pipeline and return a SceneContext with shot_blocks populated.

        Args:
            description:      Natural-language scene description.
            num_shots:        Number of shots to generate.
            scene_id:         Identifier for this scene.
            use_llm:          Whether to use the LLM for script analysis.
            emotion_override: Force a specific emotion (skips detection).

        Returns:
            SceneContext with shot_blocks containing fully assembled ShotBlocks.
        """
        # ---- Filter 1: Parser ----
        logger.info("Parser Filter: analysing script…")
        context = analyse_script(
            description,
            scene_id=scene_id,
            use_llm=use_llm,
            api_key=self.api_key,
            model=self.model,
        )

        if emotion_override:
            context = context.model_copy(update={"detected_emotion": emotion_override})

        logger.info(
            "Detected %d character(s), emotion=%s",
            len(context.characters),
            context.detected_emotion,
        )

        # ---- Filters 2+3: Director + Geometry (interleaved per shot) ----
        for shot_idx in range(num_shots):
            logger.info("Director+Geometry Filter: building shot %d…", shot_idx)
            shot_block, context = build_shot(context, shot_idx)

            # ---- Filter 4: Formatter ----
            shot_block = attach_prompts(shot_block)

            # Replace the last appended shot with the prompt-enriched version
            updated_shots = list(context.shot_blocks[:-1]) + [shot_block]
            context = context.model_copy(update={"shot_blocks": updated_shots})

        logger.info("Pipeline complete. %d shots generated.", num_shots)
        return context

    def to_json(self, context: SceneContext, indent: int = 2) -> str:
        """Serialise the full SceneContext to a JSON string."""
        return context.model_dump_json(indent=indent)

    def controlnet_coords(self, context: SceneContext) -> list[dict]:
        """Return ControlNet coordinate dicts for the final shot's characters."""
        if not context.shot_blocks:
            return []
        final_shot = context.shot_blocks[-1]
        return positions_to_controlnet(final_shot.characters)

    def validate_axis_consistency(self, context: SceneContext) -> bool:
        """Verify that all shots in the context share the same axis_side.

        Acceptance criterion: 'axis_side' parameter must remain constant
        (unless the scene is reset).

        Returns:
            True if all shots are axis-consistent, False otherwise.
        """
        sides = {sb.camera_angle.axis_side for sb in context.shot_blocks}
        return len(sides) <= 1

    def validate_visual_anchor_consistency(self, context: SceneContext) -> bool:
        """Verify that each character's visual_anchor is identical across all shots.

        Acceptance criterion: CHAR_A's visual description must be 100%
        character-level identical throughout the JSON sequence.

        Returns:
            True if all anchors are consistent, False otherwise.
        """
        anchor_registry: dict[str, str] = {}
        for shot in context.shot_blocks:
            for char in shot.characters:
                if char.char_id in anchor_registry:
                    if anchor_registry[char.char_id] != char.visual_anchor:
                        return False
                else:
                    anchor_registry[char.char_id] = char.visual_anchor
        return True
