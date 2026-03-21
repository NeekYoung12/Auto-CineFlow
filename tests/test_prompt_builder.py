"""Tests for the prompt builder."""

import pytest

from autocineflow.models import (
    AxisSide,
    CameraAngle,
    CameraAngleType,
    Character,
    CharacterFacing,
    FramingParams,
    LightingParams,
    MotionInstruction,
    Position,
    ShotBlock,
    ShotType,
)
from autocineflow.prompt_builder import (
    attach_prompts,
    build_negative_prompt,
    build_sd_prompt,
)


def _make_shot_block(
    shot_type: ShotType = ShotType.MCU,
    emotion_contrast: float = 0.5,
    emotion_saturation: float = 0.5,
    mood: str = "neutral",
) -> ShotBlock:
    return ShotBlock(
        shot_index=0,
        framing=FramingParams(
            shot_type=shot_type,
            focal_length_mm=50,
            subjects=["CHAR_A"],
        ),
        camera_angle=CameraAngle(
            angle_type=CameraAngleType.EYE_LEVEL,
            axis_side=AxisSide.LEFT,
        ),
        lighting=LightingParams(
            contrast=emotion_contrast,
            saturation=emotion_saturation,
            mood=mood,
        ),
        motion_instruction=MotionInstruction(motion_type="STATIC", intensity=0.0),
        characters=[
            Character(
                char_id="CHAR_A",
                visual_anchor="man in black coat",
                pos=Position(x=0.33, y=0.4),
                facing=CharacterFacing.RIGHT,
            )
        ],
    )


class TestBuildSdPrompt:
    def test_prompt_contains_shot_type_token(self):
        shot = _make_shot_block(ShotType.CLOSE_UP)
        prompt = build_sd_prompt(shot)
        assert "close-up" in prompt.lower()

    def test_prompt_contains_visual_anchor(self):
        shot = _make_shot_block()
        prompt = build_sd_prompt(shot)
        assert "man in black coat" in prompt

    def test_prompt_contains_focal_length(self):
        shot = _make_shot_block()
        prompt = build_sd_prompt(shot)
        assert "50mm" in prompt

    def test_prompt_contains_lighting_tokens(self):
        shot = _make_shot_block(mood="intense")
        prompt = build_sd_prompt(shot)
        # intense mood → "harsh lighting" or "high contrast"
        assert any(tok in prompt.lower() for tok in ("harsh", "contrast"))

    def test_prompt_contains_quality_tokens(self):
        shot = _make_shot_block()
        prompt = build_sd_prompt(shot)
        assert "cinematic" in prompt.lower()

    def test_prompt_is_non_empty_string(self):
        shot = _make_shot_block()
        prompt = build_sd_prompt(shot)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_master_shot_tokens(self):
        shot = _make_shot_block(ShotType.MASTER_SHOT)
        prompt = build_sd_prompt(shot)
        assert "wide" in prompt.lower() or "establishing" in prompt.lower()


class TestBuildNegativePrompt:
    def test_close_up_negative_contains_wide(self):
        shot = _make_shot_block(ShotType.CLOSE_UP)
        neg = build_negative_prompt(shot)
        assert "wide shot" in neg.lower()

    def test_master_shot_negative_contains_close_up(self):
        shot = _make_shot_block(ShotType.MASTER_SHOT)
        neg = build_negative_prompt(shot)
        assert "close-up" in neg.lower()

    def test_negative_contains_base_tokens(self):
        shot = _make_shot_block()
        neg = build_negative_prompt(shot)
        assert "blurry" in neg.lower()
        assert "deformed" in neg.lower()


class TestAttachPrompts:
    def test_attach_populates_sd_prompt(self):
        shot = _make_shot_block()
        enriched = attach_prompts(shot)
        assert enriched.sd_prompt != ""

    def test_attach_populates_negative_prompt(self):
        shot = _make_shot_block()
        enriched = attach_prompts(shot)
        assert enriched.negative_prompt != ""

    def test_original_not_mutated(self):
        shot = _make_shot_block()
        _ = attach_prompts(shot)
        assert shot.sd_prompt == ""  # original unchanged
