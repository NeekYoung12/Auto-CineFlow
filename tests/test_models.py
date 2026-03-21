"""Tests for Pydantic models and data structures."""

import pytest
from pydantic import ValidationError

from autocineflow.models import (
    EMOTION_MATRIX,
    SHOT_TEMPLATES,
    AxisSide,
    CameraAngle,
    CameraAngleType,
    Character,
    CharacterFacing,
    FramingParams,
    LightingParams,
    MotionInstruction,
    Position,
    SceneContext,
    ShotBlock,
    ShotType,
)


class TestPosition:
    def test_valid_position(self):
        pos = Position(x=0.33, y=0.4)
        assert pos.x == 0.33
        assert pos.y == 0.4

    def test_out_of_range_x(self):
        with pytest.raises(ValidationError):
            Position(x=1.5, y=0.5)

    def test_out_of_range_y(self):
        with pytest.raises(ValidationError):
            Position(x=0.5, y=-0.1)

    def test_boundary_values(self):
        pos = Position(x=0.0, y=1.0)
        assert pos.x == 0.0
        assert pos.y == 1.0


class TestCharacter:
    def test_valid_character(self):
        char = Character(
            char_id="CHAR_A",
            visual_anchor="tall man with grey coat",
            pos=Position(x=0.33, y=0.4),
            facing=CharacterFacing.RIGHT,
        )
        assert char.char_id == "CHAR_A"
        assert char.facing == CharacterFacing.RIGHT

    def test_visual_anchor_immutability(self):
        """Visual anchor must match original value exactly — acceptance criterion."""
        anchor = "woman with red hair, wearing a blue dress"
        char = Character(
            char_id="CHAR_A",
            visual_anchor=anchor,
            pos=Position(x=0.33, y=0.5),
            facing=CharacterFacing.RIGHT,
        )
        assert char.visual_anchor == anchor

    def test_facing_logic_left_facing_left_invalid(self):
        """A character on the left (x < 0.5) must not face LEFT."""
        with pytest.raises(ValidationError):
            Character(
                char_id="CHAR_A",
                visual_anchor="person A",
                pos=Position(x=0.25, y=0.5),
                facing=CharacterFacing.LEFT,  # invalid: left-side char must face RIGHT
            )

    def test_facing_logic_right_facing_right_invalid(self):
        """A character on the right (x > 0.5) must not face RIGHT."""
        with pytest.raises(ValidationError):
            Character(
                char_id="CHAR_B",
                visual_anchor="person B",
                pos=Position(x=0.75, y=0.5),
                facing=CharacterFacing.RIGHT,  # invalid: right-side char must face LEFT
            )

    def test_facing_camera_is_always_valid(self):
        """CAMERA facing is valid for any position."""
        char = Character(
            char_id="CHAR_A",
            visual_anchor="person A",
            pos=Position(x=0.2, y=0.5),
            facing=CharacterFacing.CAMERA,
        )
        assert char.facing == CharacterFacing.CAMERA


class TestShotBlock:
    def _make_shot_block(self, shot_index: int = 0) -> ShotBlock:
        return ShotBlock(
            shot_index=shot_index,
            framing=FramingParams(
                shot_type=ShotType.MEDIUM_SHOT,
                focal_length_mm=35,
                subjects=["CHAR_A"],
            ),
            camera_angle=CameraAngle(
                angle_type=CameraAngleType.EYE_LEVEL,
                axis_side=AxisSide.LEFT,
            ),
            lighting=LightingParams(contrast=0.5, saturation=0.5, mood="neutral"),
            motion_instruction=MotionInstruction(motion_type="STATIC", intensity=0.0),
        )

    def test_shot_block_has_all_required_fields(self):
        """Data Pass: every ShotBlock must contain framing, camera_angle, lighting, motion."""
        shot = self._make_shot_block()
        assert shot.framing is not None
        assert shot.camera_angle is not None
        assert shot.lighting is not None
        assert shot.motion_instruction is not None

    def test_shot_index_non_negative(self):
        with pytest.raises(ValidationError):
            ShotBlock(
                shot_index=-1,
                framing=FramingParams(shot_type=ShotType.MCU, focal_length_mm=50, subjects=[]),
                camera_angle=CameraAngle(angle_type=CameraAngleType.EYE_LEVEL, axis_side=AxisSide.LEFT),
                lighting=LightingParams(contrast=0.5, saturation=0.5, mood="neutral"),
                motion_instruction=MotionInstruction(motion_type="STATIC", intensity=0.0),
            )


class TestShotTemplates:
    def test_all_shot_types_have_template(self):
        """REQ-03: every ShotType must have a corresponding template."""
        for shot_type in ShotType:
            assert shot_type in SHOT_TEMPLATES, f"Missing template for {shot_type}"

    def test_focal_lengths(self):
        """Shot templates must use industry-standard focal lengths."""
        assert SHOT_TEMPLATES[ShotType.MASTER_SHOT].focal_length_mm == 24
        assert SHOT_TEMPLATES[ShotType.CLOSE_UP].focal_length_mm == 85


class TestEmotionMatrix:
    def test_all_emotions_present(self):
        expected = {"angry", "furious", "sad", "happy", "joyful", "tense", "scared", "calm", "romantic", "neutral"}
        assert expected == set(EMOTION_MATRIX.keys())

    def test_angry_params(self):
        """REQ-04: 'angry' must map to high contrast and high motion."""
        params = EMOTION_MATRIX["angry"]
        assert params["contrast"] == 0.8
        assert params["motion"] == 0.7

    def test_all_values_in_range(self):
        for emotion, params in EMOTION_MATRIX.items():
            assert 0.0 <= params["contrast"] <= 1.0, f"{emotion}.contrast out of range"
            assert 0.0 <= params["saturation"] <= 1.0, f"{emotion}.saturation out of range"
            assert 0.0 <= params["motion"] <= 1.0, f"{emotion}.motion out of range"
