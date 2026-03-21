"""Tests for director logic (shot selection, axis enforcement, emotion mapping)."""

import pytest

from autocineflow.director_logic import (
    build_camera_angle,
    build_framing,
    build_lighting,
    build_motion,
    build_shot,
    choose_shot_type,
)
from autocineflow.models import (
    AxisSide,
    CameraAngleType,
    Character,
    CharacterFacing,
    Position,
    SceneContext,
    ShotType,
)


def _make_two_char_context(emotion: str = "neutral") -> SceneContext:
    return SceneContext(
        scene_id="TEST_SCENE",
        description="Two people face each other.",
        characters=[
            Character(
                char_id="CHAR_A",
                visual_anchor="man in black coat",
                pos=Position(x=0.25, y=0.5),
                facing=CharacterFacing.RIGHT,
            ),
            Character(
                char_id="CHAR_B",
                visual_anchor="woman in red dress",
                pos=Position(x=0.75, y=0.5),
                facing=CharacterFacing.LEFT,
            ),
        ],
        detected_emotion=emotion,
    )


class TestChooseShotType:
    def test_first_shot_two_chars_is_master(self):
        """REQ-03: First shot of a two-character scene must be MASTER_SHOT."""
        assert choose_shot_type("neutral", shot_index=0, num_characters=2) == ShotType.MASTER_SHOT

    def test_first_shot_single_char_not_master(self):
        """Single character: first shot doesn't need to be MASTER_SHOT."""
        result = choose_shot_type("neutral", shot_index=0, num_characters=1)
        assert result != ShotType.MASTER_SHOT

    def test_angry_emotion_gives_closeup(self):
        assert choose_shot_type("angry", shot_index=1, num_characters=2) == ShotType.CLOSE_UP

    def test_tense_emotion_gives_mcu(self):
        assert choose_shot_type("tense", shot_index=1, num_characters=1) == ShotType.MCU

    def test_happy_emotion_gives_medium(self):
        assert choose_shot_type("happy", shot_index=2, num_characters=2) == ShotType.MEDIUM_SHOT


class TestBuildFraming:
    def test_framing_matches_shot_type(self):
        framing = build_framing(ShotType.CLOSE_UP, ["CHAR_A"])
        assert framing.shot_type == ShotType.CLOSE_UP
        assert framing.focal_length_mm == 85

    def test_framing_subjects(self):
        framing = build_framing(ShotType.MCU, ["CHAR_A", "CHAR_B"])
        assert "CHAR_A" in framing.subjects
        assert "CHAR_B" in framing.subjects


class TestBuildLighting:
    def test_angry_lighting(self):
        """REQ-04: angry emotion must produce high-contrast lighting."""
        lighting = build_lighting("angry")
        assert lighting.contrast == 0.8
        assert lighting.saturation == 0.6

    def test_sad_lighting(self):
        lighting = build_lighting("sad")
        assert lighting.saturation == 0.2

    def test_neutral_lighting_balanced(self):
        lighting = build_lighting("neutral")
        assert lighting.contrast == 0.5
        assert lighting.saturation == 0.5


class TestBuildCameraAngle:
    def test_axis_established_on_first_shot(self):
        """REQ-02: axis_side must be set on the first shot."""
        ctx = _make_two_char_context()
        angle = build_camera_angle(ctx, ShotType.MASTER_SHOT, "neutral")
        assert angle.axis_side in (AxisSide.LEFT, AxisSide.RIGHT)

    def test_axis_locked_after_first_shot(self):
        """REQ-02: axis_side must not change after being established."""
        ctx = _make_two_char_context()
        # First shot sets the axis
        angle1 = build_camera_angle(ctx, ShotType.MASTER_SHOT, "neutral")
        ctx = ctx.model_copy(update={"axis_side": angle1.axis_side})

        # Subsequent shots must keep the same axis
        for _ in range(4):
            angle = build_camera_angle(ctx, ShotType.MCU, "angry")
            assert angle.axis_side == angle1.axis_side

    def test_tense_emotion_low_angle(self):
        ctx = _make_two_char_context("tense")
        angle = build_camera_angle(ctx, ShotType.MCU, "tense")
        assert angle.angle_type == CameraAngleType.LOW_ANGLE

    def test_sad_emotion_high_angle(self):
        ctx = _make_two_char_context("sad")
        angle = build_camera_angle(ctx, ShotType.MCU, "sad")
        assert angle.angle_type == CameraAngleType.HIGH_ANGLE


class TestBuildShot:
    def test_shot_has_all_required_fields(self):
        """Data Pass: every built shot must have framing, camera_angle, lighting, motion."""
        ctx = _make_two_char_context()
        shot, _ = build_shot(ctx, shot_index=0)
        assert shot.framing is not None
        assert shot.camera_angle is not None
        assert shot.lighting is not None
        assert shot.motion_instruction is not None

    def test_first_shot_is_master(self):
        ctx = _make_two_char_context()
        shot, _ = build_shot(ctx, shot_index=0)
        assert shot.framing.shot_type == ShotType.MASTER_SHOT

    def test_axis_consistency_five_shots(self):
        """Logic Pass: 5 consecutive shots must share the same axis_side."""
        ctx = _make_two_char_context()
        axis_sides = []
        for i in range(5):
            shot, ctx = build_shot(ctx, shot_index=i)
            axis_sides.append(shot.camera_angle.axis_side)

        assert len(set(axis_sides)) == 1, f"Axis changed across shots: {axis_sides}"

    def test_character_positions_updated(self):
        """Characters must have positions after build_shot."""
        ctx = _make_two_char_context()
        shot, _ = build_shot(ctx, shot_index=0)
        for char in shot.characters:
            assert 0.0 <= char.pos.x <= 1.0

    def test_visual_anchor_unchanged(self):
        """Visual anchors must be identical across all shots."""
        ctx = _make_two_char_context()
        original_anchors = {c.char_id: c.visual_anchor for c in ctx.characters}

        for i in range(5):
            shot, ctx = build_shot(ctx, shot_index=i)
            for char in shot.characters:
                assert char.visual_anchor == original_anchors[char.char_id], (
                    f"Visual anchor changed for {char.char_id} at shot {i}"
                )
