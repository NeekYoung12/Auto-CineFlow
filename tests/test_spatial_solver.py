"""Tests for the spatial solver (180° axis logic and coordinate mapping)."""

import math
import pytest

from autocineflow.models import (
    Character,
    CharacterFacing,
    Position,
    ShotType,
)
from autocineflow.spatial_solver import (
    _cross_product_2d,
    _sign,
    assign_character_positions,
    classify_axis_side,
    compute_axis_vector,
    is_valid_shot,
    positions_to_controlnet,
)


class TestCrossProduct:
    def test_perpendicular_vectors(self):
        # (1,0) × (0,1) = 1
        assert _cross_product_2d((1.0, 0.0), (0.0, 1.0)) == 1.0

    def test_parallel_vectors(self):
        # (1,0) × (2,0) = 0
        assert _cross_product_2d((1.0, 0.0), (2.0, 0.0)) == 0.0

    def test_negative_cross(self):
        # (0,1) × (1,0) = -1
        assert _cross_product_2d((0.0, 1.0), (1.0, 0.0)) == -1.0


class TestSign:
    def test_positive(self):
        assert _sign(5.0) == 1

    def test_negative(self):
        assert _sign(-3.0) == -1

    def test_zero(self):
        assert _sign(0.0) == 0


class TestComputeAxisVector:
    def test_horizontal_axis(self):
        """Characters side by side: axis should point right (1, 0)."""
        v = compute_axis_vector((0.0, 0.5), (1.0, 0.5))
        assert abs(v[0] - 1.0) < 1e-9
        assert abs(v[1] - 0.0) < 1e-9

    def test_normalised_length(self):
        """Axis vector must always have length 1."""
        v = compute_axis_vector((0.1, 0.2), (0.8, 0.9))
        length = math.sqrt(v[0] ** 2 + v[1] ** 2)
        assert abs(length - 1.0) < 1e-9

    def test_same_position_raises(self):
        with pytest.raises(ValueError, match="same position"):
            compute_axis_vector((0.5, 0.5), (0.5, 0.5))


class TestIsValidShot:
    """REQ-02: Axis Guard Logic — is_valid_shot must prevent axis crossing."""

    def _axis(self):
        return compute_axis_vector((0.2, 0.5), (0.8, 0.5))

    def test_same_side_is_valid(self):
        axis = self._axis()
        # Both cameras above the horizontal axis (positive cross product)
        assert is_valid_shot((-1.0, 0.5), (-2.0, 0.5), axis) is True

    def test_crossing_is_invalid(self):
        """Camera going from above the axis to below must be invalid."""
        axis = self._axis()
        prev = (-1.0, 1.0)   # above the line
        curr = (-1.0, -1.0)  # below the line
        assert is_valid_shot(prev, curr, axis) is False

    def test_on_axis_is_neutral(self):
        """A camera placed exactly on the axis (cross product = 0) is always valid."""
        axis = self._axis()
        prev = (-1.0, 1.0)
        on_axis = (0.5, 0.5)  # on the axis itself
        assert is_valid_shot(prev, on_axis, axis) is True


class TestClassifyAxisSide:
    def test_both_sides_differ(self):
        """Two cameras on opposite sides of the axis must get different side labels."""
        axis = compute_axis_vector((0.2, 0.5), (0.8, 0.5))
        # Camera A: above the horizontal axis line (y < 0 in scene space relative to chars)
        side_a = classify_axis_side((0.5, -1.0), axis)
        # Camera B: below the horizontal axis line (y > 1 in scene space)
        side_b = classify_axis_side((0.5, 2.0), axis)
        assert side_a != side_b

    def test_same_side_consistent(self):
        """Two cameras on the same side must get the same label."""
        axis = compute_axis_vector((0.2, 0.5), (0.8, 0.5))
        side_a = classify_axis_side((0.3, -1.0), axis)
        side_b = classify_axis_side((0.7, -1.0), axis)
        assert side_a == side_b


class TestAssignCharacterPositions:
    def _make_chars(self):
        char_a = Character(
            char_id="CHAR_A",
            visual_anchor="man in black coat",
            pos=Position(x=0.25, y=0.5),
            facing=CharacterFacing.RIGHT,
        )
        char_b = Character(
            char_id="CHAR_B",
            visual_anchor="woman in red dress",
            pos=Position(x=0.75, y=0.5),
            facing=CharacterFacing.LEFT,
        )
        return char_a, char_b

    def test_mcu_two_shot_positions(self):
        """MCU two-shot: CHAR_A at (0.33, 0.4), CHAR_B at (0.66, 0.4)."""
        char_a, char_b = self._make_chars()
        a, b = assign_character_positions(char_a, char_b, ShotType.MCU)
        assert a.pos.x == 0.33
        assert a.pos.y == 0.4
        assert b.pos.x == 0.66
        assert b.pos.y == 0.4

    def test_mcu_facing_logic(self):
        """MCU: left character faces RIGHT, right character faces LEFT."""
        char_a, char_b = self._make_chars()
        a, b = assign_character_positions(char_a, char_b, ShotType.MCU)
        assert a.facing == CharacterFacing.RIGHT
        assert b.facing == CharacterFacing.LEFT

    def test_single_character_centred(self):
        """Single character is placed at (0.5, 0.4) facing CAMERA."""
        char_a, _ = self._make_chars()
        a, b = assign_character_positions(char_a, None, ShotType.MCU)
        assert a.pos.x == 0.5
        assert b is None

    def test_close_up_positions(self):
        """CLOSE_UP uses same positions as MCU (nose room rule)."""
        char_a, char_b = self._make_chars()
        a, b = assign_character_positions(char_a, char_b, ShotType.CLOSE_UP)
        assert a.pos.x == 0.33
        assert b.pos.x == 0.66

    def test_master_shot_spread(self):
        """MASTER_SHOT spreads characters wider than MCU."""
        char_a, char_b = self._make_chars()
        a_ms, b_ms = assign_character_positions(char_a, char_b, ShotType.MASTER_SHOT)
        a_mcu, b_mcu = assign_character_positions(char_a, char_b, ShotType.MCU)
        assert a_ms.pos.x < a_mcu.pos.x  # more to the left
        assert b_ms.pos.x > b_mcu.pos.x  # more to the right


class TestPositionsToControlNet:
    def test_output_format(self):
        chars = [
            Character(
                char_id="CHAR_A",
                visual_anchor="person A",
                pos=Position(x=0.33, y=0.4),
                facing=CharacterFacing.RIGHT,
            )
        ]
        coords = positions_to_controlnet(chars)
        assert len(coords) == 1
        assert coords[0]["char_id"] == "CHAR_A"
        assert coords[0]["x"] == 0.33
        assert coords[0]["y"] == 0.4

    def test_controlnet_range(self):
        """All coordinate values must be in [0, 1] for ControlNet compatibility."""
        chars = [
            Character(
                char_id="CHAR_A",
                visual_anchor="p",
                pos=Position(x=0.33, y=0.4),
                facing=CharacterFacing.RIGHT,
            ),
            Character(
                char_id="CHAR_B",
                visual_anchor="q",
                pos=Position(x=0.66, y=0.4),
                facing=CharacterFacing.LEFT,
            ),
        ]
        for coord in positions_to_controlnet(chars):
            assert 0.0 <= coord["x"] <= 1.0
            assert 0.0 <= coord["y"] <= 1.0
