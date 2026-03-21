"""Spatial solver: 180° axis calculation and normalised coordinate mapping.

REQ-02: Axis Guard Logic
"""

from __future__ import annotations

import math
from typing import Tuple

from .models import (
    AxisSide,
    CameraPlacement,
    Character,
    CharacterFacing,
    Position,
    ShotType,
)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Vec2 = Tuple[float, float]


# ---------------------------------------------------------------------------
# Canonical layout presets
# ---------------------------------------------------------------------------

_MCU_FACE_LEFT = Position(x=0.33, y=0.4)
_MCU_FACE_RIGHT = Position(x=0.66, y=0.4)

_SHOT_DISTANCES: dict[ShotType, float] = {
    ShotType.MASTER_SHOT: 2.4,
    ShotType.MEDIUM_SHOT: 1.8,
    ShotType.MCU: 1.2,
    ShotType.CLOSE_UP: 0.85,
    ShotType.OVER_SHOULDER: 1.1,
}


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------


def _cross_product_2d(v1: Vec2, v2: Vec2) -> float:
    """Return the scalar (z-component) of the 2-D cross product v1 x v2."""

    return v1[0] * v2[1] - v1[1] * v2[0]


def _sign(value: float) -> int:
    """Return +1 for positive, -1 for negative, 0 for zero."""

    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _subtract(a: Vec2, b: Vec2) -> Vec2:
    """Return a - b."""

    return (a[0] - b[0], a[1] - b[1])


def _add(a: Vec2, b: Vec2) -> Vec2:
    """Return a + b."""

    return (a[0] + b[0], a[1] + b[1])


def _scale(vec: Vec2, scalar: float) -> Vec2:
    """Return vec * scalar."""

    return (vec[0] * scalar, vec[1] * scalar)


def _midpoint(a: Vec2, b: Vec2) -> Vec2:
    """Return the midpoint between two vectors."""

    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _perpendicular(axis_vector: Vec2) -> Vec2:
    """Return a unit perpendicular vector for the given axis vector."""

    return (-axis_vector[1], axis_vector[0])


# ---------------------------------------------------------------------------
# 180° Axis validation  (REQ-02)
# ---------------------------------------------------------------------------


def compute_axis_vector(char_a_pos: Vec2, char_b_pos: Vec2) -> Vec2:
    """Return the normalised axis vector from character A to character B."""

    dx = char_b_pos[0] - char_a_pos[0]
    dy = char_b_pos[1] - char_a_pos[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length == 0:
        raise ValueError("Both characters occupy the same position; cannot compute axis vector.")
    return (dx / length, dy / length)


def compute_axis_origin(char_a_pos: Vec2, char_b_pos: Vec2) -> Vec2:
    """Return the midpoint that anchors the line of action."""

    return _midpoint(char_a_pos, char_b_pos)


def is_valid_shot(
    previous_cam_pos: Vec2,
    current_cam_pos: Vec2,
    axis_vector: Vec2,
) -> bool:
    """Check that a new camera position does NOT cross the 180° axis."""

    prev_sign = _sign(_cross_product_2d(previous_cam_pos, axis_vector))
    curr_sign = _sign(_cross_product_2d(current_cam_pos, axis_vector))
    if prev_sign == 0 or curr_sign == 0:
        return True
    return prev_sign == curr_sign


def classify_axis_side(cam_pos: Vec2, axis_vector: Vec2) -> AxisSide:
    """Return which side of the 180° axis the camera position occupies."""

    cp = _cross_product_2d(cam_pos, axis_vector)
    return AxisSide.LEFT if cp >= 0 else AxisSide.RIGHT


def camera_position_for_shot(
    char_a_pos: Vec2,
    char_b_pos: Vec2,
    shot_type: ShotType,
    axis_side: AxisSide,
    focus_char_id: str | None = None,
) -> CameraPlacement:
    """Compute a scene-space camera position for the shot.

    The camera is placed on a perpendicular vector from the line of action,
    with small axial offsets to favour the focused character for singles.
    """

    axis_vector = compute_axis_vector(char_a_pos, char_b_pos)
    origin = compute_axis_origin(char_a_pos, char_b_pos)
    perpendicular = _perpendicular(axis_vector)
    side_sign = 1.0 if axis_side == AxisSide.LEFT else -1.0
    distance = _SHOT_DISTANCES[shot_type]

    base = _add(origin, _scale(perpendicular, side_sign * distance))

    axial_offset = 0.0
    if focus_char_id == "CHAR_A":
        axial_offset = -0.25
    elif focus_char_id == "CHAR_B":
        axial_offset = 0.25

    positioned = _add(base, _scale(axis_vector, axial_offset))
    return CameraPlacement(x=positioned[0], y=positioned[1], distance=distance)


# ---------------------------------------------------------------------------
# Normalised coordinate mapping  (REQ-02, acceptance criterion)
# ---------------------------------------------------------------------------


def assign_character_positions(
    char_a: Character,
    char_b: Character | None,
    shot_type: ShotType,
    focus_char_id: str | None = None,
) -> tuple[Character, Character | None]:
    """Assign normalised canvas positions to characters based on shot type."""

    if char_b is None:
        updated_a = char_a.model_copy(
            update={"pos": Position(x=0.5, y=0.4), "facing": CharacterFacing.CAMERA}
        )
        return updated_a, None

    if focus_char_id is None:
        if shot_type in (ShotType.MCU, ShotType.CLOSE_UP):
            pos_a = _MCU_FACE_LEFT
            pos_b = _MCU_FACE_RIGHT
        elif shot_type == ShotType.OVER_SHOULDER:
            pos_a = Position(x=0.25, y=0.5)
            pos_b = Position(x=0.65, y=0.38)
        else:
            pos_a = Position(x=0.2, y=0.5)
            pos_b = Position(x=0.8, y=0.5)
    elif shot_type == ShotType.OVER_SHOULDER:
        if focus_char_id == char_a.char_id:
            pos_a = Position(x=0.35, y=0.38)
            pos_b = Position(x=0.8, y=0.56)
        else:
            pos_a = Position(x=0.2, y=0.56)
            pos_b = Position(x=0.62, y=0.38)
    elif shot_type in (ShotType.MCU, ShotType.CLOSE_UP):
        if focus_char_id == char_a.char_id:
            pos_a = Position(x=0.38, y=0.4)
            pos_b = Position(x=0.84, y=0.43)
        else:
            pos_a = Position(x=0.16, y=0.43)
            pos_b = Position(x=0.62, y=0.4)
    else:
        if focus_char_id == char_a.char_id:
            pos_a = Position(x=0.24, y=0.48)
            pos_b = Position(x=0.82, y=0.52)
        else:
            pos_a = Position(x=0.18, y=0.52)
            pos_b = Position(x=0.76, y=0.48)

    updated_a = char_a.model_copy(update={"pos": pos_a, "facing": CharacterFacing.RIGHT})
    updated_b = char_b.model_copy(update={"pos": pos_b, "facing": CharacterFacing.LEFT})
    return updated_a, updated_b


def positions_to_controlnet(characters: list[Character]) -> list[dict]:
    """Convert normalised character positions to ControlNet-compatible coordinates."""

    return [{"char_id": c.char_id, "x": c.pos.x, "y": c.pos.y} for c in characters]


def relative_camera_to_axis_origin(
    camera_position: CameraPlacement,
    axis_origin: Vec2,
) -> Vec2:
    """Return a camera vector relative to the axis origin for sign checks."""

    return _subtract((camera_position.x, camera_position.y), axis_origin)
