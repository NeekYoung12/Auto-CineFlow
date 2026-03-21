"""Spatial solver: 180° axis calculation and normalised coordinate mapping.

REQ-02: Axis Guard Logic
"""

from __future__ import annotations

import math
from typing import Tuple

from .models import (
    AxisSide,
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
# Vector helpers
# ---------------------------------------------------------------------------


def _cross_product_2d(v1: Vec2, v2: Vec2) -> float:
    """Return the scalar (z-component) of the 2-D cross product v1 × v2.

    Positive means v2 is counter-clockwise from v1; negative means clockwise.
    """
    return v1[0] * v2[1] - v1[1] * v2[0]


def _sign(value: float) -> int:
    """Return +1 for positive, -1 for negative, 0 for zero."""
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


# ---------------------------------------------------------------------------
# 180° Axis validation  (REQ-02)
# ---------------------------------------------------------------------------


def compute_axis_vector(char_a_pos: Vec2, char_b_pos: Vec2) -> Vec2:
    """Return the normalised axis vector from character A to character B.

    The 180° rule (line of action) runs along this vector.  All cameras
    must stay on the same half-plane with respect to it.
    """
    dx = char_b_pos[0] - char_a_pos[0]
    dy = char_b_pos[1] - char_a_pos[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length == 0:
        raise ValueError("Both characters occupy the same position; cannot compute axis vector.")
    return (dx / length, dy / length)


def is_valid_shot(
    previous_cam_pos: Vec2,
    current_cam_pos: Vec2,
    axis_vector: Vec2,
) -> bool:
    """Check that a new camera position does NOT cross the 180° axis.

    A valid shot keeps the camera on the same half-plane as the previous shot.
    Uses the cross-product sign test described in the design document.

    Args:
        previous_cam_pos: (x, y) of the previous camera position.
        current_cam_pos:  (x, y) of the proposed new camera position.
        axis_vector:      Normalised axis vector (from char A toward char B).

    Returns:
        True if the cut is valid (no axis crossing), False otherwise.
    """
    prev_sign = _sign(_cross_product_2d(previous_cam_pos, axis_vector))
    curr_sign = _sign(_cross_product_2d(current_cam_pos, axis_vector))
    # If either sign is 0 the camera is ON the axis, which is always acceptable
    # (it is a special case such as a "neutral" cut).
    if prev_sign == 0 or curr_sign == 0:
        return True
    return prev_sign == curr_sign


def classify_axis_side(cam_pos: Vec2, axis_vector: Vec2) -> AxisSide:
    """Return which side of the 180° axis the camera position occupies."""
    cp = _cross_product_2d(cam_pos, axis_vector)
    return AxisSide.LEFT if cp >= 0 else AxisSide.RIGHT


# ---------------------------------------------------------------------------
# Normalised coordinate mapping  (REQ-02, acceptance criterion)
# ---------------------------------------------------------------------------

# Canonical face-centre positions in the normalised (0–1) canvas.
# The left-side subject uses the left thirds line; the right-side uses the right.
_MCU_FACE_LEFT: Position = Position(x=0.33, y=0.4)
_MCU_FACE_RIGHT: Position = Position(x=0.66, y=0.4)


def assign_character_positions(
    char_a: Character,
    char_b: Character | None,
    shot_type: ShotType,
) -> tuple[Character, Character | None]:
    """Assign normalised canvas positions to characters based on shot type.

    Rules:
    * In a two-shot, CHAR_A is placed on the left, CHAR_B on the right.
    * MCU / CLOSE_UP positions follow the (0.33, 0.4) / (0.66, 0.4) rule.
    * MASTER_SHOT and MEDIUM_SHOT spread characters further apart.
    * Facing direction is set to enforce gaze logic (left→RIGHT, right→LEFT).

    Returns updated copies of the characters (originals are not mutated).
    """
    if char_b is None:
        # Single character – centre of frame
        updated_a = char_a.model_copy(
            update={"pos": Position(x=0.5, y=0.4), "facing": CharacterFacing.CAMERA}
        )
        return updated_a, None

    # Two-character shot — A on left, B on right
    if shot_type in (ShotType.MCU, ShotType.CLOSE_UP):
        pos_a = _MCU_FACE_LEFT
        pos_b = _MCU_FACE_RIGHT
    elif shot_type == ShotType.OVER_SHOULDER:
        # Foreground character (partial, blurred) on opposite side
        pos_a = Position(x=0.25, y=0.5)
        pos_b = Position(x=0.65, y=0.38)
    else:
        # MASTER_SHOT / MEDIUM_SHOT
        pos_a = Position(x=0.2, y=0.5)
        pos_b = Position(x=0.8, y=0.5)

    updated_a = char_a.model_copy(
        update={"pos": pos_a, "facing": CharacterFacing.RIGHT}
    )
    updated_b = char_b.model_copy(
        update={"pos": pos_b, "facing": CharacterFacing.LEFT}
    )
    return updated_a, updated_b


def positions_to_controlnet(characters: list[Character]) -> list[dict]:
    """Convert normalised character positions to ControlNet-compatible coordinate dicts.

    The acceptance criterion requires that `pos` coordinates are directly mappable
    to ControlNet coordinate points.  ControlNet expects values in [0, 1] with
    x=0 being the left edge and y=0 being the top edge — which matches our canvas.

    Returns a list of dicts with keys: char_id, x, y.
    """
    return [
        {"char_id": c.char_id, "x": c.pos.x, "y": c.pos.y}
        for c in characters
    ]
