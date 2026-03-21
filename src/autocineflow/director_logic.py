"""Director logic: shot selection, axis management, and emotion-driven framing.

REQ-02: Axis Guard Logic
REQ-03: Shot Template Library
REQ-04: Semantic-Parameter Mapping
"""

from __future__ import annotations

from .models import (
    AxisSide,
    CameraAngle,
    CameraAngleType,
    Character,
    EMOTION_MATRIX,
    FramingParams,
    LightingParams,
    MotionInstruction,
    SceneContext,
    ShotBlock,
    ShotType,
    SHOT_TEMPLATES,
)
from .spatial_solver import (
    assign_character_positions,
    classify_axis_side,
    compute_axis_vector,
    is_valid_shot,
)

# ---------------------------------------------------------------------------
# Default camera position constants (top-down view of the scene)
# ---------------------------------------------------------------------------

# Cameras are placed slightly above the midpoint between the two characters.
# These are *scene-space* coordinates (not normalised canvas); the
# spatial_solver works with these for axis validation.
_DEFAULT_CAM_LEFT: tuple[float, float] = (-1.0, 0.5)   # camera on the left half-plane
_DEFAULT_CAM_RIGHT: tuple[float, float] = (1.0, 0.5)   # camera on the right half-plane


# ---------------------------------------------------------------------------
# Emotion → shot type heuristics
# ---------------------------------------------------------------------------

_EMOTION_TO_SHOT: dict[str, ShotType] = {
    "angry":    ShotType.CLOSE_UP,
    "furious":  ShotType.CLOSE_UP,
    "sad":      ShotType.MCU,
    "happy":    ShotType.MEDIUM_SHOT,
    "joyful":   ShotType.MEDIUM_SHOT,
    "tense":    ShotType.MCU,
    "scared":   ShotType.CLOSE_UP,
    "calm":     ShotType.MEDIUM_SHOT,
    "romantic": ShotType.MCU,
    "neutral":  ShotType.MEDIUM_SHOT,
}


def _emotion_params(emotion: str) -> dict:
    """Return the emotion matrix entry for the given keyword (falls back to neutral)."""
    return EMOTION_MATRIX.get(emotion.lower(), EMOTION_MATRIX["neutral"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def choose_shot_type(emotion: str, shot_index: int, num_characters: int) -> ShotType:
    """Choose an appropriate shot type based on emotion, index, and character count.

    Rules:
    * Shot 0 (establishing): always MASTER_SHOT when there are 2 characters,
      so that spatial relationships are locked in first.
    * Subsequent shots: derive from emotion heuristic.
    * Single character defaults: MCU for emotional content, MEDIUM_SHOT otherwise.
    """
    if shot_index == 0 and num_characters >= 2:
        return ShotType.MASTER_SHOT
    return _EMOTION_TO_SHOT.get(emotion.lower(), ShotType.MEDIUM_SHOT)


def build_framing(
    shot_type: ShotType,
    primary_subjects: list[str],
) -> FramingParams:
    """Build a FramingParams object from a shot type and subject IDs."""
    template = SHOT_TEMPLATES[shot_type]
    return FramingParams(
        shot_type=shot_type,
        focal_length_mm=template.focal_length_mm,
        subjects=primary_subjects,
    )


def build_camera_angle(
    context: SceneContext,
    shot_type: ShotType,
    emotion: str,
) -> CameraAngle:
    """Build CameraAngle and enforce 180° axis consistency (REQ-02).

    The axis_side is established on the first shot of a scene and locked for
    all subsequent shots.  Any proposed camera that would cross the axis is
    rejected and replaced with the established side.
    """
    # Determine the desired camera position in scene space
    proposed_cam = _DEFAULT_CAM_LEFT  # default: always start on the left half-plane

    # Compute the axis vector if two characters are present
    if len(context.characters) >= 2:
        char_a = context.characters[0]
        char_b = context.characters[1]
        axis_vec = compute_axis_vector(
            (char_a.pos.x, char_a.pos.y),
            (char_b.pos.x, char_b.pos.y),
        )
        desired_side = classify_axis_side(proposed_cam, axis_vec)
    else:
        desired_side = AxisSide.LEFT

    # On the first shot: establish the axis side
    if context.axis_side is None:
        locked_side = desired_side
    else:
        locked_side = context.axis_side
        # Validate that the proposed camera doesn't cross the axis
        if len(context.characters) >= 2:
            char_a = context.characters[0]
            char_b = context.characters[1]
            axis_vec = compute_axis_vector(
                (char_a.pos.x, char_a.pos.y),
                (char_b.pos.x, char_b.pos.y),
            )
            prev_cam = (
                _DEFAULT_CAM_LEFT
                if context.axis_side == AxisSide.LEFT
                else _DEFAULT_CAM_RIGHT
            )
            if not is_valid_shot(prev_cam, proposed_cam, axis_vec):
                # Flip to the locked side instead
                proposed_cam = (
                    _DEFAULT_CAM_LEFT
                    if locked_side == AxisSide.LEFT
                    else _DEFAULT_CAM_RIGHT
                )

    # Emotion-driven angle adjustment
    angle_type = CameraAngleType.EYE_LEVEL
    tilt = 0.0
    if emotion in ("scared", "tense"):
        angle_type = CameraAngleType.LOW_ANGLE
        tilt = 10.0
    elif emotion in ("sad",):
        angle_type = CameraAngleType.HIGH_ANGLE
        tilt = -10.0

    return CameraAngle(
        angle_type=angle_type,
        axis_side=locked_side,
        tilt_degrees=tilt,
    )


def build_lighting(emotion: str) -> LightingParams:
    """Build LightingParams from the emotion matrix (REQ-04)."""
    params = _emotion_params(emotion)
    return LightingParams(
        contrast=float(params["contrast"]),
        saturation=float(params["saturation"]),
        mood=str(params["mood"]),
    )


def build_motion(emotion: str, shot_type: ShotType) -> MotionInstruction:
    """Build a MotionInstruction from emotion and shot type."""
    params = _emotion_params(emotion)
    intensity = float(params["motion"])

    # Camera move heuristics
    if shot_type == ShotType.CLOSE_UP and intensity > 0.5:
        motion_type = "DOLLY_IN"
    elif shot_type == ShotType.MASTER_SHOT:
        motion_type = "STATIC"
        intensity = 0.0
    elif intensity < 0.2:
        motion_type = "STATIC"
    elif intensity > 0.6:
        motion_type = "HANDHELD"
    else:
        motion_type = "DOLLY_IN"

    return MotionInstruction(motion_type=motion_type, intensity=intensity)


def update_axis_side(context: SceneContext, camera_angle: CameraAngle) -> SceneContext:
    """Return a new SceneContext with the axis_side locked from camera_angle."""
    if context.axis_side is None:
        return context.model_copy(update={"axis_side": camera_angle.axis_side})
    return context


def build_shot(
    context: SceneContext,
    shot_index: int,
    emotion: str | None = None,
) -> tuple[ShotBlock, SceneContext]:
    """Build a single ShotBlock and return the updated SceneContext.

    This is the main entry point for the Director filter.

    Args:
        context:     Current scene state.
        shot_index:  Zero-based shot number in the sequence.
        emotion:     Override emotion for this shot; falls back to context.detected_emotion.

    Returns:
        A (ShotBlock, updated_SceneContext) tuple.
    """
    emotion = (emotion or context.detected_emotion).lower()
    num_chars = len(context.characters)

    # 1. Choose shot type
    shot_type = choose_shot_type(emotion, shot_index, num_chars)

    # 2. Assign spatial positions to characters
    char_a = context.characters[0] if num_chars > 0 else None
    char_b = context.characters[1] if num_chars > 1 else None

    if char_a is not None:
        char_a, char_b = assign_character_positions(char_a, char_b, shot_type)
        updated_chars = [char_a] + ([char_b] if char_b else [])
    else:
        updated_chars = []

    # Update context with positioned characters
    context = context.model_copy(update={"characters": updated_chars})

    # 3. Build camera angle (enforces 180° rule)
    camera_angle = build_camera_angle(context, shot_type, emotion)

    # 4. Lock axis side in context
    context = update_axis_side(context, camera_angle)

    # 5. Build framing
    subject_ids = [c.char_id for c in updated_chars]
    framing = build_framing(shot_type, subject_ids)

    # 6. Build lighting & motion
    lighting = build_lighting(emotion)
    motion = build_motion(emotion, shot_type)

    shot_block = ShotBlock(
        shot_index=shot_index,
        framing=framing,
        camera_angle=camera_angle,
        lighting=lighting,
        motion_instruction=motion,
        characters=updated_chars,
    )

    # Append shot to context
    updated_shots = list(context.shot_blocks) + [shot_block]
    context = context.model_copy(update={"shot_blocks": updated_shots})

    return shot_block, context
