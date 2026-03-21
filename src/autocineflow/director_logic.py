"""Director logic: shot planning, axis management, and emotion-driven framing.

REQ-02: Axis Guard Logic
REQ-03: Shot Template Library
REQ-04: Semantic-Parameter Mapping
"""

from __future__ import annotations

from .models import (
    AxisSide,
    BeatType,
    CameraAngle,
    CameraAngleType,
    Character,
    CompositionHint,
    EMOTION_MATRIX,
    FramingParams,
    LightingParams,
    MotionInstruction,
    SceneBeat,
    SceneContext,
    ShotBlock,
    ShotType,
    SHOT_TEMPLATES,
)
from .spatial_solver import (
    assign_character_positions,
    camera_position_for_shot,
    compute_axis_origin,
    compute_axis_vector,
    is_valid_shot,
    positions_to_controlnet,
    relative_camera_to_axis_origin,
)


_EMOTION_TO_SHOT: dict[str, ShotType] = {
    "angry": ShotType.CLOSE_UP,
    "furious": ShotType.CLOSE_UP,
    "sad": ShotType.MCU,
    "happy": ShotType.MEDIUM_SHOT,
    "joyful": ShotType.MEDIUM_SHOT,
    "tense": ShotType.MCU,
    "scared": ShotType.CLOSE_UP,
    "calm": ShotType.MEDIUM_SHOT,
    "romantic": ShotType.MCU,
    "neutral": ShotType.MEDIUM_SHOT,
}

_HIGH_EMOTION = {"angry", "furious", "scared"}
_INTIMATE_EMOTION = {"sad", "tense", "romantic"}


def _emotion_params(emotion: str) -> dict:
    """Return the emotion matrix entry for the given keyword (falls back to neutral)."""

    return EMOTION_MATRIX.get(emotion.lower(), EMOTION_MATRIX["neutral"])


def choose_shot_type(emotion: str, shot_index: int, num_characters: int) -> ShotType:
    """Choose an appropriate shot type based on emotion, index, and character count."""

    if shot_index == 0 and num_characters >= 2:
        return ShotType.MASTER_SHOT
    return _EMOTION_TO_SHOT.get(emotion.lower(), ShotType.MEDIUM_SHOT)


def plan_scene_beats(context: SceneContext, num_shots: int) -> list[SceneBeat]:
    """Create a narrative beat plan for the requested number of shots."""

    emotion = context.detected_emotion.lower()
    num_chars = len(context.characters)
    primary_focus = context.primary_focus_char_id or (context.characters[0].char_id if context.characters else None)
    secondary_focus = (
        context.characters[1].char_id
        if len(context.characters) > 1
        else primary_focus
    )

    beats: list[SceneBeat] = []
    for shot_index in range(num_shots):
        if shot_index == 0 and num_chars >= 2:
            beats.append(
                SceneBeat(
                    beat_index=shot_index,
                    beat_type=BeatType.ESTABLISH,
                    emotion=emotion,
                    intensity=0.25,
                    purpose="Establish geography and lock screen direction.",
                    focus_char_id=primary_focus,
                    shot_type_hint=ShotType.MASTER_SHOT,
                )
            )
            continue

        if shot_index == 1 and num_chars >= 2:
            beats.append(
                SceneBeat(
                    beat_index=shot_index,
                    beat_type=BeatType.RELATION,
                    emotion=emotion,
                    intensity=0.4,
                    purpose="Clarify eyelines and the relationship axis.",
                    focus_char_id=secondary_focus,
                    shot_type_hint=ShotType.OVER_SHOULDER,
                )
            )
            continue

        if emotion in _HIGH_EMOTION and shot_index == num_shots - 2 and num_shots >= 3:
            beats.append(
                SceneBeat(
                    beat_index=shot_index,
                    beat_type=BeatType.ESCALATION,
                    emotion=emotion,
                    intensity=0.9,
                    purpose="Push into the emotional break with a tighter framing.",
                    focus_char_id=primary_focus,
                    shot_type_hint=ShotType.CLOSE_UP,
                )
            )
            continue

        if emotion in _HIGH_EMOTION and shot_index == num_shots - 1 and num_chars >= 2:
            beats.append(
                SceneBeat(
                    beat_index=shot_index,
                    beat_type=BeatType.REACTION,
                    emotion=emotion,
                    intensity=0.75,
                    purpose="Hold the reaction on the other performer without breaking the axis.",
                    focus_char_id=secondary_focus,
                    shot_type_hint=ShotType.CLOSE_UP,
                )
            )
            continue

        if emotion in _INTIMATE_EMOTION and shot_index >= max(2, num_shots - 2):
            beats.append(
                SceneBeat(
                    beat_index=shot_index,
                    beat_type=BeatType.RESOLUTION if shot_index == num_shots - 1 else BeatType.BUILD,
                    emotion=emotion,
                    intensity=0.6 if shot_index == num_shots - 1 else 0.5,
                    purpose="Compress the scene around the emotional exchange.",
                    focus_char_id=primary_focus if shot_index % 2 == 0 else secondary_focus,
                    shot_type_hint=ShotType.MCU,
                )
            )
            continue

        beats.append(
            SceneBeat(
                beat_index=shot_index,
                beat_type=BeatType.BUILD if shot_index < num_shots - 1 else BeatType.RESOLUTION,
                emotion=emotion,
                intensity=0.45,
                purpose="Advance the conversational beat while preserving continuity.",
                focus_char_id=primary_focus if shot_index % 2 == 0 else secondary_focus,
                shot_type_hint=choose_shot_type(emotion, shot_index, num_chars),
            )
        )

    return beats


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
    beat_type: BeatType | None = None,
) -> CameraAngle:
    """Build CameraAngle and enforce 180° axis consistency (REQ-02)."""

    locked_side = context.axis_side or AxisSide.LEFT

    angle_type = CameraAngleType.EYE_LEVEL
    tilt = 0.0
    if emotion in ("scared", "tense"):
        angle_type = CameraAngleType.LOW_ANGLE
        tilt = 10.0
    elif emotion in ("sad",):
        angle_type = CameraAngleType.HIGH_ANGLE
        tilt = -10.0
    elif emotion == "furious" and shot_type == ShotType.CLOSE_UP:
        angle_type = CameraAngleType.DUTCH_ANGLE
        tilt = 18.0

    if beat_type == BeatType.ESTABLISH:
        angle_type = CameraAngleType.EYE_LEVEL
        tilt = 0.0

    return CameraAngle(
        angle_type=angle_type,
        axis_side=locked_side,
        tilt_degrees=tilt,
    )


def build_lighting(emotion: str, beat_intensity: float | None = None) -> LightingParams:
    """Build LightingParams from the emotion matrix (REQ-04)."""

    params = _emotion_params(emotion)
    contrast = float(params["contrast"])
    saturation = float(params["saturation"])

    if beat_intensity is not None:
        contrast = max(0.0, min(1.0, contrast + (beat_intensity - 0.5) * 0.2))
        saturation = max(0.0, min(1.0, saturation + (beat_intensity - 0.5) * 0.1))

    return LightingParams(
        contrast=contrast,
        saturation=saturation,
        mood=str(params["mood"]),
    )


def build_motion(
    emotion: str,
    shot_type: ShotType,
    beat_type: BeatType | None = None,
    intensity_override: float | None = None,
) -> MotionInstruction:
    """Build a MotionInstruction from emotion and shot type."""

    params = _emotion_params(emotion)
    intensity = intensity_override if intensity_override is not None else float(params["motion"])

    if beat_type == BeatType.ESTABLISH or shot_type == ShotType.MASTER_SHOT:
        motion_type = "STATIC"
        intensity = 0.0
    elif beat_type == BeatType.RELATION:
        motion_type = "PAN_RIGHT"
        intensity = max(intensity, 0.25)
    elif shot_type == ShotType.CLOSE_UP and intensity > 0.5:
        motion_type = "DOLLY_IN"
    elif intensity < 0.2:
        motion_type = "STATIC"
    elif intensity > 0.75:
        motion_type = "HANDHELD"
    else:
        motion_type = "DOLLY_IN"

    return MotionInstruction(motion_type=motion_type, intensity=float(intensity))


def update_axis_side(context: SceneContext, camera_angle: CameraAngle) -> SceneContext:
    """Return a new SceneContext with the axis_side locked from camera_angle."""

    if context.axis_side is None:
        return context.model_copy(update={"axis_side": camera_angle.axis_side})
    return context


def _determine_subject_ids(
    characters: list[Character],
    shot_type: ShotType,
    focus_char_id: str | None,
) -> list[str]:
    """Pick the primary shot subjects."""

    if shot_type in {ShotType.CLOSE_UP, ShotType.MCU, ShotType.OVER_SHOULDER} and focus_char_id:
        return [focus_char_id]
    return [character.char_id for character in characters]


def _build_composition(
    characters: list[Character],
    shot_type: ShotType,
    focus_char_id: str | None,
) -> CompositionHint | None:
    """Create composition notes that downstream renderers can consume."""

    if not characters:
        return None

    focus_character = None
    if focus_char_id:
        focus_character = next((char for char in characters if char.char_id == focus_char_id), None)
    if focus_character is None:
        focus_character = characters[0]

    staging = {
        ShotType.MASTER_SHOT: "Two-shot staging with clear left-right separation.",
        ShotType.MEDIUM_SHOT: "Waist-up staging with readable body language.",
        ShotType.MCU: "Faces prioritised while preserving eyeline continuity.",
        ShotType.CLOSE_UP: "Emotion-first composition with compressed background.",
        ShotType.OVER_SHOULDER: "Foreground shoulder frames the focused character.",
    }[shot_type]
    nose_room = {
        "LEFT": "Leave screen space toward the right for the eyeline.",
        "RIGHT": "Leave screen space toward the left for the eyeline.",
        "CAMERA": "Keep the face centred with balanced headroom.",
        "AWAY": "Preserve negative space in the direction of travel.",
    }[focus_character.facing.value]

    return CompositionHint(
        focus_point=focus_character.pos,
        nose_room=nose_room,
        staging=staging,
    )


def _camera_position_for_locked_axis(
    context: SceneContext,
    shot_type: ShotType,
    focus_char_id: str | None,
) -> tuple[SceneContext, object | None]:
    """Compute camera placement while preserving the locked axis side."""

    if len(context.characters) < 2:
        return context, None

    char_a = context.characters[0]
    char_b = context.characters[1]
    axis_vector = compute_axis_vector((char_a.pos.x, char_a.pos.y), (char_b.pos.x, char_b.pos.y))
    axis_origin = compute_axis_origin((char_a.pos.x, char_a.pos.y), (char_b.pos.x, char_b.pos.y))
    locked_side = context.axis_side or AxisSide.LEFT

    camera_position = camera_position_for_shot(
        (char_a.pos.x, char_a.pos.y),
        (char_b.pos.x, char_b.pos.y),
        shot_type,
        locked_side,
        focus_char_id=focus_char_id,
    )

    if context.shot_blocks and context.shot_blocks[-1].camera_position is not None:
        previous_relative = relative_camera_to_axis_origin(context.shot_blocks[-1].camera_position, axis_origin)
        current_relative = relative_camera_to_axis_origin(camera_position, axis_origin)
        if not is_valid_shot(previous_relative, current_relative, axis_vector):
            camera_position = camera_position_for_shot(
                (char_a.pos.x, char_a.pos.y),
                (char_b.pos.x, char_b.pos.y),
                shot_type,
                context.axis_side or AxisSide.LEFT,
                focus_char_id=focus_char_id,
            )

    return context, camera_position


def build_shot(
    context: SceneContext,
    shot_index: int,
    emotion: str | None = None,
) -> tuple[ShotBlock, SceneContext]:
    """Build a single ShotBlock and return the updated SceneContext."""

    beat = context.beats[shot_index] if shot_index < len(context.beats) else None
    active_emotion = (emotion or (beat.emotion if beat else context.detected_emotion)).lower()
    num_chars = len(context.characters)

    shot_type = (
        beat.shot_type_hint
        if beat and beat.shot_type_hint is not None
        else choose_shot_type(active_emotion, shot_index, num_chars)
    )
    focus_char_id = beat.focus_char_id if beat else context.primary_focus_char_id

    char_a = context.characters[0] if num_chars > 0 else None
    char_b = context.characters[1] if num_chars > 1 else None

    if char_a is not None:
        use_focus_layout = focus_char_id if shot_type in {ShotType.MCU, ShotType.CLOSE_UP, ShotType.OVER_SHOULDER} else None
        char_a, char_b = assign_character_positions(char_a, char_b, shot_type, focus_char_id=use_focus_layout)
        updated_chars = [char_a] + ([char_b] if char_b else [])
    else:
        updated_chars = []

    context = context.model_copy(update={"characters": updated_chars})

    camera_angle = build_camera_angle(
        context=context,
        shot_type=shot_type,
        emotion=active_emotion,
        beat_type=beat.beat_type if beat else None,
    )
    context = update_axis_side(context, camera_angle)
    context, camera_position = _camera_position_for_locked_axis(context, shot_type, focus_char_id)

    subject_ids = _determine_subject_ids(updated_chars, shot_type, focus_char_id)
    framing = build_framing(shot_type, subject_ids)
    lighting = build_lighting(active_emotion, beat.intensity if beat else None)
    motion = build_motion(
        active_emotion,
        shot_type,
        beat_type=beat.beat_type if beat else None,
        intensity_override=beat.intensity if beat else None,
    )
    composition = _build_composition(updated_chars, shot_type, focus_char_id)

    shot_block = ShotBlock(
        shot_index=shot_index,
        framing=framing,
        camera_angle=camera_angle,
        lighting=lighting,
        motion_instruction=motion,
        characters=updated_chars,
        beat_type=beat.beat_type if beat else None,
        story_purpose=beat.purpose if beat else "",
        composition=composition,
        camera_position=camera_position,
        controlnet_points=positions_to_controlnet(updated_chars),
        scene_location=context.scene_location,
        scene_tags=list(context.scene_tags),
    )

    updated_shots = list(context.shot_blocks) + [shot_block]
    context = context.model_copy(update={"shot_blocks": updated_shots})

    return shot_block, context
