"""Prompt builder: assembles Stable Diffusion prompts from ShotBlock data.

Acceptance criterion: the generated prompt must be usable directly in an
open-source SD workflow without manual vocabulary modification.
"""

from __future__ import annotations

from .models import (
    Character,
    CharacterFacing,
    CameraAngleType,
    LightingParams,
    MotionInstruction,
    ShotBlock,
    ShotType,
)

# ---------------------------------------------------------------------------
# Shot type → SD framing keywords
# ---------------------------------------------------------------------------

_SHOT_TYPE_TOKENS: dict[ShotType, str] = {
    ShotType.MASTER_SHOT:   "extreme wide shot, establishing shot, full environment visible",
    ShotType.MEDIUM_SHOT:   "medium shot, waist up, full body context",
    ShotType.MCU:           "medium close-up, chest and shoulders, face clearly visible",
    ShotType.CLOSE_UP:      "close-up shot, face fills frame, extreme detail",
    ShotType.OVER_SHOULDER: "over the shoulder shot, two people, depth of field",
}

# Camera angle → SD tokens
_ANGLE_TOKENS: dict[CameraAngleType, str] = {
    CameraAngleType.EYE_LEVEL: "eye level angle",
    CameraAngleType.HIGH_ANGLE: "high angle shot, looking down",
    CameraAngleType.LOW_ANGLE: "low angle shot, looking up, dramatic",
    CameraAngleType.DUTCH_ANGLE: "dutch angle, tilted camera",
}

# Mood → lighting / colour grade tokens
_MOOD_TOKENS: dict[str, str] = {
    "intense":     "harsh lighting, high contrast, deep shadows",
    "explosive":   "extreme contrast, blown-out highlights, dramatic lighting",
    "melancholic": "soft diffused light, desaturated, blue tones",
    "warm":        "warm golden lighting, soft shadows, vibrant colours",
    "euphoric":    "bright sunlight, vivid colours, joyful atmosphere",
    "suspenseful": "dim lighting, teal and orange colour grade, tension",
    "fearful":     "low key lighting, cold blue tones, long shadows",
    "serene":      "soft natural light, pastel palette, calm atmosphere",
    "intimate":    "warm candlelight, bokeh background, romantic lighting",
    "neutral":     "balanced natural lighting, neutral colour grade",
}

# Motion → SD cinematography tokens
_MOTION_TOKENS: dict[str, str] = {
    "STATIC":     "static camera, steady",
    "DOLLY_IN":   "dolly in, camera approaching subject",
    "DOLLY_OUT":  "dolly out, camera pulling back",
    "PAN_LEFT":   "panning left",
    "PAN_RIGHT":  "panning right",
    "HANDHELD":   "handheld camera, slight motion blur, documentary style",
}

# Negative prompt base tokens
_NEGATIVE_BASE = (
    "blurry, deformed, ugly, bad anatomy, extra limbs, missing limbs, "
    "watermark, text, logo, signature, cropped, bad proportions"
)


def _character_tokens(character: Character) -> str:
    """Return SD tokens for a single character."""
    facing_token = {
        CharacterFacing.LEFT:   "facing left",
        CharacterFacing.RIGHT:  "facing right",
        CharacterFacing.CAMERA: "facing camera",
        CharacterFacing.AWAY:   "facing away from camera",
    }.get(character.facing, "")

    return f"{character.visual_anchor}, {facing_token}"


def _lighting_tokens(lighting: LightingParams) -> str:
    """Derive SD lighting tokens from LightingParams."""
    mood_str = _MOOD_TOKENS.get(lighting.mood, "natural lighting")
    contrast_adj = (
        "extreme contrast" if lighting.contrast > 0.75
        else "high contrast" if lighting.contrast > 0.6
        else "low contrast" if lighting.contrast < 0.35
        else "balanced contrast"
    )
    saturation_adj = (
        "vivid colours" if lighting.saturation > 0.8
        else "desaturated" if lighting.saturation < 0.3
        else ""
    )
    parts = [mood_str, contrast_adj]
    if saturation_adj:
        parts.append(saturation_adj)
    return ", ".join(parts)


def build_sd_prompt(shot_block: ShotBlock) -> str:
    """Assemble a complete Stable Diffusion prompt from a ShotBlock.

    The prompt is structured as:
      [framing] + [characters] + [camera angle] + [lighting] + [motion] + [quality tokens]

    Returns the positive prompt string.
    """
    parts: list[str] = []

    # 1. Shot framing
    parts.append(_SHOT_TYPE_TOKENS.get(shot_block.framing.shot_type, ""))

    # 2. Characters (visual anchors + facing direction)
    for char in shot_block.characters:
        parts.append(_character_tokens(char))

    # 3. Camera angle
    parts.append(_ANGLE_TOKENS.get(shot_block.camera_angle.angle_type, "eye level"))

    # 4. Focal length (cinematic descriptor)
    fl = shot_block.framing.focal_length_mm
    parts.append(f"{fl}mm lens, cinematic")

    # 5. Lighting / colour grade
    parts.append(_lighting_tokens(shot_block.lighting))

    # 6. Motion
    motion_key = shot_block.motion_instruction.motion_type
    parts.append(_MOTION_TOKENS.get(motion_key, "static camera"))

    # 7. Quality booster tokens
    parts.append(
        "cinematic photography, film grain, professional cinematography, "
        "8k resolution, highly detailed"
    )

    prompt = ", ".join(p for p in parts if p)
    return prompt


def build_negative_prompt(shot_block: ShotBlock) -> str:
    """Build a standard negative prompt for the shot."""
    extras: list[str] = []
    if shot_block.framing.shot_type == ShotType.CLOSE_UP:
        extras.append("full body, wide shot")
    elif shot_block.framing.shot_type == ShotType.MASTER_SHOT:
        extras.append("close-up, portrait")
    return _NEGATIVE_BASE + (", " + ", ".join(extras) if extras else "")


def attach_prompts(shot_block: ShotBlock) -> ShotBlock:
    """Return a copy of shot_block with sd_prompt and negative_prompt populated."""
    sd = build_sd_prompt(shot_block)
    neg = build_negative_prompt(shot_block)
    return shot_block.model_copy(update={"sd_prompt": sd, "negative_prompt": neg})
