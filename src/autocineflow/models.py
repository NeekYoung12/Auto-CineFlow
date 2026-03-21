"""Pydantic data models for Auto-CineFlow.

Defines the structured output schema (ShotBlock) and supporting types
for the cinematic storyboard parameter generator.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class CharacterFacing(str, Enum):
    """Direction a character is facing within the frame."""

    LEFT = "LEFT"
    RIGHT = "RIGHT"
    CAMERA = "CAMERA"
    AWAY = "AWAY"


class ShotType(str, Enum):
    """Standard cinematographic shot types (framing sizes)."""

    MASTER_SHOT = "MASTER_SHOT"
    MEDIUM_SHOT = "MEDIUM_SHOT"
    MCU = "MCU"
    CLOSE_UP = "CLOSE_UP"
    OVER_SHOULDER = "OVER_SHOULDER"


class CameraAngleType(str, Enum):
    """Vertical camera angle relative to subject."""

    EYE_LEVEL = "EYE_LEVEL"
    HIGH_ANGLE = "HIGH_ANGLE"
    LOW_ANGLE = "LOW_ANGLE"
    DUTCH_ANGLE = "DUTCH_ANGLE"


class AxisSide(str, Enum):
    """Which side of the 180° axis the camera is on."""

    LEFT = "LEFT"
    RIGHT = "RIGHT"


class BeatType(str, Enum):
    """Narrative beat labels used to plan a shot sequence."""

    ESTABLISH = "ESTABLISH"
    RELATION = "RELATION"
    BUILD = "BUILD"
    ESCALATION = "ESCALATION"
    REACTION = "REACTION"
    RESOLUTION = "RESOLUTION"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class Position(BaseModel):
    """Normalised 2-D canvas position (range 0.0-1.0 on each axis)."""

    x: float = Field(..., ge=0.0, le=1.0, description="Horizontal position (0=left, 1=right)")
    y: float = Field(..., ge=0.0, le=1.0, description="Vertical position (0=top, 1=bottom)")


class DialogueLine(BaseModel):
    """Single dialogue fragment extracted from the scene description."""

    text: str = Field(..., description="Dialogue content without surrounding quotation marks")
    speaker_hint: Optional[str] = Field(
        default=None,
        description="Best-effort speaker hint when the analyser can infer one",
    )


class Character(BaseModel):
    """Character state including visual anchor and canvas position.

    REQ-01: Character State Machine.
    """

    char_id: str = Field(..., description="Unique identifier, e.g. 'CHAR_A' or 'CHAR_B'")
    visual_anchor: str = Field(
        ...,
        description=(
            "Immutable visual description used as the SD prompt anchor "
            "(hair, clothing, distinguishing features). Must remain 100% identical "
            "across all shots in a sequence."
        ),
    )
    pos: Position = Field(..., description="Normalised canvas position of the character")
    facing: CharacterFacing = Field(
        default=CharacterFacing.CAMERA,
        description="Direction the character is facing in the current shot",
    )

    @model_validator(mode="after")
    def validate_facing_vs_position(self) -> "Character":
        """Enforce gaze logic: a character on the left should face RIGHT, and vice versa."""
        if self.pos.x < 0.5 and self.facing == CharacterFacing.LEFT:
            raise ValueError(
                f"Character {self.char_id} is positioned on the left (x={self.pos.x}) "
                "but is facing LEFT, which violates screen-direction logic."
            )
        if self.pos.x > 0.5 and self.facing == CharacterFacing.RIGHT:
            raise ValueError(
                f"Character {self.char_id} is positioned on the right (x={self.pos.x}) "
                "but is facing RIGHT, which violates screen-direction logic."
            )
        return self


class ShotTemplate(BaseModel):
    """Shot template mapping a shot type to its canonical focal length.

    REQ-03: Shot Template Library.
    """

    shot_type: ShotType
    focal_length_mm: int = Field(..., description="Lens focal length in millimetres")
    description: str = Field(..., description="Brief human-readable description of the shot")


class FramingParams(BaseModel):
    """Camera framing / optics parameters."""

    shot_type: ShotType
    focal_length_mm: int = Field(..., description="Effective focal length in mm")
    subjects: list[str] = Field(
        default_factory=list,
        description="List of char_ids that are the primary subjects of this shot",
    )


class CameraAngle(BaseModel):
    """Camera angle parameters."""

    angle_type: CameraAngleType = Field(default=CameraAngleType.EYE_LEVEL)
    axis_side: AxisSide = Field(
        ...,
        description=(
            "Which side of the 180° relationship axis the camera occupies. "
            "Must remain constant within a scene (REQ-02)."
        ),
    )
    tilt_degrees: float = Field(
        default=0.0,
        ge=-45.0,
        le=45.0,
        description="Camera tilt in degrees (positive = looking up, negative = looking down)",
    )


class LightingParams(BaseModel):
    """Lighting and colour-grade parameters derived from emotion mapping.

    REQ-04: Semantic-Parameter Mapping.
    """

    contrast: float = Field(..., ge=0.0, le=1.0, description="Image contrast (0=flat, 1=extreme)")
    saturation: float = Field(..., ge=0.0, le=1.0, description="Colour saturation (0=B&W, 1=vivid)")
    mood: str = Field(..., description="Human-readable mood label, e.g. 'tense', 'warm'")


class MotionInstruction(BaseModel):
    """Camera / subject motion parameters."""

    motion_type: str = Field(
        ...,
        description=(
            "Type of camera move, e.g. 'STATIC', 'DOLLY_IN', 'DOLLY_OUT', "
            "'PAN_LEFT', 'PAN_RIGHT', 'HANDHELD'"
        ),
    )
    intensity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Motion intensity (0=completely static, 1=extreme movement)",
    )


class CompositionHint(BaseModel):
    """Composition and staging guidance for downstream renderers."""

    focus_point: Optional[Position] = Field(
        default=None,
        description="Primary attention point on the 0-1 canvas",
    )
    nose_room: str = Field(
        default="",
        description="Short note describing how eyeline room is preserved",
    )
    staging: str = Field(
        default="",
        description="Blocking or staging note for the shot",
    )


class CameraPlacement(BaseModel):
    """Abstract scene-space camera placement used for geometry validation."""

    x: float = Field(..., description="Camera x position in scene space")
    y: float = Field(..., description="Camera y position in scene space")
    distance: float = Field(..., ge=0.0, description="Distance from the axis origin")


class SceneBeat(BaseModel):
    """Narrative beat planned before concrete shots are generated."""

    beat_index: int = Field(..., ge=0)
    beat_type: BeatType
    emotion: str = Field(default="neutral")
    intensity: float = Field(..., ge=0.0, le=1.0)
    purpose: str = Field(default="", description="Human-readable story purpose for the beat")
    focus_char_id: Optional[str] = Field(
        default=None,
        description="Primary character emphasis for this beat, if any",
    )
    shot_type_hint: Optional[ShotType] = Field(
        default=None,
        description="Suggested shot type picked by the planner",
    )


# ---------------------------------------------------------------------------
# Top-level Shot Block
# ---------------------------------------------------------------------------


class ShotBlock(BaseModel):
    """A single cinematographic shot with all parameters needed for AI rendering.

    Each ShotBlock must contain: framing, camera_angle, lighting,
    motion_instruction (Data Pass acceptance criterion).
    """

    shot_index: int = Field(..., ge=0, description="Zero-based index of this shot in the sequence")
    framing: FramingParams
    camera_angle: CameraAngle
    lighting: LightingParams
    motion_instruction: MotionInstruction
    characters: list[Character] = Field(
        default_factory=list,
        description="Characters present in this shot with their current state",
    )
    sd_prompt: str = Field(
        default="",
        description="Assembled Stable Diffusion prompt ready for rendering",
    )
    negative_prompt: str = Field(
        default="",
        description="SD negative prompt",
    )
    beat_type: Optional[BeatType] = Field(
        default=None,
        description="Narrative beat category for this shot",
    )
    story_purpose: str = Field(
        default="",
        description="Short story-level reason this shot exists",
    )
    composition: Optional[CompositionHint] = Field(
        default=None,
        description="Composition and staging guidance",
    )
    camera_position: Optional[CameraPlacement] = Field(
        default=None,
        description="Scene-space camera placement used for axis validation",
    )
    controlnet_points: list[dict[str, float | str]] = Field(
        default_factory=list,
        description="ControlNet-compatible points derived from character positions",
    )
    scene_location: str = Field(
        default="",
        description="Best-effort scene location tag",
    )
    scene_tags: list[str] = Field(
        default_factory=list,
        description="Environment or tone tags copied from the analysed scene",
    )


# ---------------------------------------------------------------------------
# Scene-level context (feeds the pipeline)
# ---------------------------------------------------------------------------


class SceneContext(BaseModel):
    """Accumulated scene state passed between pipeline filters."""

    scene_id: str = Field(default="SCENE_01")
    description: str = Field(..., description="Raw natural-language scene description")
    characters: list[Character] = Field(default_factory=list)
    axis_side: Optional[AxisSide] = Field(
        default=None,
        description="Established axis side for the scene (set on first shot, locked thereafter)",
    )
    shot_blocks: list[ShotBlock] = Field(default_factory=list)
    detected_emotion: str = Field(
        default="neutral",
        description="Primary emotion keyword detected in this scene",
    )
    dialogue: list[DialogueLine] = Field(
        default_factory=list,
        description="Dialogue fragments extracted from the source description",
    )
    scene_location: str = Field(
        default="",
        description="Best-effort location tag such as 'tavern' or 'field'",
    )
    scene_tags: list[str] = Field(
        default_factory=list,
        description="Supplementary scene descriptors for prompts",
    )
    beats: list[SceneBeat] = Field(
        default_factory=list,
        description="Planned narrative beats for the generated shot sequence",
    )
    primary_focus_char_id: Optional[str] = Field(
        default=None,
        description="Character currently driving the scene emphasis",
    )
    analysis_source: str = Field(
        default="rule_based",
        description="How the scene was parsed: rule_based or llm",
    )


# ---------------------------------------------------------------------------
# Shot template library  (REQ-03)
# ---------------------------------------------------------------------------

SHOT_TEMPLATES: dict[ShotType, ShotTemplate] = {
    ShotType.MASTER_SHOT: ShotTemplate(
        shot_type=ShotType.MASTER_SHOT,
        focal_length_mm=24,
        description="Wide establishing shot showing full environment and all characters",
    ),
    ShotType.MEDIUM_SHOT: ShotTemplate(
        shot_type=ShotType.MEDIUM_SHOT,
        focal_length_mm=35,
        description="Waist-up framing, shows body language and environment context",
    ),
    ShotType.MCU: ShotTemplate(
        shot_type=ShotType.MCU,
        focal_length_mm=50,
        description="Medium close-up from chest/shoulders, face clearly readable",
    ),
    ShotType.CLOSE_UP: ShotTemplate(
        shot_type=ShotType.CLOSE_UP,
        focal_length_mm=85,
        description="Face fills most of frame, for intense emotional moments",
    ),
    ShotType.OVER_SHOULDER: ShotTemplate(
        shot_type=ShotType.OVER_SHOULDER,
        focal_length_mm=35,
        description="Over-the-shoulder two-shot establishing spatial relationship",
    ),
}


# ---------------------------------------------------------------------------
# Emotion matrix  (REQ-04)
# ---------------------------------------------------------------------------

EMOTION_MATRIX: dict[str, dict[str, float | str]] = {
    "angry": {"contrast": 0.8, "saturation": 0.6, "motion": 0.7, "mood": "intense"},
    "furious": {"contrast": 0.9, "saturation": 0.5, "motion": 0.9, "mood": "explosive"},
    "sad": {"contrast": 0.6, "saturation": 0.2, "motion": 0.1, "mood": "melancholic"},
    "happy": {"contrast": 0.5, "saturation": 0.9, "motion": 0.3, "mood": "warm"},
    "joyful": {"contrast": 0.4, "saturation": 1.0, "motion": 0.4, "mood": "euphoric"},
    "tense": {"contrast": 0.9, "saturation": 0.4, "motion": 0.5, "mood": "suspenseful"},
    "scared": {"contrast": 0.85, "saturation": 0.3, "motion": 0.6, "mood": "fearful"},
    "calm": {"contrast": 0.4, "saturation": 0.6, "motion": 0.1, "mood": "serene"},
    "romantic": {"contrast": 0.45, "saturation": 0.75, "motion": 0.15, "mood": "intimate"},
    "neutral": {"contrast": 0.5, "saturation": 0.5, "motion": 0.2, "mood": "neutral"},
}
