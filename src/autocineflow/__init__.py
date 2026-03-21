"""Auto-CineFlow: LLM-driven cinematic storyboard parameter generator."""

from importlib import import_module

from .models import (
    AxisSide,
    BeatType,
    CameraAngle,
    CameraAngleType,
    CameraPlacement,
    Character,
    CharacterFacing,
    CompositionHint,
    DialogueLine,
    FramingParams,
    LightingParams,
    MotionInstruction,
    SceneBeat,
    SceneContext,
    ShotBlock,
    ShotTemplate,
    ShotType,
)

__all__ = [
    "AxisSide",
    "BeatType",
    "CameraAngle",
    "CameraAngleType",
    "CameraPlacement",
    "Character",
    "ProjectExecutionPlan",
    "ProjectDashboard",
    "ProjectRenderQAReport",
    "ProjectChangePlan",
    "CharacterFacing",
    "CompositionHint",
    "DeliveryShot",
    "DialogueLine",
    "FramingParams",
    "LightingParams",
    "MotionInstruction",
    "ProjectPackage",
    "ProjectSceneInput",
    "RenderExpectation",
    "RenderQAReport",
    "RenderJob",
    "RenderPreset",
    "SceneBeat",
    "SceneContext",
    "ShotBlock",
    "ShotTemplate",
    "ShotType",
    "StoryboardPackage",
    "CineFlowPipeline",
]

_LAZY_EXPORTS = {
    "DeliveryShot": ".delivery",
    "ProjectPackage": ".project_delivery",
    "ProjectSceneInput": ".project_delivery",
    "ProjectChangePlan": ".change_planner",
    "ProjectExecutionPlan": ".execution_planner",
    "ProjectDashboard": ".project_dashboard",
    "ProjectRenderQAReport": ".project_render_qa",
    "RenderExpectation": ".render_qa",
    "RenderQAReport": ".render_qa",
    "RenderJob": ".delivery",
    "RenderPreset": ".delivery",
    "StoryboardPackage": ".delivery",
    "CineFlowPipeline": ".pipeline",
}


def __getattr__(name: str):
    """Lazily load heavy exports to keep CLI module execution clean."""

    if name in _LAZY_EXPORTS:
        module = import_module(_LAZY_EXPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
