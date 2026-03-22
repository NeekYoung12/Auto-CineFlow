"""Models for reference retrieval and character consistency packaging."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FaceIDProfile(BaseModel):
    """FaceID-compatible identity descriptor derived from reference portraits."""

    profile_id: str
    source: str = "surrogate_face_descriptor"
    reference_path: str = ""
    vector: list[float] = Field(default_factory=list)


class CharacterReferenceAsset(BaseModel):
    """Indexed character asset available for retrieval."""

    asset_id: str
    character_id: str
    source_root: str
    source_project: str = ""
    manifest_path: str = ""
    prompt: str = ""
    tags: list[str] = Field(default_factory=list)
    base_image_path: str = ""
    face_reference_path: str = ""
    view_paths: dict[str, str] = Field(default_factory=dict)
    faceid_profile: FaceIDProfile | None = None
    score: float = Field(default=0.0, ge=0.0)


class SceneReferenceAsset(BaseModel):
    """Indexed scene or shot asset available for retrieval."""

    asset_id: str
    scene_id: str = ""
    source_root: str
    source_project: str = ""
    manifest_path: str = ""
    description: str = ""
    scene_location: str = ""
    tags: list[str] = Field(default_factory=list)
    preview_paths: list[str] = Field(default_factory=list)
    palette_signature: list[float] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0)


class ReferenceLibrary(BaseModel):
    """Retrieval-ready index of character and scene assets."""

    roots: list[str] = Field(default_factory=list)
    character_assets: list[CharacterReferenceAsset] = Field(default_factory=list)
    scene_assets: list[SceneReferenceAsset] = Field(default_factory=list)


class CharacterFusionPlan(BaseModel):
    """Fused character design plan assembled from retrieved candidates."""

    char_id: str
    target_visual_anchor: str
    selected_candidate_ids: list[str] = Field(default_factory=list)
    source_reference_images: list[str] = Field(default_factory=list)
    multiview_reference_images: dict[str, str] = Field(default_factory=dict)
    multiview_prompts: dict[str, str] = Field(default_factory=dict)
    fused_identity_prompt: str = ""
    face_notes: str = ""
    makeup_notes: str = ""
    wardrobe_notes: str = ""
    faceid_profile: FaceIDProfile | None = None


class SceneConsistencyPlan(BaseModel):
    """Scene-level set dressing and environment reference plan."""

    scene_id: str
    scene_location: str = ""
    selected_candidate_ids: list[str] = Field(default_factory=list)
    reference_images: list[str] = Field(default_factory=list)
    reference_clips: list[str] = Field(default_factory=list)
    set_dressing_prompt: str = ""
    lighting_reference_notes: str = ""
    palette_signature: list[float] = Field(default_factory=list)


class ShotReferenceBundle(BaseModel):
    """Per-shot reference bundle that can feed render backends."""

    shot_id: str
    primary_subjects: list[str] = Field(default_factory=list)
    character_reference_images: list[str] = Field(default_factory=list)
    scene_reference_images: list[str] = Field(default_factory=list)
    faceid_profile_ids: dict[str, str] = Field(default_factory=dict)
    selected_character_views: dict[str, str] = Field(default_factory=dict)
    prompt_suffix: str = ""


class ConsistencyPackage(BaseModel):
    """Full character/scene consistency package attached to one storyboard scene."""

    scene_id: str
    retrieval_roots: list[str] = Field(default_factory=list)
    generated_at: str
    library_character_count: int = Field(default=0, ge=0)
    library_scene_count: int = Field(default=0, ge=0)
    character_candidates: dict[str, list[CharacterReferenceAsset]] = Field(default_factory=dict)
    scene_candidates: list[SceneReferenceAsset] = Field(default_factory=list)
    character_plans: list[CharacterFusionPlan] = Field(default_factory=list)
    scene_plan: SceneConsistencyPlan | None = None
    shot_bundles: list[ShotReferenceBundle] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
