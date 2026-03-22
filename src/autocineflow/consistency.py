"""Character and scene consistency packaging built on local reference retrieval."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import unicodedata

from .consistency_models import (
    CharacterFusionPlan,
    ConsistencyPackage,
    FaceIDProfile,
    ReferenceLibrary,
    SceneConsistencyPlan,
    ShotReferenceBundle,
)
from .models import SceneContext, ShotType
from .reference_rag import build_reference_library, retrieve_character_assets, retrieve_scene_assets

_STANDARD_VIEWS = (
    "wide_shot",
    "three_quarter_right",
    "profile_right",
    "birds_eye",
    "low_angle",
    "three_quarter_left",
    "profile_left",
    "close_up",
)

_STYLE_KEYWORDS = {
    "makeup": {"makeup", "lipstick", "eyeliner", "mascara", "blush", "scar", "stubble", "beard"},
    "wardrobe": {"coat", "dress", "jacket", "suit", "shirt", "armor", "hat", "hoodie", "trench", "uniform"},
    "face": {"face", "eyes", "jaw", "cheek", "nose", "hair", "freckles", "eyebrow", "beard", "scar"},
}


def default_reference_roots() -> list[Path]:
    """Return the default local asset roots used for retrieval."""

    repo_root = Path(__file__).resolve().parents[2]
    sibling_root = repo_root.parent / "video-studio-system"
    candidates = [
        repo_root / "out",
        sibling_root / "ai_film_studio" / "assets",
        sibling_root / "runtime_outputs",
    ]
    return [path for path in candidates if path.exists()]


def _flatten_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or _contains_private_use_chars(normalized) or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _phrase_list(*sources: str) -> list[str]:
    phrases: list[str] = []
    for source in sources:
        for part in source.replace(";", ",").split(","):
            cleaned = part.strip()
            if cleaned:
                phrases.append(cleaned)
    return _flatten_unique(phrases)


def _style_notes(phrases: list[str], bucket: str) -> str:
    keywords = _STYLE_KEYWORDS[bucket]
    selected = [phrase for phrase in phrases if any(keyword in phrase.lower() for keyword in keywords)]
    return ", ".join(selected[:4])


def _merge_face_profiles(profiles: list[FaceIDProfile]) -> FaceIDProfile | None:
    valid = [profile for profile in profiles if profile.vector]
    if not valid:
        return None
    dimension = len(valid[0].vector)
    merged = []
    for index in range(dimension):
        merged.append(round(sum(profile.vector[index] for profile in valid) / len(valid), 4))
    return FaceIDProfile(
        profile_id=f"merged_{valid[0].profile_id}",
        source="surrogate_face_descriptor_average",
        reference_path=valid[0].reference_path,
        vector=merged,
    )


def _build_multiview_prompts(fused_prompt: str) -> dict[str, str]:
    view_descriptions = {
        "wide_shot": "full body hero sheet, wardrobe fully visible",
        "three_quarter_right": "three quarter right turn, identity stable",
        "profile_right": "clean right profile, silhouette readable",
        "birds_eye": "top view costume breakup and silhouette",
        "low_angle": "low angle power portrait, same face and outfit",
        "three_quarter_left": "three quarter left turn, identity stable",
        "profile_left": "clean left profile, silhouette readable",
        "close_up": "tight close-up beauty shot, facial detail preserved",
    }
    return {
        view: f"{fused_prompt}, {description}, multiview character sheet, same identity, same wardrobe"
        for view, description in view_descriptions.items()
    }


def _contains_private_use_chars(text: str) -> bool:
    return any(unicodedata.category(char) == "Co" for char in text)


def _shot_preferred_view(shot_type: str) -> str:
    if shot_type == ShotType.CLOSE_UP.value:
        return "close_up"
    if shot_type == ShotType.MCU.value:
        return "three_quarter_right"
    if shot_type == ShotType.OVER_SHOULDER.value:
        return "profile_right"
    return "wide_shot"


def build_consistency_package(
    context: SceneContext,
    shot_records: list[dict[str, object]],
    reference_roots: list[str | Path] | None = None,
    reference_library: ReferenceLibrary | None = None,
    character_top_k: int = 3,
    scene_top_k: int = 4,
) -> ConsistencyPackage:
    """Build a consistency package from retrieved character and scene references."""

    library = reference_library or build_reference_library(list(reference_roots or default_reference_roots()))
    retrieval_roots = library.roots
    character_candidates: dict[str, list] = {}
    character_plans: list[CharacterFusionPlan] = []
    notes: list[str] = []

    for character in context.characters:
        query = " ".join(
            [
                character.visual_anchor,
                context.description,
                context.scene_location,
                " ".join(context.scene_tags),
            ]
        ).strip()
        candidates = retrieve_character_assets(library, query, top_k=character_top_k)
        character_candidates[character.char_id] = candidates

        candidate_prompts = [candidate.prompt for candidate in candidates if candidate.prompt]
        prompt_phrases = _phrase_list(character.visual_anchor, *candidate_prompts)
        if not prompt_phrases:
            prompt_phrases = _phrase_list(character.visual_anchor)

        multiview_reference_images: dict[str, str] = {}
        source_reference_images: list[str] = []
        face_profiles: list[FaceIDProfile] = []
        selected_candidate_ids: list[str] = []
        for candidate in candidates:
            selected_candidate_ids.append(candidate.asset_id)
            if candidate.face_reference_path:
                source_reference_images.append(candidate.face_reference_path)
            if candidate.base_image_path:
                source_reference_images.append(candidate.base_image_path)
            for view in _STANDARD_VIEWS:
                if view in candidate.view_paths and view not in multiview_reference_images:
                    multiview_reference_images[view] = candidate.view_paths[view]
            if candidate.faceid_profile:
                face_profiles.append(candidate.faceid_profile)

        source_reference_images = _flatten_unique(source_reference_images)[:6]
        if not candidates:
            notes.append(f"{character.char_id}: no character references retrieved")

        fused_prompt = ", ".join(prompt_phrases[:10]) or character.visual_anchor
        character_plans.append(
            CharacterFusionPlan(
                char_id=character.char_id,
                target_visual_anchor=character.visual_anchor,
                selected_candidate_ids=selected_candidate_ids,
                source_reference_images=source_reference_images,
                multiview_reference_images=multiview_reference_images,
                multiview_prompts=_build_multiview_prompts(fused_prompt),
                fused_identity_prompt=fused_prompt,
                face_notes=_style_notes(prompt_phrases, "face"),
                makeup_notes=_style_notes(prompt_phrases, "makeup"),
                wardrobe_notes=_style_notes(prompt_phrases, "wardrobe"),
                faceid_profile=_merge_face_profiles(face_profiles),
            )
        )

    scene_query = " ".join(
        [
            context.description,
            context.scene_location,
            context.detected_emotion,
            " ".join(context.scene_tags),
        ]
    ).strip()
    scene_candidates = retrieve_scene_assets(library, scene_query, top_k=scene_top_k)
    scene_images: list[str] = []
    scene_clips: list[str] = []
    palette_signature: list[float] = []
    for candidate in scene_candidates:
        for path in candidate.preview_paths:
            suffix = Path(path).suffix.lower()
            if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".ppm"}:
                scene_images.append(path)
            elif suffix in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
                scene_clips.append(path)
        if candidate.palette_signature and not palette_signature:
            palette_signature = candidate.palette_signature
    scene_images = _flatten_unique(scene_images)[:6]
    scene_clips = _flatten_unique(scene_clips)[:3]
    if not scene_candidates:
        notes.append("scene: no environment references retrieved")

    scene_plan = SceneConsistencyPlan(
        scene_id=context.scene_id,
        scene_location=context.scene_location,
        selected_candidate_ids=[candidate.asset_id for candidate in scene_candidates],
        reference_images=scene_images,
        reference_clips=scene_clips,
        set_dressing_prompt=", ".join(
            _phrase_list(
                context.scene_location,
                " ".join(context.scene_tags),
                context.detected_emotion,
                *(candidate.description for candidate in scene_candidates),
            )[:12]
        ),
        lighting_reference_notes=", ".join(_phrase_list(context.detected_emotion, " ".join(context.scene_tags))[:6]),
        palette_signature=palette_signature,
    )

    plan_by_char = {plan.char_id: plan for plan in character_plans}
    shot_bundles: list[ShotReferenceBundle] = []
    for shot in shot_records:
        shot_type = str(shot.get("shot_type", ""))
        preferred_view = _shot_preferred_view(shot_type)
        character_reference_images: list[str] = []
        faceid_profile_ids: dict[str, str] = {}
        selected_views: dict[str, str] = {}
        prompt_terms: list[str] = []

        primary_subjects = list(shot.get("primary_subjects", []))
        for subject in primary_subjects:
            plan = plan_by_char.get(subject)
            if not plan:
                continue
            selected_path = plan.multiview_reference_images.get(preferred_view) or next(
                iter(plan.multiview_reference_images.values()),
                "",
            )
            if selected_path:
                character_reference_images.append(selected_path)
                selected_views[subject] = preferred_view
            character_reference_images.extend(plan.source_reference_images[:2])
            if plan.faceid_profile:
                faceid_profile_ids[subject] = plan.faceid_profile.profile_id
            prompt_terms.append(plan.fused_identity_prompt)

        prompt_terms.append(scene_plan.set_dressing_prompt)
        shot_bundles.append(
            ShotReferenceBundle(
                shot_id=str(shot.get("shot_id", "")),
                primary_subjects=primary_subjects,
                character_reference_images=_flatten_unique(character_reference_images)[:6],
                scene_reference_images=scene_plan.reference_images[:4],
                faceid_profile_ids=faceid_profile_ids,
                selected_character_views=selected_views,
                prompt_suffix=", ".join(_flatten_unique(prompt_terms)[:8]),
            )
        )

    return ConsistencyPackage(
        scene_id=context.scene_id,
        retrieval_roots=retrieval_roots,
        generated_at=datetime.now(timezone.utc).isoformat(),
        library_character_count=len(library.character_assets),
        library_scene_count=len(library.scene_assets),
        character_candidates=character_candidates,
        scene_candidates=scene_candidates,
        character_plans=character_plans,
        scene_plan=scene_plan,
        shot_bundles=shot_bundles,
        notes=notes,
    )


def consistency_package_json(package: ConsistencyPackage, indent: int = 2) -> str:
    """Serialise a consistency package to JSON."""

    return package.model_dump_json(indent=indent)


def consistency_review_markdown(package: ConsistencyPackage) -> str:
    """Human-readable review of retrieved references and fusion plans."""

    lines = [
        f"# Consistency Package {package.scene_id}",
        "",
        f"- Retrieval Roots: `{len(package.retrieval_roots)}`",
        f"- Character Library Size: `{package.library_character_count}`",
        f"- Scene Library Size: `{package.library_scene_count}`",
        "",
        "## Character Plans",
        "",
    ]
    for plan in package.character_plans:
        lines.extend(
            [
                f"### {plan.char_id}",
                "",
                f"- Target Anchor: {plan.target_visual_anchor}",
                f"- Candidate IDs: `{', '.join(plan.selected_candidate_ids) or 'none'}`",
                f"- Face Notes: {plan.face_notes or 'n/a'}",
                f"- Makeup Notes: {plan.makeup_notes or 'n/a'}",
                f"- Wardrobe Notes: {plan.wardrobe_notes or 'n/a'}",
                f"- FaceID Profile: `{plan.faceid_profile.profile_id if plan.faceid_profile else 'n/a'}`",
                "",
                "**Fusion Prompt**",
                "",
                plan.fused_identity_prompt or "n/a",
                "",
            ]
        )
    lines.extend(["", "## Scene Plan", ""])
    if package.scene_plan:
        lines.extend(
            [
                f"- Scene Location: `{package.scene_plan.scene_location or 'unspecified'}`",
                f"- Candidate IDs: `{', '.join(package.scene_plan.selected_candidate_ids) or 'none'}`",
                f"- Reference Images: `{len(package.scene_plan.reference_images)}`",
                f"- Reference Clips: `{len(package.scene_plan.reference_clips)}`",
                "",
                package.scene_plan.set_dressing_prompt or "n/a",
                "",
            ]
        )
    if package.notes:
        lines.extend(["## Notes", ""])
        lines.extend(f"- {note}" for note in package.notes)
    return "\n".join(lines).strip() + "\n"


def write_consistency_package(package: ConsistencyPackage, output_dir: str | Path) -> dict[str, Path]:
    """Write consistency outputs to disk."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / "consistency_package.json"
    review_path = target_dir / "consistency_review.md"
    json_path.write_text(consistency_package_json(package, indent=2), encoding="utf-8")
    review_path.write_text(consistency_review_markdown(package), encoding="utf-8")
    return {
        "consistency_package": json_path,
        "consistency_review": review_path,
    }
