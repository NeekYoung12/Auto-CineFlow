"""Lightweight retrieval over local character and scene asset libraries."""

from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path

from PIL import Image, ImageStat

from .consistency_models import (
    CharacterReferenceAsset,
    FaceIDProfile,
    ReferenceLibrary,
    SceneReferenceAsset,
)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".ppm"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.IGNORECASE)


def _tokenize(*parts: str) -> set[str]:
    tokens: set[str] = set()
    for part in parts:
        if not part:
            continue
        tokens.update(token.lower() for token in _TOKEN_RE.findall(part))
    return tokens


def _stable_id(*parts: str) -> str:
    digest = sha256("::".join(parts).encode("utf-8")).hexdigest()
    return digest[:12]


def _score_tokens(query: set[str], candidate: set[str]) -> float:
    if not query or not candidate:
        return 0.0
    overlap = len(query.intersection(candidate))
    coverage = overlap / len(query)
    balance = overlap / len(query.union(candidate))
    return round((coverage * 0.75) + (balance * 0.25), 4)


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _is_local_file(path_or_url: str) -> bool:
    if not path_or_url or path_or_url.startswith(("http://", "https://")):
        return False
    return Path(path_or_url).exists()


def _source_project(root: Path, item_path: Path) -> str:
    try:
        relative_parts = item_path.relative_to(root).parts
    except ValueError:
        relative_parts = item_path.parts
    return relative_parts[0] if relative_parts else item_path.parent.name


def _candidate_face_path(base_image_path: str, view_paths: dict[str, str]) -> str:
    preferred = (
        view_paths.get("close_up")
        or view_paths.get("three_quarter_right")
        or view_paths.get("three_quarter_left")
        or base_image_path
    )
    return preferred if _is_local_file(preferred) else ""


def _build_faceid_profile(image_path: str, profile_prefix: str) -> FaceIDProfile | None:
    if not _is_local_file(image_path):
        return None

    path = Path(image_path)
    with Image.open(path) as image:
        image = image.convert("L")
        width, height = image.size
        left = int(width * 0.18)
        top = int(height * 0.12)
        right = max(left + 8, int(width * 0.82))
        bottom = max(top + 8, int(height * 0.88))
        crop = image.crop((left, top, right, bottom)).resize((8, 8))
        pixels = [value / 255.0 for value in crop.tobytes()]
        mean_value = sum(pixels) / len(pixels)
        vector = [round(value - mean_value, 4) for value in pixels]

    return FaceIDProfile(
        profile_id=f"{profile_prefix}_{_stable_id(str(path), str(path.stat().st_size))}",
        reference_path=str(path),
        vector=vector,
    )


def _build_palette_signature(image_path: str) -> list[float]:
    if not _is_local_file(image_path):
        return []
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((16, 16))
        stat = ImageStat.Stat(image)
        return [round(channel / 255.0, 4) for channel in stat.mean[:3]]


def _scene_preview_paths(base_dir: Path) -> list[str]:
    paths: list[str] = []
    for candidate in sorted(base_dir.rglob("*")):
        if candidate.suffix.lower() in _IMAGE_EXTENSIONS | _VIDEO_EXTENSIONS:
            paths.append(str(candidate))
        if len(paths) >= 6:
            break
    return paths


def _storyboard_output_root(manifest_path: Path) -> Path:
    if manifest_path.parent.name == "delivery":
        return manifest_path.parent.parent
    return manifest_path.parent


def _scan_character_manifests(root: Path) -> list[CharacterReferenceAsset]:
    assets: list[CharacterReferenceAsset] = []
    for manifest_path in root.rglob("*_manifest.json"):
        payload = _load_json(manifest_path)
        if not isinstance(payload, dict):
            continue
        if "character_id" not in payload or "standardized_views" not in payload:
            continue

        view_paths = {
            view: path
            for view, path in (payload.get("standardized_views", {}) or {}).items()
            if isinstance(path, str)
        }
        base_image_path = str(payload.get("base_image_path", "") or "")
        face_reference_path = _candidate_face_path(base_image_path, view_paths)
        prompt = str(payload.get("prompt", "") or "")
        tags = sorted(
            _tokenize(
                prompt,
                payload.get("character_id", ""),
                " ".join(view_paths.keys()),
                str(manifest_path.parent),
            )
        )
        assets.append(
            CharacterReferenceAsset(
                asset_id=f"charref_{_stable_id(str(manifest_path))}",
                character_id=str(payload.get("character_id", "") or manifest_path.stem),
                source_root=str(root),
                source_project=_source_project(root, manifest_path),
                manifest_path=str(manifest_path),
                prompt=prompt,
                tags=tags,
                base_image_path=base_image_path,
                face_reference_path=face_reference_path,
                view_paths=view_paths,
                faceid_profile=_build_faceid_profile(face_reference_path, str(payload.get("character_id", "char"))),
            )
        )
    return assets


def _scan_storyboard_packages(root: Path) -> list[SceneReferenceAsset]:
    assets: list[SceneReferenceAsset] = []
    for manifest_path in root.rglob("storyboard_package.json"):
        payload = _load_json(manifest_path)
        if not isinstance(payload, dict):
            continue
        shots = payload.get("shots", []) or []
        prompt_parts = [str(shot.get("prompt", "")) for shot in shots[:3] if isinstance(shot, dict)]
        scene_id = str(payload.get("scene_id", "") or manifest_path.parent.name)
        scene_location = str(payload.get("scene_location", "") or "")
        scene_tags = payload.get("scene_tags", []) or []
        output_root = _storyboard_output_root(manifest_path)
        preview_paths = _scene_preview_paths(output_root)
        preview_image = next((path for path in preview_paths if Path(path).suffix.lower() in _IMAGE_EXTENSIONS), "")
        assets.append(
            SceneReferenceAsset(
                asset_id=f"sceneref_{_stable_id(str(manifest_path))}",
                scene_id=scene_id,
                source_root=str(root),
                source_project=str(payload.get("project_name", "") or _source_project(root, manifest_path)),
                manifest_path=str(manifest_path),
                description=" ".join(part for part in prompt_parts if part),
                scene_location=scene_location,
                tags=sorted(_tokenize(scene_id, scene_location, " ".join(map(str, scene_tags)), str(output_root))),
                preview_paths=preview_paths,
                palette_signature=_build_palette_signature(preview_image),
            )
        )
    return assets


def _scan_report_json(root: Path) -> list[SceneReferenceAsset]:
    assets: list[SceneReferenceAsset] = []
    for report_path in root.rglob("report.json"):
        payload = _load_json(report_path)
        if not isinstance(payload, dict):
            continue
        analysis = payload.get("analysis", {}) if isinstance(payload.get("analysis"), dict) else {}
        summary = str(analysis.get("summary", "") or payload.get("summary", "") or "")
        scene_tags = payload.get("scene_tags", []) or []

        preview_paths: list[str] = []
        for key in ("generated_image", "final_video_path", "video_path", "output_path"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                preview_paths.append(value)

        preview_image = next((path for path in preview_paths if Path(path).suffix.lower() in _IMAGE_EXTENSIONS), "")
        assets.append(
            SceneReferenceAsset(
                asset_id=f"reportref_{_stable_id(str(report_path))}",
                scene_id=report_path.parent.name,
                source_root=str(root),
                source_project=_source_project(root, report_path),
                manifest_path=str(report_path),
                description=summary,
                tags=sorted(_tokenize(summary, " ".join(map(str, scene_tags)), str(report_path.parent))),
                preview_paths=preview_paths,
                palette_signature=_build_palette_signature(preview_image),
            )
        )
    return assets


def build_reference_library(reference_roots: list[str | Path]) -> ReferenceLibrary:
    """Build a retrieval index from one or more local asset roots."""

    normalized_roots: list[Path] = []
    character_assets: list[CharacterReferenceAsset] = []
    scene_assets: list[SceneReferenceAsset] = []

    for root in reference_roots:
        path = Path(root)
        if not path.exists():
            continue
        normalized_roots.append(path)
        character_assets.extend(_scan_character_manifests(path))
        scene_assets.extend(_scan_storyboard_packages(path))
        scene_assets.extend(_scan_report_json(path))

    return ReferenceLibrary(
        roots=[str(path) for path in normalized_roots],
        character_assets=list({asset.asset_id: asset for asset in character_assets}.values()),
        scene_assets=list({asset.asset_id: asset for asset in scene_assets}.values()),
    )


def retrieve_character_assets(
    library: ReferenceLibrary,
    query: str,
    top_k: int = 3,
) -> list[CharacterReferenceAsset]:
    """Retrieve the top matching character assets for a query string."""

    query_tokens = _tokenize(query)
    ranked: list[CharacterReferenceAsset] = []
    for asset in library.character_assets:
        candidate_tokens = _tokenize(asset.prompt, " ".join(asset.tags), asset.character_id)
        score = _score_tokens(query_tokens, candidate_tokens)
        if asset.faceid_profile and asset.faceid_profile.vector:
            score += 0.05
        if asset.view_paths:
            score += min(0.08, len(asset.view_paths) * 0.01)
        ranked.append(asset.model_copy(update={"score": round(score, 4)}))

    ranked.sort(key=lambda item: (-item.score, item.character_id, item.asset_id))
    return ranked[:top_k]


def retrieve_scene_assets(
    library: ReferenceLibrary,
    query: str,
    top_k: int = 4,
) -> list[SceneReferenceAsset]:
    """Retrieve the top matching scene assets for a query string."""

    query_tokens = _tokenize(query)
    ranked: list[SceneReferenceAsset] = []
    for asset in library.scene_assets:
        candidate_tokens = _tokenize(asset.description, asset.scene_location, " ".join(asset.tags), asset.scene_id)
        score = _score_tokens(query_tokens, candidate_tokens)
        if asset.preview_paths:
            score += 0.04
        if asset.palette_signature:
            score += 0.02
        ranked.append(asset.model_copy(update={"score": round(score, 4)}))

    ranked.sort(key=lambda item: (-item.score, item.scene_id, item.asset_id))
    return ranked[:top_k]
