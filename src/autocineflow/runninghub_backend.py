"""RunningHub workflow submission, upload, and output polling helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx

from .config_loader import (
    resolve_runninghub_api_format_dir,
    resolve_runninghub_settings,
    resolve_runninghub_workflow_ids,
)


def load_runninghub_api_format(
    workflow_key: str,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Load the exported RunningHub API-format workflow JSON for one workflow key."""

    api_format_dir = resolve_runninghub_api_format_dir(config_path)
    if api_format_dir is None:
        raise ValueError("RunningHub API format directory is not configured.")

    path = api_format_dir / f"{workflow_key}.json"
    if not path.exists():
        raise FileNotFoundError(f"RunningHub API format JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_runninghub_workflow_id(
    workflow_id_env: str,
    config_path: str | None = None,
) -> str:
    """Resolve one configured RunningHub workflow ID by env-style variable name."""

    workflow_ids = resolve_runninghub_workflow_ids(config_path)
    workflow_id = workflow_ids.get(workflow_id_env, "")
    if not workflow_id:
        raise ValueError(f"Missing RunningHub workflow ID: {workflow_id_env}")
    return workflow_id


def upload_runninghub_file(
    file_path: str | Path,
    file_type: str,
    config_path: str | None = None,
    timeout_seconds: float = 120.0,
) -> str:
    """Upload a local resource to RunningHub and return the server-side fileName."""

    api_key, base_url = resolve_runninghub_settings(config_path)
    if not api_key:
        raise ValueError("RunningHub API key not found in config.")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"RunningHub upload file does not exist: {path}")

    with path.open("rb") as handle:
        response = httpx.post(
            f"{base_url}/task/openapi/upload",
            data={"apiKey": api_key, "fileType": file_type},
            files={"file": (path.name, handle, "application/octet-stream")},
            timeout=timeout_seconds,
        )
    response.raise_for_status()
    payload = response.json()
    if _runninghub_code(payload) != 0:
        raise RuntimeError(f"RunningHub upload failed: {payload}")
    file_name = str((payload.get("data") or {}).get("fileName", "") or "")
    if not file_name:
        raise RuntimeError(f"RunningHub upload returned no fileName: {payload}")
    return file_name


def submit_runninghub_task(
    workflow_id: str,
    node_info_list: list[dict[str, Any]],
    config_path: str | None = None,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Submit a RunningHub workflow task and return the raw response payload."""

    api_key, base_url = resolve_runninghub_settings(config_path)
    if not api_key:
        raise ValueError("RunningHub API key not found in config.")

    response = httpx.post(
        f"{base_url}/task/openapi/create",
        headers={"Content-Type": "application/json", "Host": "www.runninghub.cn"},
        json={
            "apiKey": api_key,
            "workflowId": workflow_id,
            "nodeInfoList": node_info_list,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if _runninghub_code(payload) != 0:
        raise RuntimeError(f"RunningHub create failed: {payload}")
    return payload


def query_runninghub_outputs(
    task_id: str,
    config_path: str | None = None,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Query one RunningHub task's outputs payload."""

    api_key, base_url = resolve_runninghub_settings(config_path)
    if not api_key:
        raise ValueError("RunningHub API key not found in config.")

    response = httpx.post(
        f"{base_url}/task/openapi/outputs",
        headers={"Content-Type": "application/json", "Host": "www.runninghub.cn"},
        json={"apiKey": api_key, "taskId": task_id},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if _runninghub_code(payload) not in {0, 804}:
        raise RuntimeError(f"RunningHub outputs query failed: {payload}")
    return payload


def poll_runninghub_outputs(
    task_id: str,
    config_path: str | None = None,
    timeout_seconds: float = 900.0,
    poll_interval_seconds: float = 10.0,
) -> dict[str, Any]:
    """Poll RunningHub task outputs until media URLs are available or the task fails."""

    deadline = time.monotonic() + timeout_seconds
    last_payload: dict[str, Any] | None = None

    while time.monotonic() < deadline:
        payload = query_runninghub_outputs(task_id, config_path=config_path)
        last_payload = payload
        urls = extract_runninghub_media_urls(payload, config_path=config_path)
        if urls:
            return payload

        status = infer_runninghub_status(payload)
        if status in {"FAILED", "ERROR", "CANCELLED"}:
            raise RuntimeError(f"RunningHub task failed: {payload}")
        time.sleep(poll_interval_seconds)

    raise TimeoutError(f"Timed out waiting for RunningHub task {task_id}: {last_payload}")


def infer_runninghub_status(payload: dict[str, Any]) -> str:
    """Best-effort status extraction from a RunningHub outputs payload."""

    candidates = _collect_values_for_keys(payload, {"taskStatus", "status", "state"})
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            upper = candidate.upper()
            if upper not in {"NULL", "NONE"}:
                return upper
    if extract_runninghub_media_urls(payload):
        return "SUCCESS"
    return "RUNNING"


def extract_runninghub_media_urls(
    payload: dict[str, Any],
    config_path: str | None = None,
) -> list[str]:
    """Extract image/video URLs from a RunningHub outputs payload."""

    _, base_url = resolve_runninghub_settings(config_path)
    urls: list[str] = []
    for value in _walk_values(payload):
        if not isinstance(value, str):
            continue
        candidate = value.strip()
        if not candidate:
            continue
        if candidate.startswith("/view?"):
            candidate = f"{base_url}{candidate}"
        elif candidate.startswith("view?"):
            candidate = f"{base_url}/{candidate}"

        lowered = candidate.lower()
        if candidate.startswith(("http://", "https://")) and (
            "/view?" in lowered
            or any(ext in lowered for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".mp4", ".mov", ".webm", ".mkv"))
        ):
            if candidate not in urls:
                urls.append(candidate)
    return urls


def prepare_runninghub_job(
    workflow_key: str,
    payload: dict[str, Any],
    config_path: str | None = None,
    timeout_seconds: float = 120.0,
) -> tuple[str, list[dict[str, Any]]]:
    """Prepare a RunningHub workflow task by uploading local assets and building node overrides."""

    template = load_runninghub_api_format(workflow_key, config_path=config_path)
    overrides: list[dict[str, Any]]
    if workflow_key == "rh_shot_keyframe_faceid_v1":
        overrides = _prepare_faceid_keyframe_overrides(payload, timeout_seconds=timeout_seconds, config_path=config_path)
    elif workflow_key == "rh_shot_i2v_wan22_full_v1":
        overrides = _prepare_wan22_video_overrides(payload, timeout_seconds=timeout_seconds, config_path=config_path)
    elif workflow_key == "rh_shot_i2v_wan21_hq_v1":
        overrides = _prepare_wan22_video_overrides(payload, timeout_seconds=timeout_seconds, config_path=config_path)
    elif workflow_key == "rh_shot_i2v_framepack_fast_v1":
        overrides = _prepare_wan22_video_overrides(payload, timeout_seconds=timeout_seconds, config_path=config_path)
    else:
        raise NotImplementedError(f"RunningHub workflow preparation not implemented for: {workflow_key}")

    available_nodes = set(template.keys())
    validated = [item for item in overrides if item["nodeId"] in available_nodes]
    return resolve_runninghub_workflow_id(payload["workflow_id_env"], config_path=config_path), validated


def _prepare_faceid_keyframe_overrides(
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
    config_path: str | None,
) -> list[dict[str, Any]]:
    inputs = payload.get("workflow_inputs", {})
    character_refs = list(inputs.get("character_reference_images", []))
    scene_refs = list(inputs.get("scene_reference_images", []))
    face_ref = _upload_if_needed(character_refs[0] if character_refs else "", config_path, timeout_seconds)
    init_ref = _upload_if_needed(scene_refs[0] if scene_refs else (character_refs[1] if len(character_refs) > 1 else (character_refs[0] if character_refs else "")), config_path, timeout_seconds)

    prompt = str(inputs.get("positive_prompt", "") or "")
    seed = int(inputs.get("seed", -1) or -1)
    steps = int(inputs.get("steps", 30) or 30)
    use_img2img = bool(init_ref)

    overrides = [
        {"nodeId": "220", "fieldName": "prompt", "fieldValue": prompt},
        {"nodeId": "221", "fieldName": "boolean", "fieldValue": False},
        {"nodeId": "201", "fieldName": "seed", "fieldValue": seed},
        {"nodeId": "214", "fieldName": "int", "fieldValue": steps},
        {"nodeId": "215", "fieldName": "boolean", "fieldValue": use_img2img},
    ]
    if face_ref:
        overrides.append({"nodeId": "198", "fieldName": "image", "fieldValue": face_ref})
    if init_ref:
        overrides.append({"nodeId": "205", "fieldName": "image", "fieldValue": init_ref})
    return overrides


def _prepare_wan22_video_overrides(
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
    config_path: str | None,
) -> list[dict[str, Any]]:
    contract = payload.get("request_contract", {})
    first_candidates = list(contract.get("first_frame_candidates", []))
    scene_refs = list(contract.get("scene_reference_images", []))
    first_frame = _upload_if_needed(first_candidates[0] if first_candidates else "", config_path, timeout_seconds)
    tail_source = scene_refs[0] if scene_refs else (first_candidates[1] if len(first_candidates) > 1 else (first_candidates[0] if first_candidates else ""))
    tail_frame = _upload_if_needed(tail_source, config_path, timeout_seconds)

    overrides = [
        {"nodeId": "234", "fieldName": "value", "fieldValue": str(contract.get("prompt", "") or "")},
        {"nodeId": "201", "fieldName": "text", "fieldValue": str(contract.get("negative_prompt", "") or "")},
    ]
    if first_frame:
        overrides.append({"nodeId": "230", "fieldName": "image", "fieldValue": first_frame})
    if tail_frame:
        overrides.append({"nodeId": "225", "fieldName": "image", "fieldValue": tail_frame})
    return overrides


def _upload_if_needed(
    path_or_url: str,
    config_path: str | None,
    timeout_seconds: float,
) -> str:
    if not path_or_url:
        return ""
    if path_or_url.startswith(("http://", "https://")):
        return path_or_url
    return upload_runninghub_file(path_or_url, "image", config_path=config_path, timeout_seconds=timeout_seconds)


def _collect_values_for_keys(obj: Any, keys: set[str]) -> list[Any]:
    values: list[Any] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in keys:
                values.append(value)
            values.extend(_collect_values_for_keys(value, keys))
    elif isinstance(obj, list):
        for item in obj:
            values.extend(_collect_values_for_keys(item, keys))
    return values


def _walk_values(obj: Any) -> list[Any]:
    values: list[Any] = []
    if isinstance(obj, dict):
        for value in obj.values():
            values.append(value)
            values.extend(_walk_values(value))
    elif isinstance(obj, list):
        for item in obj:
            values.append(item)
            values.extend(_walk_values(item))
    return values


def _runninghub_code(payload: dict[str, Any]) -> int:
    """Normalize RunningHub response code handling without treating 0 as falsy failure."""

    if "code" in payload:
        return int(payload["code"])
    if "status" in payload:
        try:
            return int(payload["status"])
        except (TypeError, ValueError):
            return 0 if str(payload["status"]).lower() in {"success", "ok"} else -1
    return -1
