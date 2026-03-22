"""Provider credential and account probes for real backend readiness."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from .config_loader import (
    resolve_runninghub_api_format_dir,
    resolve_runninghub_settings,
    resolve_runninghub_workflow_ids,
    resolve_volcengine_ark_settings,
)
from .runninghub_workflows import recommended_runninghub_workflows


class ProviderProbeResult(BaseModel):
    """Health result for a single external generation provider."""

    provider: str
    ok: bool
    endpoint: str = ""
    message: str = ""
    details: dict[str, object] = Field(default_factory=dict)


class ProviderProbeReport(BaseModel):
    """Aggregate provider probe report."""

    results: list[ProviderProbeResult] = Field(default_factory=list)


def probe_runninghub_account(
    config_path: str | None = None,
    timeout_seconds: float = 30.0,
) -> ProviderProbeResult:
    """Probe RunningHub account status using the configured API key."""

    api_key, base_url = resolve_runninghub_settings(config_path)
    if not api_key:
        return ProviderProbeResult(
            provider="runninghub",
            ok=False,
            endpoint=f"{base_url}/uc/openapi/accountStatus",
            message="missing_api_key",
        )

    endpoint = f"{base_url}/uc/openapi/accountStatus"
    response = httpx.post(endpoint, json={"apiKey": api_key}, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    code = payload.get("code", payload.get("status", 0)) if isinstance(payload, dict) else 0
    ok = str(code) in {"0", "200", "success", "SUCCESS"} or bool(payload.get("data")) if isinstance(payload, dict) else False
    return ProviderProbeResult(
        provider="runninghub",
        ok=ok,
        endpoint=endpoint,
        message=str(payload.get("msg", payload.get("message", "ok")) if isinstance(payload, dict) else "ok"),
        details=payload if isinstance(payload, dict) else {},
    )


def probe_runninghub_workflow_registry(
    config_path: str | None = None,
) -> ProviderProbeResult:
    """Check whether required RunningHub workflow IDs and API export JSON files are present."""

    configured_ids = resolve_runninghub_workflow_ids(config_path)
    api_format_dir = resolve_runninghub_api_format_dir(config_path)

    missing_ids: list[str] = []
    present_ids: list[str] = []
    missing_api_formats: list[str] = []
    present_api_formats: list[str] = []

    for profile in recommended_runninghub_workflows():
        env_name = profile.workflow_id_env
        if configured_ids.get(env_name):
            present_ids.append(env_name)
        else:
            missing_ids.append(env_name)

        expected_file = api_format_dir / f"{profile.workflow_key}.json" if api_format_dir else None
        if expected_file and expected_file.exists():
            present_api_formats.append(str(expected_file))
        else:
            missing_api_formats.append(profile.workflow_key)

    ok = not missing_ids and not missing_api_formats
    message_parts: list[str] = []
    if missing_ids:
        message_parts.append(f"missing_ids={len(missing_ids)}")
    if missing_api_formats:
        message_parts.append(f"missing_api_formats={len(missing_api_formats)}")
    if not message_parts:
        message_parts.append("configured")

    return ProviderProbeResult(
        provider="runninghub_workflows",
        ok=ok,
        endpoint=str(api_format_dir or ""),
        message=", ".join(message_parts),
        details={
            "present_ids": present_ids,
            "missing_ids": missing_ids,
            "api_format_dir": str(api_format_dir) if api_format_dir else "",
            "present_api_formats": present_api_formats,
            "missing_api_formats": missing_api_formats,
        },
    )


def probe_volcengine_ark_config(config_path: str | None = None) -> ProviderProbeResult:
    """Check whether Volcengine ARK credentials are present and normalized."""

    api_key, base_url = resolve_volcengine_ark_settings(config_path)
    return ProviderProbeResult(
        provider="volcengine_ark",
        ok=bool(api_key),
        endpoint=f"{base_url}/images/generations",
        message="configured" if api_key else "missing_api_key",
    )


def build_provider_probe_report(
    config_path: str | None = None,
    timeout_seconds: float = 30.0,
) -> ProviderProbeReport:
    """Build a small real-provider readiness report."""

    results = [
        probe_volcengine_ark_config(config_path=config_path),
        probe_runninghub_account(config_path=config_path, timeout_seconds=timeout_seconds),
        probe_runninghub_workflow_registry(config_path=config_path),
    ]
    return ProviderProbeReport(results=results)


def provider_probe_report_json(report: ProviderProbeReport, indent: int = 2) -> str:
    """Serialise a provider probe report."""

    return report.model_dump_json(indent=indent)


def write_provider_probe_report(report: ProviderProbeReport, output_dir: str | Path) -> dict[str, Path]:
    """Write provider probe outputs to disk."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / "provider_probe_report.json"
    json_path.write_text(provider_probe_report_json(report, indent=2), encoding="utf-8")
    return {"probe_report": json_path}


def main() -> int:
    """CLI entry point for probing provider readiness."""

    import argparse

    parser = argparse.ArgumentParser(description="Probe external provider readiness from local config.")
    parser.add_argument("--config-path", default=None, help="Path to config file")
    parser.add_argument("--output-dir", required=True, help="Directory for probe outputs")
    parser.add_argument("--timeout-seconds", type=float, default=30.0, help="HTTP timeout for live probes")
    args = parser.parse_args()

    report = build_provider_probe_report(config_path=args.config_path, timeout_seconds=args.timeout_seconds)
    output_files = write_provider_probe_report(report, args.output_dir)
    print(
        json.dumps(
            {
                "results": [result.model_dump(mode="json") for result in report.results],
                "output_files": {key: str(value) for key, value in output_files.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
