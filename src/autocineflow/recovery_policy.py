"""Strategy-based recovery planning for submission and render failures."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from .submission import SubmissionBatch, SubmissionRecord


class RecoveryAction(str, Enum):
    """High-level recovery actions."""

    RETRY = "RETRY"
    PAUSE_QUEUE = "PAUSE_QUEUE"
    MANUAL_FIX = "MANUAL_FIX"
    IGNORE = "IGNORE"


class RecoverySeverity(str, Enum):
    """Severity of a failure class."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RecoveryDecision(BaseModel):
    """Decision for a single failed submission record."""

    job_id: str
    shot_id: str
    action: RecoveryAction
    severity: RecoverySeverity
    reason: str
    provider_status_code: int = 0
    provider_status_message: str = ""
    retry_allowed: bool = False


class RecoveryPlan(BaseModel):
    """Aggregated recovery plan for a submission batch."""

    source_id: str
    generated_at: str
    decision_count: int = Field(..., ge=0)
    decisions: list[RecoveryDecision] = Field(default_factory=list)
    queue_paused: bool = False
    retry_job_ids: list[str] = Field(default_factory=list)


def classify_submission_record(record: SubmissionRecord) -> RecoveryDecision | None:
    """Classify a submission record into a recovery decision when needed."""

    code = record.provider_status_code
    provider_message = record.provider_status_message
    raw_message = (record.message or "").strip()
    if (code == 0 and not provider_message) and raw_message.startswith("{"):
        try:
            payload = json.loads(raw_message)
            base_resp = payload.get("base_resp", {})
            if isinstance(base_resp, dict):
                code = int(base_resp.get("status_code", 0) or 0)
                provider_message = str(base_resp.get("status_msg", "") or "")
        except Exception:  # noqa: BLE001
            pass

    message = (provider_message or raw_message).lower()
    backend_missing = not record.backend_job_id

    if code == 0 and not backend_missing:
        return None
    if "insufficient balance" in message or code == 1008:
        return RecoveryDecision(
            job_id=record.job_id,
            shot_id=record.shot_id,
            action=RecoveryAction.PAUSE_QUEUE,
            severity=RecoverySeverity.CRITICAL,
            reason="provider_balance_exhausted",
            provider_status_code=code,
            provider_status_message=provider_message or record.provider_status_message,
            retry_allowed=False,
        )
    if "invalid params" in message or code == 2013:
        return RecoveryDecision(
            job_id=record.job_id,
            shot_id=record.shot_id,
            action=RecoveryAction.MANUAL_FIX,
            severity=RecoverySeverity.HIGH,
            reason="provider_rejected_parameters",
            provider_status_code=code,
            provider_status_message=provider_message or record.provider_status_message,
            retry_allowed=False,
        )
    if "unexpected error" in message or code == 1000:
        return RecoveryDecision(
            job_id=record.job_id,
            shot_id=record.shot_id,
            action=RecoveryAction.RETRY,
            severity=RecoverySeverity.MEDIUM,
            reason="provider_transient_error",
            provider_status_code=code,
            provider_status_message=provider_message or record.provider_status_message,
            retry_allowed=True,
        )
    if backend_missing:
        return RecoveryDecision(
            job_id=record.job_id,
            shot_id=record.shot_id,
            action=RecoveryAction.RETRY,
            severity=RecoverySeverity.MEDIUM,
            reason="missing_backend_job_id",
            provider_status_code=code,
            provider_status_message=provider_message or record.provider_status_message,
            retry_allowed=True,
        )
    return None


def build_recovery_plan(batch: SubmissionBatch) -> RecoveryPlan:
    """Build a strategy-based recovery plan from a submission batch."""

    decisions = [
        decision
        for record in batch.records
        if (decision := classify_submission_record(record)) is not None
    ]
    queue_paused = any(decision.action == RecoveryAction.PAUSE_QUEUE for decision in decisions)
    retry_job_ids = [decision.job_id for decision in decisions if decision.retry_allowed]

    return RecoveryPlan(
        source_id=batch.source_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        decision_count=len(decisions),
        decisions=decisions,
        queue_paused=queue_paused,
        retry_job_ids=retry_job_ids,
    )


def recovery_plan_json(plan: RecoveryPlan, indent: int = 2) -> str:
    """Serialise a recovery plan to JSON."""

    return plan.model_dump_json(indent=indent)


def recovery_plan_markdown(plan: RecoveryPlan) -> str:
    """Export a human-readable recovery plan."""

    lines = [
        f"# Recovery Plan {plan.source_id}",
        "",
        f"- Decision Count: `{plan.decision_count}`",
        f"- Queue Paused: `{plan.queue_paused}`",
        f"- Retry Jobs: `{len(plan.retry_job_ids)}`",
        "",
        "## Decisions",
        "",
    ]
    for decision in plan.decisions:
        lines.extend(
            [
                f"### {decision.job_id}",
                "",
                f"- Shot: `{decision.shot_id}`",
                f"- Action: `{decision.action.value}`",
                f"- Severity: `{decision.severity.value}`",
                f"- Reason: `{decision.reason}`",
                f"- Provider Status: `{decision.provider_status_code}:{decision.provider_status_message}`",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def write_recovery_plan(plan: RecoveryPlan, output_dir: str | Path) -> dict[str, Path]:
    """Write recovery plan outputs to disk."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "recovery_plan.json"
    markdown_path = output_dir / "recovery_plan.md"
    json_path.write_text(recovery_plan_json(plan, indent=2), encoding="utf-8")
    markdown_path.write_text(recovery_plan_markdown(plan), encoding="utf-8")
    return {
        "plan_json": json_path,
        "plan_markdown": markdown_path,
    }


def main() -> int:
    """CLI entry point for recovery planning."""

    import argparse

    parser = argparse.ArgumentParser(description="Build a recovery plan from a submission batch.")
    parser.add_argument("--batch-file", required=True, help="Path to submission_batch.json")
    parser.add_argument("--output-dir", required=True, help="Directory for recovery plan outputs")
    args = parser.parse_args()

    batch = SubmissionBatch.model_validate_json(Path(args.batch_file).read_text(encoding="utf-8"))
    plan = build_recovery_plan(batch)
    output_files = write_recovery_plan(plan, args.output_dir)
    print(
        json.dumps(
            {
                "source_id": plan.source_id,
                "decision_count": plan.decision_count,
                "queue_paused": plan.queue_paused,
                "output_files": {key: str(value) for key, value in output_files.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
