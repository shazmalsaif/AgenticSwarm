from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any, Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, Body, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..agent.healing_agent import BrokenLocatorContext, HealingAgent
from ..agent.triage_agent import FailureEvidence, TriageAgent

router = APIRouter(prefix="/api/findings", tags=["Findings"])

FindingCategory = Literal[
    "SELECTOR_DRIFT",
    "BACKEND_API_ERROR",
    "NETWORK_TIMEOUT",
    "ASSERTION_FAILURE",
    "CONSOLE_EXCEPTION",
]
FindingSeverity = Literal["critical", "high", "medium", "low"]
FindingStatus = Literal["open", "healed", "acknowledged", "ignored", "resolved"]


class FindingCreateRequest(BaseModel):
    """Payload to ingest failure telemetry, triage root cause, and optionally synthesize locator patches."""

    project_id: Optional[str] = Field("proj-staging-demo", description="Associated project ID")
    test_run_id: Optional[str] = Field(None, description="Associated test run ID if part of a run")
    scenario_id: Optional[str] = None
    scenario_title: Optional[str] = None
    step_number: Optional[int] = None
    target_url: str = Field("", description="Page URL where the failure occurred")
    error_message: str = Field(..., min_length=1, description="Captured error or timeout message")
    failed_selector: Optional[str] = Field(None, description="Broken CSS/XPath selector if applicable")
    http_status: Optional[int] = Field(None, description="Observed HTTP status code around failure")
    console_errors: list[str] = Field(default_factory=list)
    network_failures: list[str] = Field(default_factory=list)
    dom_excerpt: Optional[str] = Field(None, description="HTML fragment around the failure state")
    action_description: Optional[str] = Field(None, description="Step intent when failure happened")


class FindingStatusUpdate(BaseModel):
    """Update status or resolution notes for an existing finding."""

    status: FindingStatus
    resolution_notes: Optional[str] = Field(None, max_length=1000)


class FindingHealRequest(BaseModel):
    """Optional overrides when triggering self-healing on an existing finding."""

    failed_selector: Optional[str] = None
    dom_excerpt: Optional[str] = None
    element_role: Optional[str] = None
    accessible_name: Optional[str] = None
    element_text: Optional[str] = None


class FindingResponse(BaseModel):
    """Full bug finding with TriageAgent diagnosis and HealingAgent code patches."""

    id: str
    project_id: Optional[str] = None
    test_run_id: Optional[str] = None
    scenario_id: Optional[str] = None
    scenario_title: Optional[str] = None
    step_number: Optional[int] = None
    target_url: str
    error_message: str
    failed_selector: Optional[str] = None
    http_status: Optional[int] = None
    category: FindingCategory
    severity: FindingSeverity
    confidence: float
    root_cause_summary: str
    verdict: str
    needs_healing: bool
    recommended_next_agent: str
    evidence_highlights: list[str] = Field(default_factory=list)
    console_errors: list[str] = Field(default_factory=list)
    network_failures: list[str] = Field(default_factory=list)
    dom_excerpt: Optional[str] = None
    action_description: Optional[str] = None
    healing_patch: Optional[dict[str, Any]] = None
    status: FindingStatus
    resolution_notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime


_STORE_LOCK = Lock()
_FINDINGS: dict[str, FindingResponse] = {}


def _get_or_404(finding_id: str) -> FindingResponse:
    with _STORE_LOCK:
        finding = _FINDINGS.get(finding_id)
    if not finding:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Finding '{finding_id}' not found")
    return finding


def record_finding_from_run(
    evidence: FailureEvidence | dict[str, Any],
    *,
    project_id: Optional[str] = None,
    test_run_id: Optional[str] = None,
    diagnosis: Optional[dict[str, Any]] = None,
    healing_patch: Optional[dict[str, Any]] = None,
) -> FindingResponse:
    """
    Create and store a structured finding from a failure event.

    Uses supplied `diagnosis`/`healing_patch` when already computed by the LangGraph swarm,
    otherwise invokes TriageAgent (and HealingAgent when eligible).
    """
    ev = evidence if isinstance(evidence, FailureEvidence) else FailureEvidence.model_validate(evidence)

    if diagnosis is None:
        diag_obj = TriageAgent().triage_failure(ev)
        diag_dict = diag_obj.model_dump()
    else:
        diag_dict = dict(diagnosis)

    needs_healing = bool(diag_dict.get("needs_healing"))
    resolved_patch = dict(healing_patch) if healing_patch else None

    if resolved_patch is None and needs_healing and ev.failed_selector:
        suggestion = HealingAgent().heal_locator(
            BrokenLocatorContext(
                failure_id=ev.failure_id,
                failed_selector=ev.failed_selector,
                action_description=ev.action_description,
                dom_excerpt=ev.dom_excerpt,
                target_url=ev.target_url,
                error_message=ev.error_message,
            )
        )
        resolved_patch = suggestion.model_dump()

    now = datetime.now(timezone.utc)
    finding = FindingResponse(
        id=f"fnd-{uuid4().hex[:8]}",
        project_id=project_id,
        test_run_id=test_run_id,
        scenario_id=ev.scenario_id,
        scenario_title=ev.scenario_title,
        step_number=ev.step_number,
        target_url=ev.target_url,
        error_message=ev.error_message,
        failed_selector=ev.failed_selector,
        http_status=ev.http_status,
        category=diag_dict["category"],
        severity=diag_dict["severity"],
        confidence=float(diag_dict.get("confidence", 0.85)),
        root_cause_summary=str(diag_dict.get("root_cause_summary", ev.error_message)),
        verdict=str(diag_dict.get("verdict", "")),
        needs_healing=needs_healing,
        recommended_next_agent=str(diag_dict.get("recommended_next_agent", "human_review")),
        evidence_highlights=list(diag_dict.get("evidence_highlights") or []),
        console_errors=list(ev.console_errors),
        network_failures=list(ev.network_failures),
        dom_excerpt=ev.dom_excerpt,
        action_description=ev.action_description,
        healing_patch=resolved_patch,
        status="healed" if resolved_patch is not None else "open",
        resolution_notes=None,
        created_at=now,
        updated_at=now,
    )
    with _STORE_LOCK:
        _FINDINGS[finding.id] = finding
    return finding


@router.get("", response_model=list[FindingResponse], summary="List Findings")
def list_findings(
    test_run_id: Optional[str] = Query(None),
    project_id: Optional[str] = Query(None),
    severity: Optional[FindingSeverity] = Query(None),
    category: Optional[FindingCategory] = Query(None),
    finding_status: Optional[FindingStatus] = Query(None, alias="status"),
    needs_healing: Optional[bool] = Query(None),
) -> list[FindingResponse]:
    """List triaged QA findings with optional multi-field filters."""
    with _STORE_LOCK:
        items = list(_FINDINGS.values())

    if test_run_id:
        items = [f for f in items if f.test_run_id == test_run_id]
    if project_id:
        items = [f for f in items if f.project_id == project_id]
    if severity:
        items = [f for f in items if f.severity == severity]
    if category:
        items = [f for f in items if f.category == category]
    if finding_status:
        items = [f for f in items if f.status == finding_status]
    if needs_healing is not None:
        items = [f for f in items if f.needs_healing == needs_healing]

    return sorted(items, key=lambda f: f.created_at, reverse=True)


@router.post("", response_model=FindingResponse, status_code=status.HTTP_201_CREATED, summary="Triage & Record Finding")
def create_finding(payload: FindingCreateRequest) -> FindingResponse:
    """Ingest failure evidence, classify root cause via TriageAgent, and synthesize healing patches if eligible."""
    evidence = FailureEvidence(
        scenario_id=payload.scenario_id,
        scenario_title=payload.scenario_title,
        step_number=payload.step_number,
        target_url=payload.target_url,
        error_message=payload.error_message,
        failed_selector=payload.failed_selector,
        http_status=payload.http_status,
        console_errors=payload.console_errors,
        network_failures=payload.network_failures,
        dom_excerpt=payload.dom_excerpt,
        action_description=payload.action_description,
    )
    return record_finding_from_run(
        evidence,
        project_id=payload.project_id,
        test_run_id=payload.test_run_id,
    )


@router.get("/{finding_id}", response_model=FindingResponse, summary="Get Finding Details")
def get_finding(finding_id: str) -> FindingResponse:
    """Retrieve a single finding including diagnosis metadata and framework healing snippets."""
    return _get_or_404(finding_id)


@router.patch("/{finding_id}", response_model=FindingResponse, summary="Update Finding Status")
def update_finding_status(finding_id: str, payload: FindingStatusUpdate) -> FindingResponse:
    """Update a finding's workflow status and optional resolution notes."""
    existing = _get_or_404(finding_id)
    updates: dict[str, Any] = {
        "status": payload.status,
        "updated_at": datetime.now(timezone.utc),
    }
    if payload.resolution_notes is not None:
        updates["resolution_notes"] = payload.resolution_notes

    updated = existing.model_copy(update=updates)
    with _STORE_LOCK:
        _FINDINGS[finding_id] = updated
    return updated


@router.post("/{finding_id}/heal", response_model=FindingResponse, summary="Trigger HealingAgent on Finding")
def heal_finding(
    finding_id: str,
    payload: Optional[FindingHealRequest] = Body(default=None),
) -> FindingResponse:
    """Trigger or refresh HealingAgent locator synthesis for a finding."""
    existing = _get_or_404(finding_id)
    opts = payload or FindingHealRequest()
    selector = (opts.failed_selector or existing.failed_selector or "").strip()
    if not selector:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Finding does not have a failed_selector to heal.",
        )

    suggestion = HealingAgent().heal_locator(
        BrokenLocatorContext(
            failure_id=existing.id,
            failed_selector=selector,
            action_description=existing.action_description,
            element_role=opts.element_role,
            accessible_name=opts.accessible_name,
            element_text=opts.element_text,
            dom_excerpt=opts.dom_excerpt or existing.dom_excerpt,
            target_url=existing.target_url,
            error_message=existing.error_message,
        )
    )
    updated = existing.model_copy(
        update={
            "failed_selector": selector,
            "dom_excerpt": opts.dom_excerpt or existing.dom_excerpt,
            "healing_patch": suggestion.model_dump(),
            "status": "healed",
            "updated_at": datetime.now(timezone.utc),
        }
    )
    with _STORE_LOCK:
        _FINDINGS[finding_id] = updated
    return updated


@router.delete("/{finding_id}", summary="Delete Finding")
def delete_finding(finding_id: str) -> dict[str, Any]:
    """Remove a finding record."""
    _get_or_404(finding_id)
    with _STORE_LOCK:
        _FINDINGS.pop(finding_id, None)
    return {"deleted": True, "finding_id": finding_id}
