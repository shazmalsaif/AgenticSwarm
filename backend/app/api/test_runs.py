from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..agent.healing_agent import BrokenLocatorContext, HealingAgent, build_swarm_graph
from ..agent.planner_agent import list_available_models
from ..browser.playwright_runner import PlaywrightRunner
from ..browser.trace_manager import get_stream_manager, get_trace_manager
from .findings import record_finding_from_run
from .projects import _PROJECTS

router = APIRouter(prefix="/api/test-runs", tags=["Test Runs"])

TestRunStatus = Literal["passed", "healed", "failed"]


class TestRunCreateRequest(BaseModel):
    """Payload to launch an end-to-end LangGraph multi-agent test run."""

    project_id: Optional[str] = Field("proj-staging-demo", description="Target project ID")
    target_url: Optional[str] = Field(None, description="Override URL; defaults to project target_url when omitted")
    dom_summary: Optional[str] = Field(None, description="Optional DOM or route summary for planning")
    dom_snapshot: Optional[str] = Field(None, description="Optional HTML snapshot for navigation and ARIA healing")
    objectives: list[str] = Field(default_factory=list, description="Explicit QA objectives")
    persona_count: int = Field(3, ge=1, le=8, description="Number of synthetic QA personas")
    raw_failures: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Optional captured runtime failures to triage and heal during this run",
    )
    provider: Optional[str] = Field(None, description="Optional LangChain provider override (google_genai, openai, openrouter, deterministic)")
    model_name: Optional[str] = Field(None, description="Optional LangChain model override")
    use_live_browser: bool = Field(True, description="Fetch live site DOM and run Playwright browser")
    record_traces: bool = Field(True, description="Record execution traces and visual step frames")
    stream_id: Optional[str] = Field(None, description="Client stream ID for live SSE frame streaming")


class QuickHealRequest(BaseModel):
    """Standalone request payload to synthesize resilient locators for a broken selector."""

    failed_selector: str = Field(..., min_length=1)
    action_description: Optional[str] = None
    element_role: Optional[str] = None
    accessible_name: Optional[str] = None
    element_text: Optional[str] = None
    dom_excerpt: Optional[str] = None
    target_url: str = ""
    error_message: Optional[str] = None
    provider: Optional[str] = None
    model_name: Optional[str] = None


class TestRunResponse(BaseModel):
    """Complete execution artifact produced by the 5-agent LangGraph swarm."""

    id: str
    project_id: Optional[str] = None
    target_url: str
    status: TestRunStatus
    provider: Optional[str] = None
    model_name: Optional[str] = None
    scenarios_count: int
    personas_count: int
    failures_count: int
    healed_count: int
    finding_ids: list[str] = Field(default_factory=list)
    test_plan: dict[str, Any]
    personas: dict[str, Any]
    navigation_results: list[dict[str, Any]]
    trace_artifacts: list[dict[str, Any]] = Field(default_factory=list)
    video_urls: list[str] = Field(default_factory=list)
    triage_report: dict[str, Any]
    healing_report: Optional[dict[str, Any]] = None
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


_STORE_LOCK = Lock()
_TEST_RUNS: dict[str, TestRunResponse] = {}


def _get_or_404(run_id: str) -> TestRunResponse:
    with _STORE_LOCK:
        run = _TEST_RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Test run '{run_id}' not found")
    return run


def _resolve_target_url(project_id: Optional[str], explicit_url: Optional[str]) -> str:
    if explicit_url and explicit_url.strip():
        url = explicit_url.strip()
        return url if url.startswith(("http://", "https://")) else f"https://{url}"
    if project_id and project_id in _PROJECTS:
        return _PROJECTS[project_id].target_url
    return "https://staging.store.nexus"


def _resolve_model_choice(provider: Optional[str], model_name: Optional[str]) -> tuple[str, str]:
    catalog = list_available_models()
    resolved_provider = (provider or "").strip()
    resolved_model = (model_name or "").strip()

    if resolved_model and not resolved_provider and ":" in resolved_model:
        prefix, rest = resolved_model.split(":", 1)
        resolved_provider = prefix.strip()
        resolved_model = rest.strip()

    if not resolved_provider:
        resolved_provider = str(catalog["default_provider"])
    if not resolved_model:
        resolved_model = str(catalog["default_model"])
    return resolved_provider, resolved_model


def _execute_swarm_run(payload: TestRunCreateRequest) -> TestRunResponse:
    run_id = f"run-{uuid4().hex[:8]}"
    resolved_url = _resolve_target_url(payload.project_id, payload.target_url)
    resolved_provider, resolved_model = _resolve_model_choice(payload.provider, payload.model_name)
    stream_mgr = get_stream_manager()

    if payload.stream_id:
        stream_mgr.init_stream(
            payload.stream_id,
            target_url=resolved_url,
            provider=resolved_provider,
            model_name=resolved_model,
            run_id=run_id,
        )

    dom_snapshot = payload.dom_snapshot
    dom_summary = payload.dom_summary
    if not dom_snapshot and not dom_summary:
        captured = PlaywrightRunner().capture_dom_snapshot(
            target_url=resolved_url,
            use_live_browser=payload.use_live_browser,
            stream_id=payload.stream_id,
        )
        dom_snapshot = captured.get("dom_snapshot")
        dom_summary = captured.get("dom_summary") or dom_snapshot

    swarm = build_swarm_graph(provider=resolved_provider, model_name=resolved_model)
    initial_state: dict[str, Any] = {
        "target_url": resolved_url,
        "dom_summary": dom_summary or dom_snapshot,
        "dom_snapshot": dom_snapshot or dom_summary,
        "objectives": list(payload.objectives),
        "persona_count": payload.persona_count,
        "use_live_browser": payload.use_live_browser,
        "record_traces": payload.record_traces,
        "run_id": run_id,
        "stream_id": payload.stream_id,
    }
    if payload.raw_failures:
        initial_state["raw_failures"] = [dict(f) for f in payload.raw_failures]

    try:
        final_state = swarm.invoke(initial_state)
    except Exception:
        if payload.stream_id:
            stream_mgr.complete_stream(
                payload.stream_id,
                run_id=run_id,
                status="failed",
                message="AgentSwarm execution failed.",
            )
        raise

    test_plan = final_state.get("test_plan") or {}
    personas = final_state.get("personas") or {}
    navigation_results = final_state.get("navigation_results") or []
    triage_report = final_state.get("triage_report") or {}
    healing_report = final_state.get("healing_report")

    raw_failures: list[dict[str, Any]] = list(final_state.get("raw_failures") or [])
    diagnoses: list[dict[str, Any]] = list(triage_report.get("diagnoses") or [])
    suggestions: list[dict[str, Any]] = list((healing_report or {}).get("suggestions") or [])
    suggestions_by_id = {s.get("failure_id"): s for s in suggestions if isinstance(s, dict) and s.get("failure_id")}

    finding_ids: list[str] = []
    for idx, failure in enumerate(raw_failures):
        diag = diagnoses[idx] if idx < len(diagnoses) else None
        fid = (diag or {}).get("failure_id") or failure.get("failure_id")
        patch = suggestions_by_id.get(fid)
        recorded = record_finding_from_run(
            failure,
            project_id=payload.project_id,
            test_run_id=run_id,
            diagnosis=diag,
            healing_patch=patch,
        )
        finding_ids.append(recorded.id)

    failures_count = len(raw_failures)
    healed_count = len(suggestions)
    if failures_count == 0:
        run_status: TestRunStatus = "passed"
    elif healed_count > 0 and healed_count == failures_count:
        run_status = "healed"
    else:
        run_status = "failed"

    trace_artifacts = [t.model_dump() for t in get_trace_manager().list_traces(run_id=run_id)]
    video_urls = [
        str(nr["video_url"])
        for nr in navigation_results
        if isinstance(nr, dict) and nr.get("video_url")
    ]
    for ta in trace_artifacts:
        if ta.get("video_url") and ta["video_url"] not in video_urls:
            video_urls.append(str(ta["video_url"]))

    if payload.stream_id:
        stream_mgr.complete_stream(
            payload.stream_id,
            run_id=run_id,
            status="completed",
            video_urls=video_urls,
            message=f"Completed run {run_id} ({run_status.upper()})",
        )

    config_snapshot = payload.model_dump()
    if dom_snapshot and not config_snapshot.get("dom_snapshot"):
        config_snapshot["dom_snapshot"] = dom_snapshot[:12000]

    plan_gen_by = str(test_plan.get("generated_by") or "")
    actual_model = resolved_model
    if plan_gen_by.startswith("langchain:") and ":" in plan_gen_by:
        actual_model = plan_gen_by.split(":", 1)[1].strip() or resolved_model

    run_record = TestRunResponse(
        id=run_id,
        project_id=payload.project_id,
        target_url=resolved_url,
        status=run_status,
        provider=resolved_provider,
        model_name=actual_model,
        scenarios_count=len(test_plan.get("scenarios") or []),
        personas_count=len(personas.get("personas") or []),
        failures_count=failures_count,
        healed_count=healed_count,
        finding_ids=finding_ids,
        test_plan=test_plan,
        personas=personas,
        navigation_results=navigation_results,
        trace_artifacts=trace_artifacts,
        video_urls=video_urls,
        triage_report=triage_report,
        healing_report=healing_report,
        config_snapshot=config_snapshot,
        created_at=datetime.now(timezone.utc),
    )
    with _STORE_LOCK:
        _TEST_RUNS[run_id] = run_record
    return run_record


@router.get("/models", summary="List Available AI Models")
def get_available_models() -> dict[str, Any]:
    """Return selectable LLM providers and models along with API key readiness."""
    return list_available_models()


@router.get("/stream-state/{stream_id}", summary="Get Live Stream State")
def get_live_stream_state(stream_id: str) -> dict[str, Any]:
    """Poll current stage progress and streamed screenshot frames for an active or completed run."""
    state = get_stream_manager().get_state(stream_id)
    if state is None:
        return {
            "stream_id": stream_id,
            "status": "pending",
            "current_stage": "planner",
            "stage_index": 0,
            "stage_message": "Initializing AgentSwarm stream...",
            "frames": [],
            "logs": [],
            "partial_results": {},
            "video_urls": [],
        }
    return state


@router.get("/stream/{stream_id}", summary="Stream Live Browser Screenshots (SSE)")
async def stream_test_run(stream_id: str):
    """Server-Sent Events endpoint streaming live agent stages and browser step screenshots."""

    async def _event_generator():
        stream_mgr = get_stream_manager()
        cursor = 0
        idle_ticks = 0
        while idle_ticks < 240:  # up to ~60s at 0.25s poll
            events, st_status = stream_mgr.get_events_since(stream_id, cursor)
            if events:
                idle_ticks = 0
                for ev in events:
                    cursor += 1
                    ev_type = ev.get("type", "message")
                    payload_json = json.dumps(ev.get("data", {}))
                    yield f"event: {ev_type}\ndata: {payload_json}\n\n"
                    if ev_type == "complete":
                        return
            else:
                if st_status in ("completed", "failed"):
                    return
                idle_ticks += 1
            await asyncio.sleep(0.25)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/videos/{filename}", summary="Serve Recorded Browser Video")
def get_recorded_video(filename: str):
    """Serve a Playwright-recorded .webm video artifact safely by filename."""
    safe_name = Path(filename).name
    video_path = get_trace_manager().artifacts_dir / "videos" / safe_name
    if not video_path.exists() or not video_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Video '{safe_name}' not found")
    return FileResponse(path=str(video_path), media_type="video/webm", filename=safe_name)


@router.get("", response_model=list[TestRunResponse], summary="List Test Runs")
def list_test_runs(
    project_id: Optional[str] = Query(None),
    run_status: Optional[TestRunStatus] = Query(None, alias="status"),
) -> list[TestRunResponse]:
    """List executed swarm test runs, optionally filtered by project_id or status."""
    with _STORE_LOCK:
        items = list(_TEST_RUNS.values())
    if project_id:
        items = [r for r in items if r.project_id == project_id]
    if run_status:
        items = [r for r in items if r.status == run_status]
    return sorted(items, key=lambda r: r.created_at, reverse=True)


@router.post("", response_model=TestRunResponse, status_code=status.HTTP_201_CREATED, summary="Execute AgentSwarm Run")
def create_test_run(payload: TestRunCreateRequest) -> TestRunResponse:
    """Launch the full LangGraph multi-agent swarm workflow and persist discovered findings."""
    return _execute_swarm_run(payload)


@router.post("/heal-locator", summary="Synthesize Self-Healing Locator Patches")
def quick_heal_locator(payload: QuickHealRequest) -> dict[str, Any]:
    """Standalone endpoint invoking HealingAgent to synthesize Playwright, Cypress, and Selenium patches."""
    agent = HealingAgent(provider=payload.provider, model_name=payload.model_name)
    suggestion = agent.heal_locator(
        BrokenLocatorContext(
            failed_selector=payload.failed_selector,
            action_description=payload.action_description,
            element_role=payload.element_role,
            accessible_name=payload.accessible_name,
            element_text=payload.element_text,
            dom_excerpt=payload.dom_excerpt,
            target_url=payload.target_url,
            error_message=payload.error_message,
        )
    )
    return suggestion.model_dump()


@router.get("/{run_id}", response_model=TestRunResponse, summary="Get Test Run")
def get_test_run(run_id: str) -> TestRunResponse:
    """Retrieve complete results and artifacts of a specific test run."""
    return _get_or_404(run_id)


@router.post("/{run_id}/rerun", response_model=TestRunResponse, status_code=status.HTTP_201_CREATED, summary="Re-run Test Run")
def rerun_test_run(run_id: str) -> TestRunResponse:
    """Re-execute an existing test run using its saved configuration."""
    existing = _get_or_404(run_id)
    replay_payload = TestRunCreateRequest.model_validate(existing.config_snapshot)
    return _execute_swarm_run(replay_payload)


@router.delete("/{run_id}", summary="Delete Test Run")
def delete_test_run(run_id: str) -> dict[str, Any]:
    """Delete a test run record."""
    _get_or_404(run_id)
    with _STORE_LOCK:
        _TEST_RUNS.pop(run_id, None)
    return {"deleted": True, "test_run_id": run_id}
