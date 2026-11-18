from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any, Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, Body, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..agent.persona_agent import PersonaAgent
from ..agent.planner_agent import PlannerAgent, list_available_models
from ..browser.playwright_runner import PlaywrightRunner

router = APIRouter(prefix="/api/projects", tags=["Projects"])


class ProjectCreate(BaseModel):
    """Payload to register a target web application project in AgentSwarm."""

    name: str = Field(..., min_length=2, max_length=120, description="Project display name")
    target_url: str = Field(..., min_length=3, description="Base staging or preview URL under test")
    description: Optional[str] = Field(None, max_length=500, description="Optional project scope summary")
    environment: Literal["staging", "preview", "local", "production"] = "staging"
    default_framework: Literal["playwright", "cypress", "selenium"] = "playwright"


class ProjectUpdate(BaseModel):
    """Partial update payload for project configuration."""

    name: Optional[str] = Field(None, min_length=2, max_length=120)
    target_url: Optional[str] = Field(None, min_length=3)
    description: Optional[str] = Field(None, max_length=500)
    environment: Optional[Literal["staging", "preview", "local", "production"]] = None
    default_framework: Optional[Literal["playwright", "cypress", "selenium"]] = None
    status: Optional[Literal["active", "paused", "archived"]] = None


class ProjectResponse(BaseModel):
    """Serialized project record returned by the Projects API."""

    id: str
    name: str
    target_url: str
    description: Optional[str] = None
    environment: Literal["staging", "preview", "local", "production"]
    default_framework: Literal["playwright", "cypress", "selenium"]
    status: Literal["active", "paused", "archived"]
    created_at: datetime
    updated_at: datetime


class ProjectPlanPreviewRequest(BaseModel):
    """Optional overrides when previewing a project test plan and personas."""

    dom_summary: Optional[str] = Field(None, description="Optional DOM or route snapshot")
    objectives: Optional[list[str]] = Field(None, description="Specific QA focus areas")
    persona_count: int = Field(3, ge=1, le=8, description="Number of synthetic personas to generate")
    provider: Optional[str] = Field(None, description="Optional LangChain provider override")
    model_name: Optional[str] = Field(None, description="Optional LangChain model override")
    use_live_browser: bool = Field(True, description="Fetch live DOM snapshot before planning")


class ProjectPlanPreviewResponse(BaseModel):
    """Preview combining PlannerAgent scenarios and PersonaAgent profiles."""

    project_id: str
    target_url: str
    provider: Optional[str] = None
    model_name: Optional[str] = None
    test_plan: dict[str, Any]
    personas: dict[str, Any]
    generated_at: datetime


_STORE_LOCK = Lock()
_NOW = datetime.now(timezone.utc)
_PROJECTS: dict[str, ProjectResponse] = {
    "proj-staging-demo": ProjectResponse(
        id="proj-staging-demo",
        name="Nexus Store Staging",
        target_url="https://staging.store.nexus",
        description="Default staging e-commerce sandbox for autonomous E2E checkout verification.",
        environment="staging",
        default_framework="playwright",
        status="active",
        created_at=_NOW,
        updated_at=_NOW,
    )
}


def _normalize_url(raw_url: str) -> str:
    url = raw_url.strip()
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    return url


def _get_or_404(project_id: str) -> ProjectResponse:
    with _STORE_LOCK:
        project = _PROJECTS.get(project_id)
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project '{project_id}' not found")
    return project


@router.get("", response_model=list[ProjectResponse], summary="List Projects")
def list_projects(
    environment: Optional[Literal["staging", "preview", "local", "production"]] = Query(None),
    project_status: Optional[Literal["active", "paused", "archived"]] = Query(None, alias="status"),
) -> list[ProjectResponse]:
    """List registered projects, optionally filtered by environment or status."""
    with _STORE_LOCK:
        items = list(_PROJECTS.values())
    if environment:
        items = [p for p in items if p.environment == environment]
    if project_status:
        items = [p for p in items if p.status == project_status]
    return sorted(items, key=lambda p: p.created_at, reverse=True)


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED, summary="Create Project")
def create_project(payload: ProjectCreate) -> ProjectResponse:
    """Create a new target application project."""
    now = datetime.now(timezone.utc)
    project = ProjectResponse(
        id=f"proj-{uuid4().hex[:8]}",
        name=payload.name.strip(),
        target_url=_normalize_url(payload.target_url),
        description=payload.description.strip() if payload.description else None,
        environment=payload.environment,
        default_framework=payload.default_framework,
        status="active",
        created_at=now,
        updated_at=now,
    )
    with _STORE_LOCK:
        _PROJECTS[project.id] = project
    return project


@router.get("/{project_id}", response_model=ProjectResponse, summary="Get Project")
def get_project(project_id: str) -> ProjectResponse:
    """Fetch a single project by its ID."""
    return _get_or_404(project_id)


@router.patch("/{project_id}", response_model=ProjectResponse, summary="Update Project")
def update_project(project_id: str, payload: ProjectUpdate) -> ProjectResponse:
    """Update project configuration fields."""
    existing = _get_or_404(project_id)
    updates = payload.model_dump(exclude_unset=True)
    if "target_url" in updates and updates["target_url"]:
        updates["target_url"] = _normalize_url(updates["target_url"])
    if "name" in updates and updates["name"]:
        updates["name"] = updates["name"].strip()
    updates["updated_at"] = datetime.now(timezone.utc)

    updated = existing.model_copy(update=updates)
    with _STORE_LOCK:
        _PROJECTS[project_id] = updated
    return updated


@router.delete("/{project_id}", summary="Delete Project")
def delete_project(project_id: str) -> dict[str, Any]:
    """Delete a project by ID."""
    _get_or_404(project_id)
    with _STORE_LOCK:
        _PROJECTS.pop(project_id, None)
    return {"deleted": True, "project_id": project_id}


@router.post(
    "/{project_id}/generate-plan",
    response_model=ProjectPlanPreviewResponse,
    summary="Generate Test Plan & Personas Preview",
)
def generate_project_plan(
    project_id: str,
    payload: Optional[ProjectPlanPreviewRequest] = Body(default=None),
) -> ProjectPlanPreviewResponse:
    """Invoke PlannerAgent and PersonaAgent to generate a test strategy preview for the project."""
    project = _get_or_404(project_id)
    opts = payload or ProjectPlanPreviewRequest()

    catalog = list_available_models()
    resolved_provider = (opts.provider or "").strip()
    resolved_model = (opts.model_name or "").strip()
    if resolved_model and not resolved_provider and ":" in resolved_model:
        prefix, rest = resolved_model.split(":", 1)
        resolved_provider = prefix.strip()
        resolved_model = rest.strip()
    if not resolved_provider:
        resolved_provider = str(catalog["default_provider"])
    if not resolved_model:
        resolved_model = str(catalog["default_model"])

    dom_summary = opts.dom_summary
    if not dom_summary:
        captured = PlaywrightRunner().capture_dom_snapshot(
            target_url=project.target_url,
            use_live_browser=opts.use_live_browser,
        )
        dom_summary = captured.get("dom_summary") or captured.get("dom_snapshot")

    planner = PlannerAgent(provider=resolved_provider, model_name=resolved_model)
    plan = planner.create_test_plan(
        target_url=project.target_url,
        dom_summary=dom_summary,
        objectives=opts.objectives,
    )

    persona_agent = PersonaAgent(provider=resolved_provider, model_name=resolved_model)
    personas = persona_agent.generate_personas(
        target_url=project.target_url,
        app_context=dom_summary or project.description,
        test_plan=plan,
        count=opts.persona_count,
    )

    return ProjectPlanPreviewResponse(
        project_id=project.id,
        target_url=project.target_url,
        provider=resolved_provider,
        model_name=resolved_model,
        test_plan=plan.model_dump(),
        personas=personas.model_dump(),
        generated_at=datetime.now(timezone.utc),
    )
