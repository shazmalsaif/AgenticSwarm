from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any, Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, Body, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..agent.healing_agent import HealingAgent

router = APIRouter(prefix="/api/repositories", tags=["Repositories"])


class RepositoryConnectRequest(BaseModel):
    """Payload to connect a Git repository for test file indexing and PR previews."""

    project_id: Optional[str] = Field("proj-staging-demo", description="Associated AgentSwarm project ID")
    repo_url: str = Field(..., min_length=4, description="GitHub/Git repository URL (e.g., https://github.com/org/repo)")
    branch: str = Field("main", min_length=1, max_length=120)
    framework: Literal["playwright", "cypress", "selenium"] = "playwright"
    test_directory: str = Field("e2e", min_length=1, max_length=200)
    auto_pr_enabled: bool = Field(False, description="Require human review before opening automated PRs")


class RepositoryUpdateRequest(BaseModel):
    """Partial update payload for repository settings."""

    branch: Optional[str] = Field(None, min_length=1, max_length=120)
    framework: Optional[Literal["playwright", "cypress", "selenium"]] = None
    test_directory: Optional[str] = Field(None, min_length=1, max_length=200)
    auto_pr_enabled: Optional[bool] = None


class RepositoryIndexSummary(BaseModel):
    """Summary of indexed test specifications and selectors within a repository."""

    indexed: bool = False
    last_indexed_at: Optional[datetime] = None
    test_files_count: int = 0
    indexed_files: list[str] = Field(default_factory=list)
    detected_selectors_count: int = 0
    index_notes: str = "Not indexed yet"


class RepositoryResponse(BaseModel):
    """Serialized connected repository record."""

    id: str
    project_id: Optional[str] = None
    repo_url: str
    branch: str
    framework: Literal["playwright", "cypress", "selenium"]
    test_directory: str
    auto_pr_enabled: bool
    index_summary: RepositoryIndexSummary
    created_at: datetime
    updated_at: datetime


class RepositoryIndexRequest(BaseModel):
    """Optional hints when triggering repository test file indexing."""

    test_files: Optional[list[str]] = Field(None, description="Optional list of discovered spec paths to index")
    selectors_hint: Optional[list[str]] = Field(None, description="Optional brittle selectors detected during indexing")


class PullRequestPreviewRequest(BaseModel):
    """Request to build a unified git diff and PR preview for a healed locator."""

    file_path: Optional[str] = Field(None, description="Relative path to test file in repository")
    failed_selector: str = Field(..., min_length=1, description="Original brittle selector that failed")
    healed_selector: Optional[str] = Field(
        None,
        description="Optional explicit replacement selector; synthesized via HealingAgent if omitted",
    )
    action_description: Optional[str] = Field(None, description="Human-readable intent of the healed UI control")
    dom_excerpt: Optional[str] = Field(None, description="Optional DOM HTML excerpt around target element")


class PullRequestPreviewResponse(BaseModel):
    """Non-destructive git diff & Pull Request preview gated on human approval."""

    repository_id: str
    repo_url: str
    base_branch: str
    proposed_branch: str
    pr_title: str
    pr_body: str
    file_path: str
    framework: Literal["playwright", "cypress", "selenium"]
    failed_selector: str
    healed_selector: str
    unified_diff: str
    requires_human_approval: bool = True


_STORE_LOCK = Lock()
_REPOSITORIES: dict[str, RepositoryResponse] = {}


def _get_or_404(repo_id: str) -> RepositoryResponse:
    with _STORE_LOCK:
        repo = _REPOSITORIES.get(repo_id)
    if not repo:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Repository '{repo_id}' not found")
    return repo


def _default_spec_files(test_dir: str, framework: Literal["playwright", "cypress", "selenium"]) -> list[str]:
    clean_dir = test_dir.strip("/ ")
    if framework == "cypress":
        return [f"{clean_dir}/checkout-journey.cy.ts", f"{clean_dir}/auth-flow.cy.ts"]
    if framework == "selenium":
        return [f"{clean_dir}/test_checkout_journey.py", f"{clean_dir}/test_auth_flow.py"]
    return [f"{clean_dir}/checkout-journey.spec.ts", f"{clean_dir}/auth-flow.spec.ts"]


@router.get("", response_model=list[RepositoryResponse], summary="List Connected Repositories")
def list_repositories(project_id: Optional[str] = Query(None)) -> list[RepositoryResponse]:
    """List all connected repositories, optionally filtered by project_id."""
    with _STORE_LOCK:
        items = list(_REPOSITORIES.values())
    if project_id:
        items = [r for r in items if r.project_id == project_id]
    return sorted(items, key=lambda r: r.created_at, reverse=True)


@router.post(
    "",
    response_model=RepositoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Connect Repository",
)
def connect_repository(payload: RepositoryConnectRequest) -> RepositoryResponse:
    """Connect a Git repository to an AgentSwarm project."""
    now = datetime.now(timezone.utc)
    repo = RepositoryResponse(
        id=f"repo-{uuid4().hex[:8]}",
        project_id=payload.project_id,
        repo_url=payload.repo_url.strip(),
        branch=payload.branch.strip(),
        framework=payload.framework,
        test_directory=payload.test_directory.strip(),
        auto_pr_enabled=payload.auto_pr_enabled,
        index_summary=RepositoryIndexSummary(),
        created_at=now,
        updated_at=now,
    )
    with _STORE_LOCK:
        _REPOSITORIES[repo.id] = repo
    return repo


@router.get("/{repo_id}", response_model=RepositoryResponse, summary="Get Repository")
def get_repository(repo_id: str) -> RepositoryResponse:
    """Retrieve repository metadata and current indexing summary."""
    return _get_or_404(repo_id)


@router.patch("/{repo_id}", response_model=RepositoryResponse, summary="Update Repository")
def update_repository(repo_id: str, payload: RepositoryUpdateRequest) -> RepositoryResponse:
    """Update repository branch, framework, test directory, or PR preferences."""
    existing = _get_or_404(repo_id)
    updates = payload.model_dump(exclude_unset=True)
    updates["updated_at"] = datetime.now(timezone.utc)
    updated = existing.model_copy(update=updates)
    with _STORE_LOCK:
        _REPOSITORIES[repo_id] = updated
    return updated


@router.delete("/{repo_id}", summary="Disconnect Repository")
def disconnect_repository(repo_id: str) -> dict[str, Any]:
    """Disconnect and remove a repository record."""
    _get_or_404(repo_id)
    with _STORE_LOCK:
        _REPOSITORIES.pop(repo_id, None)
    return {"deleted": True, "repository_id": repo_id}


@router.post("/{repo_id}/index", response_model=RepositoryResponse, summary="Index Repository Specs")
def index_repository(
    repo_id: str,
    payload: Optional[RepositoryIndexRequest] = Body(default=None),
) -> RepositoryResponse:
    """Index repository test files and selectors for healing and diagnosis context."""
    existing = _get_or_404(repo_id)
    req = payload or RepositoryIndexRequest()
    files = req.test_files or _default_spec_files(existing.test_directory, existing.framework)
    selectors_count = len(req.selectors_hint) if req.selectors_hint else max(len(files) * 6, 4)

    now = datetime.now(timezone.utc)
    summary = RepositoryIndexSummary(
        indexed=True,
        last_indexed_at=now,
        test_files_count=len(files),
        indexed_files=files,
        detected_selectors_count=selectors_count,
        index_notes=f"Indexed {len(files)} {existing.framework} spec files in '{existing.test_directory}' on branch '{existing.branch}'.",
    )
    updated = existing.model_copy(update={"index_summary": summary, "updated_at": now})
    with _STORE_LOCK:
        _REPOSITORIES[repo_id] = updated
    return updated


@router.post(
    "/{repo_id}/pr-preview",
    response_model=PullRequestPreviewResponse,
    summary="Generate Self-Healing Pull Request Preview",
)
def generate_pr_preview(repo_id: str, payload: PullRequestPreviewRequest) -> PullRequestPreviewResponse:
    """Generate a non-destructive unified git diff and PR draft requiring human approval."""
    repo = _get_or_404(repo_id)
    file_path = payload.file_path or _default_spec_files(repo.test_directory, repo.framework)[0]

    healed_selector = (payload.healed_selector or "").strip()
    explanation = "Replaced brittle selector with resilient semantic ARIA locator."
    if not healed_selector:
        suggestion = HealingAgent().heal_locator(
            {
                "failed_selector": payload.failed_selector,
                "action_description": payload.action_description,
                "dom_excerpt": payload.dom_excerpt,
                "target_url": repo.repo_url,
            }
        )
        framework_patch = getattr(suggestion, repo.framework)
        healed_selector = framework_patch.healed_selector
        explanation = framework_patch.explanation

    if repo.framework == "cypress":
        old_line = f"-  cy.get('{payload.failed_selector}').click();"
        new_line = f"+  {healed_selector}.click();"
    elif repo.framework == "selenium":
        old_line = f'-    target = driver.find_element(By.CSS_SELECTOR, "{payload.failed_selector}")'
        new_line = f'+    target = driver.find_element({healed_selector})'
    else:
        old_line = f"-  const target = page.locator('{payload.failed_selector}');"
        new_line = f"+  const target = {healed_selector};"

    unified_diff = "\n".join(
        [
            f"diff --git a/{file_path} b/{file_path}",
            f"--- a/{file_path}",
            f"+++ b/{file_path}",
            "@@ -12,4 +12,4 @@",
            old_line,
            new_line,
        ]
    )

    proposed_branch = f"agentswarm/heal-{uuid4().hex[:6]}"
    return PullRequestPreviewResponse(
        repository_id=repo.id,
        repo_url=repo.repo_url,
        base_branch=repo.branch,
        proposed_branch=proposed_branch,
        pr_title=f"fix(e2e): heal broken locator '{payload.failed_selector}'",
        pr_body=(
            f"### Automated Selector Healing Preview\n\n"
            f"- **Framework:** `{repo.framework}`\n"
            f"- **Failed Selector:** `{payload.failed_selector}`\n"
            f"- **Replacement:** `{healed_selector}`\n"
            f"- **Rationale:** {explanation}\n\n"
            f"> Pending human approval before creating branch `{proposed_branch}`."
        ),
        file_path=file_path,
        framework=repo.framework,
        failed_selector=payload.failed_selector,
        healed_selector=healed_selector,
        unified_diff=unified_diff,
        requires_human_approval=True,
    )
