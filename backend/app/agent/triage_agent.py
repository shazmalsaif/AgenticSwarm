from __future__ import annotations

import logging
from typing import Any, Literal, Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .planner_agent import invoke_with_model_fallback, resolve_chat_model

logger = logging.getLogger(__name__)


class FailureEvidence(BaseModel):
    """Telemetry captured from a failed browser step or runtime trace."""

    failure_id: Optional[str] = None
    scenario_id: Optional[str] = None
    scenario_title: Optional[str] = None
    step_number: Optional[int] = None
    target_url: str = ""
    current_url: Optional[str] = None
    error_message: str
    failed_selector: Optional[str] = None
    http_status: Optional[int] = None
    console_errors: list[str] = Field(default_factory=list)
    network_failures: list[str] = Field(default_factory=list)
    dom_excerpt: Optional[str] = None
    screenshot_b64: Optional[str] = None
    blocking_overlay: Optional[str] = None
    action_description: Optional[str] = None


class TriageDiagnosis(BaseModel):
    """Root-cause classification and routing decision for a captured failure."""

    failure_id: str
    category: Literal[
        "NAVIGATION_MISSING",
        "OVERLAY_BLOCKING",
        "SELECTOR_DRIFT",
        "BACKEND_API_ERROR",
        "NETWORK_TIMEOUT",
        "ASSERTION_FAILURE",
        "CONSOLE_EXCEPTION",
    ]
    severity: Literal["critical", "high", "medium", "low"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    root_cause_summary: str
    verdict: str
    needs_healing: bool
    recommended_next_agent: Literal["healing_agent", "human_review", "retry_verification"]
    evidence_highlights: list[str] = Field(default_factory=list)


class TriageReport(BaseModel):
    """Batch triage summary across all failures in a test execution."""

    target_url: str = ""
    total_failures: int
    healing_candidates: int
    diagnoses: list[TriageDiagnosis] = Field(default_factory=list)
    generated_by: str = Field("langchain", description="Origin engine identifier")


class TriageGraphState(TypedDict, total=False):
    target_url: str
    raw_failures: list[Any]
    dom_snapshot: Optional[str]
    coerced_failures: list[FailureEvidence]
    triage_report: TriageReport
    healing_candidates: list[dict[str, Any]]
    needs_healing: bool


class TriageAgent:
    """
    Model-agnostic root-cause triage agent powered by LangChain & LangGraph.

    Checks page URL (`about:blank` -> NAVIGATION_MISSING) and blocking overlays
    (`OVERLAY_BLOCKING`) BEFORE ever classifying anything as BACKEND_API_ERROR or NETWORK_TIMEOUT.
    """

    SYSTEM_PROMPT = (
        "You are the Diagnosis & Triage Agent for AgentSwarm. Classify each test failure into one of: "
        "NAVIGATION_MISSING, OVERLAY_BLOCKING, SELECTOR_DRIFT, BACKEND_API_ERROR, NETWORK_TIMEOUT, "
        "ASSERTION_FAILURE, or CONSOLE_EXCEPTION.\n"
        "CRITICAL RULES:\n"
        "- If Current URL is 'about:blank' or empty -> ALWAYS classify as NAVIGATION_MISSING (NEVER backend/network).\n"
        "- If a visible Blocking Overlay (#intro-overlay, modal, dialog backdrop) intercepted clicks -> ALWAYS classify as OVERLAY_BLOCKING (NEVER backend/network).\n"
        "- Set needs_healing=True and recommended_next_agent='healing_agent' ONLY for SELECTOR_DRIFT with healthy backend."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        llm: Optional[BaseChatModel] = None,
    ) -> None:
        self._api_key = api_key
        self._model_name = model_name
        self._provider = provider
        self._custom_llm_injected = llm is not None
        self._llm: Optional[BaseChatModel] = llm or resolve_chat_model(
            model_name=model_name,
            provider=provider,
            api_key=api_key,
            temperature=0.1,
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM_PROMPT),
                (
                    "human",
                    "Failure ID: {failure_id}\n"
                    "Current URL: {current_url}\n"
                    "Target URL: {target_url}\n"
                    "Blocking Overlay: {blocking_overlay}\n"
                    "Goal: {scenario_id} - {scenario_title}\n"
                    "Action Description: {action_description}\n"
                    "Failed Selector: {failed_selector}\n"
                    "HTTP Status: {http_status}\n"
                    "Error Message: {error_message}\n"
                    "Console Errors: {console_errors}\n"
                    "Network Failures: {network_failures}\n"
                    "DOM Excerpt:\n{dom_excerpt}\n",
                ),
            ]
        )
        self._chain = (self._prompt | self._llm.with_structured_output(TriageDiagnosis)) if self._llm else None
        self._graph = self.build_triage_graph()

    @staticmethod
    def _coerce_evidence(evidence: FailureEvidence | dict[str, Any], index: int = 1) -> FailureEvidence:
        item = evidence.model_copy() if isinstance(evidence, FailureEvidence) else FailureEvidence.model_validate(evidence)
        if not item.failure_id:
            item.failure_id = f"FAIL-{index:03d}"
        if item.current_url is None and item.target_url:
            item.current_url = item.target_url
        return item

    @staticmethod
    def _check_environment_root_cause(item: FailureEvidence) -> Optional[TriageDiagnosis]:
        """Run mandatory pre-checks for missing navigation (about:blank) and blocking overlays before network/backend triage."""
        url_check = (item.current_url or item.target_url or "").strip().lower()
        err_low = (item.error_message or "").lower()
        dom_low = (item.dom_excerpt or "").lower()

        if not url_check or url_check.startswith("about:blank") or "about:blank" in err_low or "not navigated to target app" in err_low:
            return TriageDiagnosis(
                failure_id=item.failure_id or "FAIL-001",
                category="NAVIGATION_MISSING",
                severity="critical",
                confidence=0.99,
                root_cause_summary=(
                    f"Browser page remained at '{url_check or 'about:blank'}' without navigating to the target URL. "
                    "Not a backend API or network timeout failure."
                ),
                verdict="Missing Navigation (about:blank)",
                needs_healing=False,
                recommended_next_agent="human_review",
                evidence_highlights=[
                    f"current_url={url_check or 'empty'}",
                    item.error_message[:140],
                ],
            )

        has_overlay = bool(item.blocking_overlay) or any(
            tok in err_low
            for tok in ("intercepts pointer events", "blocking overlay", "intro-overlay", "other element would receive pointer")
        ) or (
            any(tok in err_low for tok in ("timeout", "not visible", "waiting for locator"))
            and 'id="intro-overlay"' in dom_low
        )
        if has_overlay:
            ov_name = item.blocking_overlay or ("#intro-overlay" if "intro-overlay" in (err_low + dom_low) else "visible modal/overlay")
            return TriageDiagnosis(
                failure_id=item.failure_id or "FAIL-001",
                category="OVERLAY_BLOCKING",
                severity="high",
                confidence=0.98,
                root_cause_summary=(
                    f"Visible blocking overlay '{ov_name}' intercepted interaction during step execution. "
                    "Dismiss or wait for overlay clearance first; not a backend or network error."
                ),
                verdict=f"Blocked by Overlay ({ov_name})",
                needs_healing=False,
                recommended_next_agent="retry_verification",
                evidence_highlights=[
                    f"blocking_overlay={ov_name}",
                    f"current_url={item.current_url or item.target_url}",
                    item.error_message[:140],
                ],
            )
        return None

    def _invoke_chain(self, item: FailureEvidence) -> TriageDiagnosis:
        env_diag = self._check_environment_root_cause(item)
        if env_diag is not None:
            return env_diag

        prompt_input = {
            "failure_id": item.failure_id or "FAIL-001",
            "current_url": item.current_url or item.target_url or "N/A",
            "target_url": item.target_url,
            "blocking_overlay": item.blocking_overlay or "None",
            "scenario_id": item.scenario_id or "N/A",
            "scenario_title": item.scenario_title or "N/A",
            "action_description": item.action_description or "N/A",
            "failed_selector": item.failed_selector or "N/A",
            "http_status": str(item.http_status) if item.http_status is not None else "N/A",
            "error_message": item.error_message,
            "console_errors": str(item.console_errors),
            "network_failures": str(item.network_failures),
            "dom_excerpt": (item.dom_excerpt or "N/A")[:2000],
        }
        result, _ = invoke_with_model_fallback(
            self._prompt,
            TriageDiagnosis,
            prompt_input,
            primary_chain=self._chain,
            provider=self._provider,
            model_name=self._model_name,
            api_key=self._api_key,
            temperature=0.1,
            allow_fallback=not self._custom_llm_injected,
        )
        if not isinstance(result, TriageDiagnosis):
            result = TriageDiagnosis.model_validate(result)
        result.failure_id = item.failure_id or result.failure_id

        # Hard guardrail: LLM may never blame backend/network when url was blank or overlay blocked
        guardrail = self._check_environment_root_cause(item)
        if guardrail is not None:
            return guardrail
        return result

    def triage_failure(self, evidence: FailureEvidence | dict[str, Any]) -> TriageDiagnosis:
        item = self._coerce_evidence(evidence)
        env_diag = self._check_environment_root_cause(item)
        if env_diag is not None:
            return env_diag
        if self._chain:
            try:
                return self._invoke_chain(item)
            except Exception as exc:
                logger.warning("[TriageAgent] LangChain triage failed, using rule engine: %s", exc)
        return self._classify_deterministically(item)

    def _node_collect_failures(self, state: TriageGraphState) -> dict[str, Any]:
        raw_failures = list(state.get("raw_failures") or [])
        fallback_dom = state.get("dom_snapshot")
        coerced: list[FailureEvidence] = []
        for i, item in enumerate(raw_failures, start=1):
            ev = self._coerce_evidence(item, index=i)
            if not ev.dom_excerpt and fallback_dom:
                ev.dom_excerpt = str(fallback_dom)[:2000]
            coerced.append(ev)
        return {"coerced_failures": coerced}

    def _node_classify_failures(self, state: TriageGraphState) -> dict[str, Any]:
        target_url = state.get("target_url", "")
        coerced = state.get("coerced_failures") or []
        diagnoses: list[TriageDiagnosis] = []
        used_llm = False

        for item in coerced:
            env_diag = self._check_environment_root_cause(item)
            if env_diag is not None:
                diagnoses.append(env_diag)
                continue
            if self._chain:
                try:
                    diagnoses.append(self._invoke_chain(item))
                    used_llm = True
                    continue
                except Exception as exc:
                    logger.warning("[TriageAgent] Item %s LangChain error, falling back: %s", item.failure_id, exc)
            diagnoses.append(self._classify_deterministically(item))

        healing_count = sum(1 for d in diagnoses if d.needs_healing)
        return {
            "triage_report": TriageReport(
                target_url=target_url,
                total_failures=len(diagnoses),
                healing_candidates=healing_count,
                diagnoses=diagnoses,
                generated_by="langchain" if used_llm else "deterministic_fallback",
            )
        }

    @staticmethod
    def _node_route_healing_candidates(state: TriageGraphState) -> dict[str, Any]:
        target_url = state.get("target_url", "")
        coerced = state.get("coerced_failures") or []
        report = state["triage_report"]
        healing_candidates: list[dict[str, Any]] = []

        for evidence_item, diag in zip(coerced, report.diagnoses):
            if diag.needs_healing and evidence_item.failed_selector:
                healing_candidates.append(
                    {
                        "failure_id": diag.failure_id,
                        "failed_selector": evidence_item.failed_selector,
                        "action_description": evidence_item.action_description,
                        "dom_excerpt": evidence_item.dom_excerpt or state.get("dom_snapshot"),
                        "target_url": evidence_item.current_url or evidence_item.target_url or target_url,
                        "blocking_overlay": evidence_item.blocking_overlay,
                        "error_message": evidence_item.error_message,
                    }
                )

        return {"healing_candidates": healing_candidates, "needs_healing": bool(healing_candidates)}

    def build_triage_graph(self):
        builder = StateGraph(TriageGraphState)
        builder.add_node("collect_failures", self._node_collect_failures)
        builder.add_node("classify_failures", self._node_classify_failures)
        builder.add_node("route_healing_candidates", self._node_route_healing_candidates)
        builder.add_edge(START, "collect_failures")
        builder.add_edge("collect_failures", "classify_failures")
        builder.add_edge("classify_failures", "route_healing_candidates")
        builder.add_edge("route_healing_candidates", END)
        return builder.compile()

    def triage_batch(self, failures: list[FailureEvidence | dict[str, Any]], target_url: str = "") -> TriageReport:
        return self._graph.invoke({"raw_failures": failures, "target_url": target_url})["triage_report"]

    @classmethod
    def _classify_deterministically(cls, item: FailureEvidence) -> TriageDiagnosis:
        env_diag = cls._check_environment_root_cause(item)
        if env_diag is not None:
            return env_diag

        err_lower = (item.error_message or "").lower()
        http_status = item.http_status
        highlights: list[str] = []

        if http_status is not None and http_status >= 400 and http_status != 504:
            highlights.append(f"HTTP {http_status} from target endpoint")
            return TriageDiagnosis(
                failure_id=item.failure_id or "FAIL-001",
                category="BACKEND_API_ERROR",
                severity="critical" if http_status >= 500 else "high",
                confidence=0.96,
                root_cause_summary=f"Server returned HTTP {http_status}: {item.error_message[:180]}",
                verdict="Backend API / Server Error (Self-Healing Suppressed)",
                needs_healing=False,
                recommended_next_agent="human_review",
                evidence_highlights=highlights,
            )

        if item.network_failures and any(c in nf for nf in item.network_failures for c in ("500", "502", "503", "400", "401", "403")):
            highlights.extend(item.network_failures[:3])
            return TriageDiagnosis(
                failure_id=item.failure_id or "FAIL-001",
                category="BACKEND_API_ERROR",
                severity="critical",
                confidence=0.92,
                root_cause_summary="Network failure log shows failing HTTP 4xx/5xx API call.",
                verdict="Backend API Failure",
                needs_healing=False,
                recommended_next_agent="human_review",
                evidence_highlights=highlights,
            )

        network_markers = ("net::err_", "econnrefused", "enotfound", "navigation timeout", "gateway timeout", "err_connection")
        if http_status == 504 or any(marker in err_lower for marker in network_markers):
            highlights.append(item.error_message[:120])
            return TriageDiagnosis(
                failure_id=item.failure_id or "FAIL-001",
                category="NETWORK_TIMEOUT",
                severity="high",
                confidence=0.9,
                root_cause_summary="Target host timed out or refused connection during navigation.",
                verdict="Transient Network or Host Reachability Timeout",
                needs_healing=False,
                recommended_next_agent="retry_verification",
                evidence_highlights=highlights,
            )

        selector_markers = (
            "waiting for selector",
            "waiting for locator",
            "unable to locate element",
            "nosuchelementexception",
            "element not found",
            "strict mode violation",
            "resolved to 0 elements",
        )
        if any(m in err_lower for m in selector_markers) or (
            item.failed_selector and any(m in err_lower for m in ("timeout", "not found", "disabled"))
        ):
            highlights.append(f"Failed selector: {item.failed_selector or 'unspecified'}")
            return TriageDiagnosis(
                failure_id=item.failure_id or "FAIL-001",
                category="SELECTOR_DRIFT",
                severity="medium",
                confidence=0.94,
                root_cause_summary=f"UI selector '{item.failed_selector or 'target'}' could not resolve while page remained reachable.",
                verdict="Selector Drift -> Triggering Healing Agent",
                needs_healing=True,
                recommended_next_agent="healing_agent",
                evidence_highlights=highlights,
            )

        if item.console_errors:
            return TriageDiagnosis(
                failure_id=item.failure_id or "FAIL-001",
                category="CONSOLE_EXCEPTION",
                severity="high",
                confidence=0.88,
                root_cause_summary=f"Uncaught browser console error: {item.console_errors[0][:160]}",
                verdict="Frontend JavaScript Runtime Exception",
                needs_healing=False,
                recommended_next_agent="human_review",
                evidence_highlights=item.console_errors[:3],
            )

        return TriageDiagnosis(
            failure_id=item.failure_id or "FAIL-001",
            category="ASSERTION_FAILURE",
            severity="high",
            confidence=0.82,
            root_cause_summary=f"Functional expectation failed: {item.error_message[:180]}",
            verdict="Functional Assertion Mismatch",
            needs_healing=False,
            recommended_next_agent="human_review",
            evidence_highlights=[item.error_message[:160]],
        )

    def run_node(self, state: dict[str, Any]) -> dict[str, Any]:
        target_url = state.get("target_url", "")
        raw_failures: list[Any] = list(state.get("raw_failures") or [])
        if not raw_failures and state.get("navigation_results"):
            for nav in state["navigation_results"]:
                if isinstance(nav, dict):
                    raw_failures.extend(nav.get("failures") or [])
        result_state = self._graph.invoke(
            {
                "raw_failures": raw_failures,
                "target_url": target_url,
                "dom_snapshot": state.get("dom_snapshot") or state.get("dom_summary"),
            }
        )
        return {
            "triage_report": result_state["triage_report"].model_dump(),
            "healing_candidates": result_state["healing_candidates"],
            "needs_healing": result_state["needs_healing"],
            "current_agent": "triage",
        }


def triage_node(state: dict[str, Any]) -> dict[str, Any]:
    return TriageAgent().run_node(state)
