from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Load environment variables from backend/.env if present
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)


class FailureEvidence(BaseModel):
    """Telemetry captured from a failed browser step or runtime trace."""

    failure_id: Optional[str] = None
    scenario_id: Optional[str] = None
    scenario_title: Optional[str] = None
    step_number: Optional[int] = None
    target_url: str = ""
    error_message: str
    failed_selector: Optional[str] = None
    http_status: Optional[int] = None
    console_errors: list[str] = Field(default_factory=list)
    network_failures: list[str] = Field(default_factory=list)
    dom_excerpt: Optional[str] = None
    action_description: Optional[str] = None


class TriageDiagnosis(BaseModel):
    """Root-cause classification and routing decision for a captured failure."""

    failure_id: str
    category: Literal[
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
    generated_by: str = Field("gemini", description="Origin engine ('gemini' or 'deterministic_fallback')")


class TriageAgent:
    """
    Root-cause triage agent.

    Distinguishes frontend UI selector drift from genuine backend API failures,
    network timeouts, console exceptions, and functional assertion bugs so downstream
    self-healing only triggers on true locator regressions.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None) -> None:
        self.api_key = (api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
        self.model_name = (model_name or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()
        self._client: Optional[genai.Client] = genai.Client(api_key=self.api_key) if self.api_key else None

    @staticmethod
    def _coerce_evidence(evidence: FailureEvidence | dict[str, Any], index: int = 1) -> FailureEvidence:
        if isinstance(evidence, FailureEvidence):
            item = evidence.model_copy()
        else:
            item = FailureEvidence.model_validate(evidence)
        if not item.failure_id:
            item.failure_id = f"FAIL-{index:03d}"
        return item

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    def _invoke_gemini(self, item: FailureEvidence) -> TriageDiagnosis:
        if not self._client:
            raise RuntimeError("Gemini client is not configured.")

        prompt = (
            f"Failure ID: {item.failure_id}\n"
            f"Target URL: {item.target_url}\n"
            f"Scenario: {item.scenario_id} - {item.scenario_title}\n"
            f"Action Description: {item.action_description}\n"
            f"Failed Selector: {item.failed_selector}\n"
            f"HTTP Status: {item.http_status}\n"
            f"Error Message: {item.error_message}\n"
            f"Console Errors: {item.console_errors}\n"
            f"Network Failures: {item.network_failures}\n"
            f"DOM Excerpt:\n{(item.dom_excerpt or 'N/A')[:2000]}\n"
        )
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are the Diagnosis & Triage Agent for AgentSwarm. Classify each test failure into "
                    "SELECTOR_DRIFT, BACKEND_API_ERROR, NETWORK_TIMEOUT, ASSERTION_FAILURE, or CONSOLE_EXCEPTION. "
                    "Set needs_healing=True and recommended_next_agent='healing_agent' ONLY when the root cause "
                    "is SELECTOR_DRIFT with a broken selector and healthy backend behavior."
                ),
                response_mime_type="application/json",
                response_schema=TriageDiagnosis,
                temperature=0.1,
            ),
        )
        if response.parsed is not None and isinstance(response.parsed, TriageDiagnosis):
            diagnosis = response.parsed
            diagnosis.failure_id = item.failure_id or diagnosis.failure_id
            return diagnosis
        diagnosis = TriageDiagnosis.model_validate_json(response.text)
        diagnosis.failure_id = item.failure_id or diagnosis.failure_id
        return diagnosis

    def triage_failure(self, evidence: FailureEvidence | dict[str, Any]) -> TriageDiagnosis:
        """Analyze a single failure evidence record and return a structured diagnosis."""
        item = self._coerce_evidence(evidence)
        if self._client:
            try:
                return self._invoke_gemini(item)
            except Exception as exc:
                logger.warning("[TriageAgent] Gemini triage failed, using rule engine: %s", exc)
        return self._classify_deterministically(item)

    @staticmethod
    def _classify_deterministically(item: FailureEvidence) -> TriageDiagnosis:
        err_lower = (item.error_message or "").lower()
        http_status = item.http_status
        highlights: list[str] = []

        if http_status is not None and http_status >= 400 and http_status != 504:
            highlights.append(f"HTTP status {http_status} returned from target endpoint")
            if item.network_failures:
                highlights.extend(item.network_failures[:2])
            return TriageDiagnosis(
                failure_id=item.failure_id or "FAIL-001",
                category="BACKEND_API_ERROR",
                severity="critical" if http_status >= 500 else "high",
                confidence=0.96,
                root_cause_summary=f"Server returned HTTP {http_status} response: {item.error_message[:180]}",
                verdict="Backend API / Server Error (Self-Healing Suppressed)",
                needs_healing=False,
                recommended_next_agent="human_review",
                evidence_highlights=highlights,
            )

        if item.network_failures and any("500" in nf or "502" in nf or "503" in nf or "400" in nf or "401" in nf or "403" in nf for nf in item.network_failures):
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
            highlights.append(f"Network connectivity or navigation timeout: {item.error_message[:120]}")
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
        has_selector_drift = bool(item.failed_selector) and (
            any(marker in err_lower for marker in selector_markers) or "timeout" in err_lower or "not found" in err_lower
        )
        if not has_selector_drift and any(marker in err_lower for marker in selector_markers):
            has_selector_drift = True

        if has_selector_drift:
            highlights.append(f"Failed selector: {item.failed_selector or 'unspecified'}")
            if http_status is not None:
                highlights.append(f"Healthy HTTP {http_status} confirmed (pure markup refactor)")
            if item.dom_excerpt:
                highlights.append("DOM snapshot available for semantic ARIA locator synthesis")
            return TriageDiagnosis(
                failure_id=item.failure_id or "FAIL-001",
                category="SELECTOR_DRIFT",
                severity="medium",
                confidence=0.95 if item.failed_selector else 0.85,
                root_cause_summary=(
                    f"UI selector '{item.failed_selector or 'target'}' failed while application remained responsive. "
                    "Likely DOM markup or attribute refactor."
                ),
                verdict="Pure Markup / Selector Drift -> Triggering Healing Agent",
                needs_healing=True,
                recommended_next_agent="healing_agent",
                evidence_highlights=highlights,
            )

        if item.console_errors:
            highlights.extend(item.console_errors[:3])
            return TriageDiagnosis(
                failure_id=item.failure_id or "FAIL-001",
                category="CONSOLE_EXCEPTION",
                severity="high",
                confidence=0.88,
                root_cause_summary=f"Uncaught browser runtime console error detected: {item.console_errors[0][:160]}",
                verdict="Frontend JavaScript Runtime Exception",
                needs_healing=False,
                recommended_next_agent="human_review",
                evidence_highlights=highlights,
            )

        highlights.append(item.error_message[:160])
        return TriageDiagnosis(
            failure_id=item.failure_id or "FAIL-001",
            category="ASSERTION_FAILURE",
            severity="high",
            confidence=0.82,
            root_cause_summary=f"Functional expectation failed: {item.error_message[:180]}",
            verdict="Functional Assertion Mismatch",
            needs_healing=False,
            recommended_next_agent="human_review",
            evidence_highlights=highlights,
        )

    def triage_batch(
        self,
        failures: list[FailureEvidence | dict[str, Any]],
        target_url: str = "",
    ) -> TriageReport:
        """Analyze all failures from a test run and produce a consolidated TriageReport."""
        diagnoses: list[TriageDiagnosis] = []
        used_gemini = False

        for idx, raw in enumerate(failures, start=1):
            item = self._coerce_evidence(raw, index=idx)
            if self._client:
                try:
                    diag = self._invoke_gemini(item)
                    used_gemini = True
                except Exception as exc:
                    logger.warning("[TriageAgent] Batch item %s Gemini error, falling back: %s", item.failure_id, exc)
                    diag = self._classify_deterministically(item)
            else:
                diag = self._classify_deterministically(item)
            diagnoses.append(diag)

        healing_count = sum(1 for d in diagnoses if d.needs_healing)
        return TriageReport(
            target_url=target_url,
            total_failures=len(diagnoses),
            healing_candidates=healing_count,
            diagnoses=diagnoses,
            generated_by="gemini" if used_gemini else "deterministic_fallback",
        )

    def run_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph-compatible node entrypoint for root-cause failure triage."""
        target_url = state.get("target_url", "")
        raw_failures: list[Any] = list(state.get("raw_failures") or [])

        if not raw_failures and state.get("navigation_results"):
            for nav in state["navigation_results"]:
                if isinstance(nav, dict):
                    raw_failures.extend(nav.get("failures") or [])

        coerced = [self._coerce_evidence(item, index=i) for i, item in enumerate(raw_failures, start=1)]
        report = self.triage_batch(coerced, target_url=target_url)

        healing_candidates: list[dict[str, Any]] = []
        for evidence_item, diag in zip(coerced, report.diagnoses):
            if diag.needs_healing and evidence_item.failed_selector:
                healing_candidates.append(
                    {
                        "failure_id": diag.failure_id,
                        "failed_selector": evidence_item.failed_selector,
                        "action_description": evidence_item.action_description,
                        "dom_excerpt": evidence_item.dom_excerpt or state.get("dom_snapshot") or state.get("dom_summary"),
                        "target_url": evidence_item.target_url or target_url,
                        "error_message": evidence_item.error_message,
                    }
                )

        return {
            "triage_report": report.model_dump(),
            "healing_candidates": healing_candidates,
            "current_agent": "triage",
        }


def triage_node(state: dict[str, Any]) -> dict[str, Any]:
    """Functional LangGraph node wrapper around TriageAgent."""
    return TriageAgent().run_node(state)
