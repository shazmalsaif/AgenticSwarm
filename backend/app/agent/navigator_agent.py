from __future__ import annotations

import logging
import os
import re
import time
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


class BrowserAction(BaseModel):
    """Atomic Playwright-compatible browser action."""

    step_number: int = Field(..., ge=1)
    action_type: Literal["goto", "click", "fill", "wait_for_selector", "assert_visible", "assert_url"]
    selector: Optional[str] = Field(None, description="Primary CSS, text, or role selector")
    fallback_selectors: list[str] = Field(default_factory=list, description="Backup locators")
    value: Optional[str] = Field(None, description="URL or input value where applicable")
    description: str = Field("", description="Human-readable action intent")
    timeout_ms: int = Field(5000, ge=100, description="Action timeout in milliseconds")


class _BrowserActionPlan(BaseModel):
    actions: list[BrowserAction] = Field(default_factory=list)


class NavigationStepResult(BaseModel):
    """Execution outcome of a single BrowserAction step."""

    step_number: int
    action: BrowserAction
    status: Literal["passed", "failed", "skipped"]
    duration_ms: float = 0.0
    current_url: str = ""
    error_message: Optional[str] = None
    failed_selector: Optional[str] = None
    http_status: Optional[int] = None
    console_errors: list[str] = Field(default_factory=list)
    dom_excerpt: Optional[str] = None


class NavigationRunResult(BaseModel):
    """Aggregated browser run output for a single QA test scenario."""

    scenario_id: str
    scenario_title: str
    target_url: str
    overall_status: Literal["passed", "failed"]
    steps: list[NavigationStepResult] = Field(default_factory=list)
    failures: list[dict[str, Any]] = Field(default_factory=list)
    execution_mode: Literal["playwright", "simulated"] = "simulated"


class NavigatorAgent:
    """
    Browser navigation & step execution agent.

    Translates scenario steps into structured Playwright actions and executes them
    either against a live Playwright Page object or an intelligent DOM-aware simulator.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None) -> None:
        self.api_key = (api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
        self.model_name = (model_name or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()
        self._client: Optional[genai.Client] = genai.Client(api_key=self.api_key) if self.api_key else None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    def _invoke_gemini_actions(self, prompt: str) -> list[BrowserAction]:
        if not self._client:
            raise RuntimeError("Gemini client is not configured.")

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are the Browser Navigator Agent for AgentSwarm. Convert test scenario steps "
                    "into precise, ordered Playwright BrowserAction items using the provided DOM snapshot "
                    "when available. Prefer resilient accessible selectors when present."
                ),
                response_mime_type="application/json",
                response_schema=_BrowserActionPlan,
                temperature=0.1,
            ),
        )
        if response.parsed is not None and isinstance(response.parsed, _BrowserActionPlan):
            return response.parsed.actions
        parsed = _BrowserActionPlan.model_validate_json(response.text)
        return parsed.actions

    def plan_actions(
        self,
        target_url: str,
        scenario: dict[str, Any],
        dom_snapshot: Optional[str] = None,
    ) -> list[BrowserAction]:
        """Convert a scenario dict into executable BrowserAction items."""
        if self._client and scenario.get("use_llm", True):
            prompt = (
                f"Target URL: {target_url}\n"
                f"Scenario ID: {scenario.get('scenario_id', 'SCN-001')}\n"
                f"Scenario Title: {scenario.get('title', 'Test Scenario')}\n"
                f"Steps: {scenario.get('steps', [])}\n"
                f"DOM Snapshot:\n{(dom_snapshot or 'Not captured yet')[:3500]}\n"
            )
            try:
                actions = self._invoke_gemini_actions(prompt)
                if actions:
                    return actions
            except Exception as exc:
                logger.warning("[NavigatorAgent] Gemini action planning failed, using deterministic mapper: %s", exc)

        return self._map_steps_deterministically(target_url, scenario)

    @staticmethod
    def _map_steps_deterministically(target_url: str, scenario: dict[str, Any]) -> list[BrowserAction]:
        raw_steps = scenario.get("steps") or []
        if not raw_steps:
            return [
                BrowserAction(
                    step_number=1,
                    action_type="goto",
                    value=target_url,
                    description=f"Navigate to {target_url}",
                ),
                BrowserAction(
                    step_number=2,
                    action_type="assert_visible",
                    selector="main, body",
                    description="Confirm application shell rendered",
                ),
            ]

        actions: list[BrowserAction] = []
        for idx, step in enumerate(raw_steps, start=1):
            if not isinstance(step, dict):
                step = step.model_dump() if hasattr(step, "model_dump") else {"description": str(step)}
            raw_verb = str(step.get("action") or step.get("action_type") or "assert_visible").strip().lower()
            hint = step.get("target_hint") or step.get("selector")
            input_val = step.get("input_value") or step.get("value")
            desc = step.get("description") or f"Step {idx}"

            action_type: Literal["goto", "click", "fill", "wait_for_selector", "assert_visible", "assert_url"]
            if raw_verb in ("navigate", "goto", "open", "visit"):
                action_type = "goto"
                value = input_val or (hint if isinstance(hint, str) and hint.startswith("http") else target_url)
                selector = None
            elif raw_verb in ("click", "press", "submit", "tap"):
                action_type = "click"
                selector = str(hint or "button[type='submit'], button")
                value = input_val
            elif raw_verb in ("fill", "type", "input", "enter"):
                action_type = "fill"
                selector = str(hint or "input")
                value = str(input_val or "synthetic_input")
            elif raw_verb in ("wait", "wait_for_selector"):
                action_type = "wait_for_selector"
                selector = str(hint or "main")
                value = input_val
            elif raw_verb in ("assert_url", "verify_url"):
                action_type = "assert_url"
                selector = None
                value = str(input_val or hint or target_url)
            else:
                action_type = "assert_visible"
                selector = str(hint or "main, body")
                value = input_val

            actions.append(
                BrowserAction(
                    step_number=int(step.get("step_number") or idx),
                    action_type=action_type,
                    selector=selector,
                    value=value,
                    description=desc,
                )
            )
        return actions

    def execute_scenario(
        self,
        target_url: str,
        scenario: dict[str, Any],
        dom_snapshot: Optional[str] = None,
        page: Optional[Any] = None,
    ) -> NavigationRunResult:
        """Execute a single scenario via live Playwright Page or simulated DOM evaluation."""
        scenario_id = str(scenario.get("scenario_id") or "SCN-001")
        scenario_title = str(scenario.get("title") or "Untitled Scenario")
        actions = self.plan_actions(target_url=target_url, scenario=scenario, dom_snapshot=dom_snapshot)

        if page is not None:
            return self._execute_with_playwright(
                scenario_id=scenario_id,
                scenario_title=scenario_title,
                target_url=target_url,
                actions=actions,
                page=page,
            )

        return self._execute_simulated(
            scenario_id=scenario_id,
            scenario_title=scenario_title,
            target_url=target_url,
            actions=actions,
            dom_snapshot=dom_snapshot,
        )

    def _execute_with_playwright(
        self,
        scenario_id: str,
        scenario_title: str,
        target_url: str,
        actions: list[BrowserAction],
        page: Any,
    ) -> NavigationRunResult:
        step_results: list[NavigationStepResult] = []
        failures: list[dict[str, Any]] = []
        failed_already = False

        for action in actions:
            if failed_already:
                step_results.append(
                    NavigationStepResult(
                        step_number=action.step_number,
                        action=action,
                        status="skipped",
                        current_url=getattr(page, "url", target_url),
                    )
                )
                continue

            started = time.perf_counter()
            http_status: Optional[int] = None
            try:
                if action.action_type == "goto":
                    resp = page.goto(action.value or target_url, timeout=action.timeout_ms)
                    http_status = getattr(resp, "status", None) if resp else None
                elif action.action_type == "click":
                    page.locator(action.selector or "button").first.click(timeout=action.timeout_ms)
                elif action.action_type == "fill":
                    page.locator(action.selector or "input").first.fill(action.value or "", timeout=action.timeout_ms)
                elif action.action_type == "wait_for_selector":
                    page.wait_for_selector(action.selector or "body", timeout=action.timeout_ms)
                elif action.action_type == "assert_visible":
                    page.locator(action.selector or "body").first.wait_for(state="visible", timeout=action.timeout_ms)
                elif action.action_type == "assert_url":
                    expected = action.value or target_url
                    current = str(getattr(page, "url", ""))
                    if expected not in current:
                        raise AssertionError(f"Expected URL to contain '{expected}', got '{current}'")

                elapsed = round((time.perf_counter() - started) * 1000, 2)
                step_results.append(
                    NavigationStepResult(
                        step_number=action.step_number,
                        action=action,
                        status="passed",
                        duration_ms=elapsed,
                        current_url=str(getattr(page, "url", target_url)),
                        http_status=http_status,
                    )
                )
            except Exception as exc:
                failed_already = True
                elapsed = round((time.perf_counter() - started) * 1000, 2)
                dom_excerpt = None
                try:
                    dom_excerpt = page.content()[:1500]
                except Exception:
                    pass
                current_url = str(getattr(page, "url", target_url))
                err_msg = str(exc)
                step_results.append(
                    NavigationStepResult(
                        step_number=action.step_number,
                        action=action,
                        status="failed",
                        duration_ms=elapsed,
                        current_url=current_url,
                        error_message=err_msg,
                        failed_selector=action.selector,
                        http_status=http_status,
                        dom_excerpt=dom_excerpt,
                    )
                )
                failures.append(
                    {
                        "failure_id": f"{scenario_id}-STEP-{action.step_number}",
                        "scenario_id": scenario_id,
                        "scenario_title": scenario_title,
                        "step_number": action.step_number,
                        "target_url": current_url,
                        "error_message": err_msg,
                        "failed_selector": action.selector,
                        "http_status": http_status,
                        "console_errors": [],
                        "dom_excerpt": dom_excerpt,
                        "action_description": action.description,
                    }
                )

        return NavigationRunResult(
            scenario_id=scenario_id,
            scenario_title=scenario_title,
            target_url=target_url,
            overall_status="failed" if failures else "passed",
            steps=step_results,
            failures=failures,
            execution_mode="playwright",
        )

    @staticmethod
    def _selector_missing_in_dom(selector: Optional[str], dom_snapshot: Optional[str]) -> bool:
        """Detect obvious CSS #id or .class drift when an HTML DOM snapshot is supplied."""
        if not selector or not dom_snapshot or "<" not in dom_snapshot:
            return False
        # Skip multi-candidate fallback strings unless none match
        candidates = [part.strip() for part in selector.split(",") if part.strip()]
        if not candidates:
            return False

        for candidate in candidates:
            if candidate.startswith("getBy") or " " not in candidate and candidate in ("main", "body", "nav", "button", "input"):
                tag_name = candidate.split("[")[0]
                if f"<{tag_name}" in dom_snapshot.lower():
                    return False
            id_match = re.search(r"#([A-Za-z0-9_-]+)", candidate)
            if id_match:
                el_id = id_match.group(1)
                if re.search(rf'id=["\'][^"\']*\b{re.escape(el_id)}\b', dom_snapshot):
                    return False
                continue
            cls_match = re.search(r"\.([A-Za-z0-9_-]+)", candidate)
            if cls_match:
                cls_name = cls_match.group(1)
                if re.search(rf'class=["\'][^"\']*\b{re.escape(cls_name)}\b', dom_snapshot):
                    return False
                continue
            return False
        return True

    def _execute_simulated(
        self,
        scenario_id: str,
        scenario_title: str,
        target_url: str,
        actions: list[BrowserAction],
        dom_snapshot: Optional[str],
    ) -> NavigationRunResult:
        step_results: list[NavigationStepResult] = []
        failures: list[dict[str, Any]] = []
        failed_already = False
        current_url = target_url

        for action in actions:
            if failed_already:
                step_results.append(
                    NavigationStepResult(
                        step_number=action.step_number,
                        action=action,
                        status="skipped",
                        current_url=current_url,
                    )
                )
                continue

            if action.action_type == "goto" and action.value:
                current_url = action.value

            selector_broken = action.action_type in ("click", "fill", "wait_for_selector", "assert_visible") and (
                self._selector_missing_in_dom(action.selector, dom_snapshot)
            )

            if selector_broken:
                failed_already = True
                err_msg = f"Timeout {action.timeout_ms}ms waiting for selector '{action.selector}'"
                dom_excerpt = dom_snapshot[:1500] if dom_snapshot else None
                step_results.append(
                    NavigationStepResult(
                        step_number=action.step_number,
                        action=action,
                        status="failed",
                        duration_ms=float(action.timeout_ms),
                        current_url=current_url,
                        error_message=err_msg,
                        failed_selector=action.selector,
                        http_status=200,
                        dom_excerpt=dom_excerpt,
                    )
                )
                failures.append(
                    {
                        "failure_id": f"{scenario_id}-STEP-{action.step_number}",
                        "scenario_id": scenario_id,
                        "scenario_title": scenario_title,
                        "step_number": action.step_number,
                        "target_url": current_url,
                        "error_message": err_msg,
                        "failed_selector": action.selector,
                        "http_status": 200,
                        "console_errors": [],
                        "dom_excerpt": dom_excerpt,
                        "action_description": action.description,
                    }
                )
            else:
                step_results.append(
                    NavigationStepResult(
                        step_number=action.step_number,
                        action=action,
                        status="passed",
                        duration_ms=42.0,
                        current_url=current_url,
                        http_status=200 if action.action_type == "goto" else None,
                    )
                )

        return NavigationRunResult(
            scenario_id=scenario_id,
            scenario_title=scenario_title,
            target_url=target_url,
            overall_status="failed" if failures else "passed",
            steps=step_results,
            failures=failures,
            execution_mode="simulated",
        )

    def run_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph-compatible node entrypoint for executing planned scenarios."""
        target_url = state.get("target_url", "https://example.com")
        dom_snapshot = state.get("dom_snapshot") or state.get("dom_summary")
        page = state.get("page")

        test_plan = state.get("test_plan") or {}
        if hasattr(test_plan, "model_dump"):
            test_plan = test_plan.model_dump()
        scenarios = test_plan.get("scenarios") or [
            {"scenario_id": "SCN-001", "title": "Default Smoke Check", "steps": []}
        ]

        results: list[dict[str, Any]] = []
        raw_failures: list[dict[str, Any]] = list(state.get("raw_failures") or [])

        for scenario in scenarios:
            run_result = self.execute_scenario(
                target_url=target_url,
                scenario=scenario,
                dom_snapshot=dom_snapshot,
                page=page,
            )
            results.append(run_result.model_dump())
            raw_failures.extend(run_result.failures)

        return {
            "navigation_results": results,
            "raw_failures": raw_failures,
            "current_agent": "navigator",
        }


def navigator_node(state: dict[str, Any]) -> dict[str, Any]:
    """Functional LangGraph node wrapper around NavigatorAgent."""
    return NavigatorAgent().run_node(state)
