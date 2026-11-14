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


class TestStepPlan(BaseModel):
    """Single executable step inside an autonomous QA scenario."""

    step_number: int = Field(..., ge=1, description="1-based order of the step")
    action: str = Field(..., description="Browser action verb (navigate, click, fill, wait, assert_visible, assert_url)")
    description: str = Field(..., description="Human-readable intent of the step")
    target_hint: Optional[str] = Field(None, description="Suggested selector or accessible role/label hint")
    input_value: Optional[str] = Field(None, description="Input payload for fill/select actions")
    expected_outcome: str = Field("", description="Expected observable result after this step")


class TestScenario(BaseModel):
    """Structured end-to-end QA test scenario created by the PlannerAgent."""

    scenario_id: str = Field(..., description="Unique scenario identifier (e.g., SCN-001)")
    title: str = Field(..., description="Concise title of the test scenario")
    category: str = Field("e2e", description="Category such as smoke, auth, checkout, navigation, edge_case")
    priority: Literal["critical", "high", "medium", "low"] = Field("high", description="Execution priority")
    description: str = Field(..., description="Detailed scope of what this scenario validates")
    preconditions: list[str] = Field(default_factory=list, description="Required setup before running steps")
    steps: list[TestStepPlan] = Field(default_factory=list, description="Ordered steps in the scenario")
    expected_result: str = Field(..., description="Final acceptance condition for the scenario")


class TestPlanResult(BaseModel):
    """Complete multi-scenario test plan for a target web application."""

    target_url: str = Field(..., description="Normalized URL of the application under test")
    summary: str = Field(..., description="High-level summary of the generated test strategy")
    scenarios: list[TestScenario] = Field(default_factory=list, description="Prioritized test scenarios")
    generated_by: str = Field("gemini", description="Origin engine ('gemini' or 'deterministic_fallback')")


class PlannerAgent:
    """
    Autonomous QA test planning agent.

    Uses Google Gemini structured output to map a target application URL,
    optional DOM summary, and test objectives into prioritized E2E scenarios.
    Falls back cleanly to deterministic planning when no API key is configured.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None) -> None:
        self.api_key = (api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
        self.model_name = (model_name or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()
        self._client: Optional[genai.Client] = genai.Client(api_key=self.api_key) if self.api_key else None

    @staticmethod
    def _normalize_url(target_url: str) -> str:
        url = (target_url or "").strip()
        if not url:
            return "https://example.com"
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        return url

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    def _invoke_gemini(self, prompt: str) -> TestPlanResult:
        if not self._client:
            raise RuntimeError("Gemini client is not configured.")

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are the Lead QA Planner Agent for AgentSwarm. Design concise, realistic, "
                    "non-destructive end-to-end browser test plans using semantic accessibility hints. "
                    "Only target staging/authorized workflows and never invent destructive actions."
                ),
                response_mime_type="application/json",
                response_schema=TestPlanResult,
                temperature=0.2,
            ),
        )
        if response.parsed is not None and isinstance(response.parsed, TestPlanResult):
            result = response.parsed
            result.generated_by = "gemini"
            return result
        result = TestPlanResult.model_validate_json(response.text)
        result.generated_by = "gemini"
        return result

    def create_test_plan(
        self,
        target_url: str,
        dom_summary: Optional[str] = None,
        objectives: Optional[list[str]] = None,
    ) -> TestPlanResult:
        """Generate a structured multi-scenario test plan for the target application."""
        normalized_url = self._normalize_url(target_url)

        if self._client:
            prompt = self._build_prompt(normalized_url, dom_summary, objectives)
            try:
                plan = self._invoke_gemini(prompt)
                plan.target_url = normalized_url
                return plan
            except Exception as exc:
                logger.warning("[PlannerAgent] Gemini planning failed, using deterministic fallback: %s", exc)

        return self._build_fallback_plan(normalized_url, dom_summary, objectives)

    @staticmethod
    def _build_prompt(
        target_url: str,
        dom_summary: Optional[str],
        objectives: Optional[list[str]],
    ) -> str:
        sections = [f"Target URL: {target_url}"]
        if objectives:
            sections.append("Testing Objectives:\n" + "\n".join(f"- {obj}" for obj in objectives))
        if dom_summary:
            sections.append(f"DOM / Route Summary:\n{dom_summary[:4000]}")
        sections.append(
            "Generate 3 to 5 prioritized end-to-end test scenarios covering smoke navigation, "
            "primary user workflow, authentication/session state, and negative edge cases."
        )
        return "\n\n".join(sections)

    @staticmethod
    def _build_fallback_plan(
        target_url: str,
        dom_summary: Optional[str],
        objectives: Optional[list[str]],
    ) -> TestPlanResult:
        dom_lower = (dom_summary or "").lower()
        scenarios: list[TestScenario] = [
            TestScenario(
                scenario_id="SCN-001",
                title="Landing Route Render & Primary Navigation Smoke",
                category="smoke",
                priority="critical",
                description="Verify initial document load, primary interactive landmarks, and zero blocking navigation errors.",
                preconditions=["Application URL is reachable", "Clean browser context"],
                steps=[
                    TestStepPlan(
                        step_number=1,
                        action="navigate",
                        description=f"Open target application at {target_url}",
                        target_hint=target_url,
                        expected_outcome="HTTP 2xx response and main document renders",
                    ),
                    TestStepPlan(
                        step_number=2,
                        action="assert_visible",
                        description="Verify primary navigation and main call-to-action controls are visible",
                        target_hint="button, main, nav",
                        expected_outcome="Primary landmarks and interactive buttons are visible",
                    ),
                ],
                expected_result="Application loads cleanly with accessible interactive controls.",
            ),
            TestScenario(
                scenario_id="SCN-002",
                title="Authentication Modal & Credential Validation Flow",
                category="auth",
                priority="high",
                description="Exercise sign-in trigger, form validation feedback, and authenticated session transition.",
                preconditions=["User starts unauthenticated"],
                steps=[
                    TestStepPlan(
                        step_number=1,
                        action="click",
                        description="Open the Sign In modal or login entry control",
                        target_hint="button[id*='signin'], getByRole('button', { name: /sign in/i })",
                        expected_outcome="Authentication form becomes visible",
                    ),
                    TestStepPlan(
                        step_number=2,
                        action="fill",
                        description="Populate email field with synthetic QA test email",
                        target_hint="input[type='email']",
                        input_value="qa.synthetic@agentswarm.dev",
                        expected_outcome="Email input reflects entered value",
                    ),
                    TestStepPlan(
                        step_number=3,
                        action="fill",
                        description="Populate password field with synthetic QA test password",
                        target_hint="input[type='password']",
                        input_value="SyntheticPass!2026",
                        expected_outcome="Password input masks and stores entry",
                    ),
                    TestStepPlan(
                        step_number=4,
                        action="click",
                        description="Submit authentication form",
                        target_hint="button[type='submit']",
                        expected_outcome="API responds without unhandled client error",
                    ),
                ],
                expected_result="Authentication request completes cleanly or shows descriptive validation feedback.",
            ),
        ]

        # Add workflow-specific scenario based on DOM hints or checkout context
        is_commerce = any(token in dom_lower for token in ("checkout", "cart", "product", "price", "order"))
        scenarios.append(
            TestScenario(
                scenario_id="SCN-003",
                title="Multi-Stage Checkout Journey" if is_commerce else "Core Interactive Workflow Verification",
                category="checkout" if is_commerce else "e2e",
                priority="high",
                description=(
                    "Execute end-to-end cart addition and checkout CTA progression while monitoring selector resilience."
                    if is_commerce
                    else "Validate primary user workflow controls, state transitions, and submission feedback."
                ),
                preconditions=["Main view loaded"],
                steps=[
                    TestStepPlan(
                        step_number=1,
                        action="click",
                        description="Trigger primary workflow action button",
                        target_hint="button[type='submit'], .cta-primary, getByRole('button', { name: /proceed|deploy|start/i })",
                        expected_outcome="Workflow transitions to active processing or next step",
                    ),
                    TestStepPlan(
                        step_number=2,
                        action="assert_visible",
                        description="Confirm workflow status output updates visibly",
                        target_hint="main, [role='status']",
                        expected_outcome="Updated application state renders without selector timeout",
                    ),
                ],
                expected_result="Core interactive workflow advances and updates DOM state reliably.",
            )
        )

        # Incorporate explicit caller objectives when provided
        if objectives:
            for idx, objective in enumerate(objectives[:3], start=4):
                scenarios.append(
                    TestScenario(
                        scenario_id=f"SCN-{idx:03d}",
                        title=f"Objective Check: {objective[:72]}",
                        category="objective",
                        priority="high",
                        description=f"Verify user-requested QA objective: {objective}",
                        preconditions=["Application loaded"],
                        steps=[
                            TestStepPlan(
                                step_number=1,
                                action="navigate",
                                description=f"Ensure application context at {target_url}",
                                target_hint=target_url,
                                expected_outcome="Application is active",
                            ),
                            TestStepPlan(
                                step_number=2,
                                action="assert_visible",
                                description=f"Validate elements related to objective: {objective}",
                                target_hint="main",
                                expected_outcome="Objective criteria satisfied without errors",
                            ),
                        ],
                        expected_result=f"Objective verified: {objective}",
                    )
                )

        return TestPlanResult(
            target_url=target_url,
            summary=f"Generated {len(scenarios)} deterministic QA test scenarios for {target_url}.",
            scenarios=scenarios,
            generated_by="deterministic_fallback",
        )

    def run_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph-compatible node entrypoint for test planning."""
        target_url = state.get("target_url", "https://example.com")
        dom_summary = state.get("dom_summary") or state.get("dom_snapshot")
        objectives = state.get("objectives")
        plan = self.create_test_plan(target_url=target_url, dom_summary=dom_summary, objectives=objectives)
        return {
            "target_url": plan.target_url,
            "test_plan": plan.model_dump(),
            "current_agent": "planner",
        }


def planner_node(state: dict[str, Any]) -> dict[str, Any]:
    """Functional LangGraph node wrapper around PlannerAgent."""
    return PlannerAgent().run_node(state)
