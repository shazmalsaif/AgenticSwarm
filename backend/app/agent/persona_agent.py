from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

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


class UserPersona(BaseModel):
    """Synthetic QA user persona representing distinct testing behaviors and edge cases."""

    persona_id: str = Field(..., description="Unique persona identifier (e.g., PRS-01)")
    name: str = Field(..., description="Descriptive persona name")
    role: str = Field(..., description="Functional role (e.g., Standard Shopper, Chaos Explorer)")
    behavior_profile: str = Field(..., description="Interaction style (e.g., methodical, chaotic_edge_tester, keyboard_only)")
    goals: list[str] = Field(default_factory=list, description="Primary testing goals for this persona")
    synthetic_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Safe synthetic test data (emails, usernames, viewport dimensions, boundary strings)",
    )
    target_scenarios: list[str] = Field(default_factory=list, description="Scenario IDs assigned to this persona")
    risk_focus: str = Field("", description="Specific failure mode this persona aims to uncover")


class PersonaSuiteResult(BaseModel):
    """Collection of generated QA personas bound to a target application."""

    target_url: str = Field(..., description="Target application URL")
    personas: list[UserPersona] = Field(default_factory=list, description="Generated synthetic QA personas")
    generated_by: str = Field("gemini", description="Origin engine ('gemini' or 'deterministic_fallback')")


class PersonaAgent:
    """
    Synthetic QA persona generator agent.

    Produces diverse user archetypes (happy-path shoppers, chaotic boundary testers,
    keyboard/accessibility navigators) tailored to the application and test plan.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None) -> None:
        self.api_key = (api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
        self.model_name = (model_name or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()
        self._client: Optional[genai.Client] = genai.Client(api_key=self.api_key) if self.api_key else None

    @staticmethod
    def _extract_scenario_ids(test_plan: Optional[Any]) -> list[str]:
        if test_plan is None:
            return ["SCN-001", "SCN-002", "SCN-003"]
        if isinstance(test_plan, dict):
            scenarios = test_plan.get("scenarios", [])
            ids = [s.get("scenario_id") for s in scenarios if isinstance(s, dict) and s.get("scenario_id")]
            return ids or ["SCN-001"]
        scenarios = getattr(test_plan, "scenarios", None) or []
        ids = [getattr(s, "scenario_id", None) for s in scenarios if getattr(s, "scenario_id", None)]
        return ids or ["SCN-001"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    def _invoke_gemini(self, prompt: str) -> PersonaSuiteResult:
        if not self._client:
            raise RuntimeError("Gemini client is not configured.")

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are the Persona Architect Agent for AgentSwarm. Generate synthetic, "
                    "non-real QA testing personas with safe synthetic test inputs and distinct behavioral "
                    "patterns to uncover regressions, broken selectors, and validation bugs."
                ),
                response_mime_type="application/json",
                response_schema=PersonaSuiteResult,
                temperature=0.3,
            ),
        )
        if response.parsed is not None and isinstance(response.parsed, PersonaSuiteResult):
            result = response.parsed
            result.generated_by = "gemini"
            return result
        result = PersonaSuiteResult.model_validate_json(response.text)
        result.generated_by = "gemini"
        return result

    def generate_personas(
        self,
        target_url: str,
        app_context: Optional[str] = None,
        test_plan: Optional[Any] = None,
        count: int = 3,
    ) -> PersonaSuiteResult:
        """Generate a calibrated suite of synthetic QA personas for the target application."""
        bounded_count = max(1, min(count, 8))
        scenario_ids = self._extract_scenario_ids(test_plan)

        if self._client:
            prompt = (
                f"Target URL: {target_url}\n"
                f"Requested Persona Count: {bounded_count}\n"
                f"Available Scenario IDs: {', '.join(scenario_ids)}\n"
                f"Application Context: {(app_context or 'Standard web application')[:3000]}\n\n"
                "Generate diverse QA personas including a happy-path user, an edge-case/chaos explorer, "
                "and an accessibility/keyboard navigator. Ensure all credentials in synthetic_data are clearly synthetic."
            )
            try:
                suite = self._invoke_gemini(prompt)
                suite.target_url = target_url
                return suite
            except Exception as exc:
                logger.warning("[PersonaAgent] Gemini generation failed, using deterministic fallback: %s", exc)

        return self._build_fallback_personas(target_url, scenario_ids, bounded_count)

    @staticmethod
    def _build_fallback_personas(
        target_url: str,
        scenario_ids: list[str],
        count: int,
    ) -> PersonaSuiteResult:
        primary_scn = scenario_ids[0]
        all_scns = scenario_ids[:4]
        templates = [
            UserPersona(
                persona_id="PRS-01",
                name="Methodical Happy-Path User",
                role="Standard Customer / Registered User",
                behavior_profile="methodical",
                goals=[
                    "Complete primary user flow without deviations",
                    "Verify expected confirmation states and navigation transitions",
                ],
                synthetic_data={
                    "email": "synthetic.happy@agentswarm.dev",
                    "full_name": "Alex Synthetic",
                    "viewport": {"width": 1440, "height": 900},
                },
                target_scenarios=all_scns,
                risk_focus="Broken primary CTAs, API 500 errors, or stalled workflows",
            ),
            UserPersona(
                persona_id="PRS-02",
                name="Edge-Case & Chaos Explorer",
                role="Adversarial QA Boundary Tester",
                behavior_profile="chaotic_edge_tester",
                goals=[
                    "Submit empty, malformed, and boundary-length form inputs",
                    "Rapidly double-click interactive buttons and toggle tabs mid-request",
                ],
                synthetic_data={
                    "email": "invalid-email-format",
                    "short_password": "12",
                    "long_name": "X" * 120,
                    "viewport": {"width": 1280, "height": 720},
                },
                target_scenarios=scenario_ids[1:] if len(scenario_ids) > 1 else [primary_scn],
                risk_focus="Unhandled 422 JSON payloads, race conditions, and UI state corruption",
            ),
            UserPersona(
                persona_id="PRS-03",
                name="Accessibility & Keyboard Navigator",
                role="Assistive Technology User",
                behavior_profile="keyboard_accessibility",
                goals=[
                    "Locate interactive controls strictly via ARIA roles and accessible labels",
                    "Verify focus traps, modal Escape handling, and mobile viewport layout",
                ],
                synthetic_data={
                    "email": "a11y.qa@agentswarm.dev",
                    "full_name": "Jordan Aria",
                    "viewport": {"width": 390, "height": 844},
                },
                target_scenarios=all_scns,
                risk_focus="Missing ARIA roles, unlabeled buttons, and brittle CSS class dependencies",
            ),
            UserPersona(
                persona_id="PRS-04",
                name="Unauthenticated Guest Explorer",
                role="Anonymous Visitor",
                behavior_profile="session_boundary",
                goals=[
                    "Probe protected routes without an active JWT session",
                    "Verify clean sign-in prompts and zero silent console exceptions",
                ],
                synthetic_data={
                    "token": None,
                    "viewport": {"width": 1366, "height": 768},
                },
                target_scenarios=[primary_scn],
                risk_focus="Auth guard leaks and unhandled 401 client errors",
            ),
        ]

        return PersonaSuiteResult(
            target_url=target_url,
            personas=templates[:count],
            generated_by="deterministic_fallback",
        )

    def run_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph-compatible node entrypoint for persona generation."""
        target_url = state.get("target_url", "https://example.com")
        app_context = state.get("app_context") or state.get("dom_summary") or state.get("dom_snapshot")
        test_plan = state.get("test_plan")
        count = int(state.get("persona_count", 3))
        suite = self.generate_personas(
            target_url=target_url,
            app_context=app_context,
            test_plan=test_plan,
            count=count,
        )
        return {
            "personas": suite.model_dump(),
            "current_agent": "persona",
        }


def persona_node(state: dict[str, Any]) -> dict[str, Any]:
    """Functional LangGraph node wrapper around PersonaAgent."""
    return PersonaAgent().run_node(state)
