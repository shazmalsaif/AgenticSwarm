from __future__ import annotations

import logging
from typing import Any, Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from .planner_agent import resolve_chat_model

logger = logging.getLogger(__name__)


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
    generated_by: str = Field("langchain", description="Origin engine identifier")


class PersonaGraphState(TypedDict, total=False):
    """LangGraph state contract for PersonaAgent workflow."""

    target_url: str
    app_context: Optional[str]
    test_plan: Optional[Any]
    count: int
    bounded_count: int
    scenario_ids: list[str]
    persona_suite: PersonaSuiteResult


class PersonaAgent:
    """
    Model-agnostic synthetic QA persona architect powered by LangChain & LangGraph.

    Uses `ChatPromptTemplate | llm.with_structured_output(PersonaSuiteResult)` inside
    a compiled LangGraph `StateGraph` (`extract_scenarios -> synthesize_personas`).
    """

    SYSTEM_PROMPT = (
        "You are the Persona Architect Agent for AgentSwarm. Generate synthetic, "
        "non-real QA testing personas with safe synthetic test inputs and distinct behavioral "
        "patterns to uncover regressions, broken selectors, and validation bugs."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        llm: Optional[BaseChatModel] = None,
    ) -> None:
        self._llm: Optional[BaseChatModel] = llm or resolve_chat_model(
            model_name=model_name,
            provider=provider,
            api_key=api_key,
            temperature=0.3,
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM_PROMPT),
                (
                    "human",
                    "Target URL: {target_url}\n"
                    "Requested Persona Count: {bounded_count}\n"
                    "Available Scenario IDs: {scenario_ids_text}\n"
                    "Application Context: {app_context}\n\n"
                    "Generate diverse QA personas including a happy-path user, an edge-case/chaos explorer, "
                    "and an accessibility/keyboard navigator. Ensure all credentials in synthetic_data are clearly synthetic.",
                ),
            ]
        )
        self._chain = (self._prompt | self._llm.with_structured_output(PersonaSuiteResult)) if self._llm else None
        self._graph = self.build_persona_graph()

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
    def _invoke_chain(self, prompt_input: dict[str, Any]) -> PersonaSuiteResult:
        if not self._chain:
            raise RuntimeError("LangChain structured output chain is not configured.")
        result = self._chain.invoke(prompt_input)
        if not isinstance(result, PersonaSuiteResult):
            result = PersonaSuiteResult.model_validate(result)
        result.generated_by = "langchain"
        return result

    def _node_extract_scenarios(self, state: PersonaGraphState) -> dict[str, Any]:
        bounded_count = max(1, min(int(state.get("count", 3)), 8))
        scenario_ids = self._extract_scenario_ids(state.get("test_plan"))
        return {
            "bounded_count": bounded_count,
            "scenario_ids": scenario_ids,
        }

    def _node_synthesize_personas(self, state: PersonaGraphState) -> dict[str, Any]:
        target_url = state.get("target_url", "https://example.com")
        bounded_count = state["bounded_count"]
        scenario_ids = state["scenario_ids"]
        app_context = (state.get("app_context") or "Standard web application")[:3000]

        if self._chain:
            try:
                suite = self._invoke_chain(
                    {
                        "target_url": target_url,
                        "bounded_count": bounded_count,
                        "scenario_ids_text": ", ".join(scenario_ids),
                        "app_context": app_context,
                    }
                )
                suite.target_url = target_url
                return {"persona_suite": suite}
            except Exception as exc:
                logger.warning("[PersonaAgent] LangChain generation failed, using deterministic fallback: %s", exc)

        suite = self._build_fallback_personas(target_url, scenario_ids, bounded_count)
        return {"persona_suite": suite}

    def build_persona_graph(self):
        """Build and compile the internal LangGraph workflow for persona synthesis."""
        builder = StateGraph(PersonaGraphState)
        builder.add_node("extract_scenarios", self._node_extract_scenarios)
        builder.add_node("synthesize_personas", self._node_synthesize_personas)

        builder.add_edge(START, "extract_scenarios")
        builder.add_edge("extract_scenarios", "synthesize_personas")
        builder.add_edge("synthesize_personas", END)
        return builder.compile()

    def generate_personas(
        self,
        target_url: str,
        app_context: Optional[str] = None,
        test_plan: Optional[Any] = None,
        count: int = 3,
    ) -> PersonaSuiteResult:
        """Execute the compiled LangGraph persona workflow and return PersonaSuiteResult."""
        result_state = self._graph.invoke(
            {
                "target_url": target_url,
                "app_context": app_context,
                "test_plan": test_plan,
                "count": count,
            }
        )
        return result_state["persona_suite"]

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
        """LangGraph parent-graph node entrypoint for persona generation."""
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
