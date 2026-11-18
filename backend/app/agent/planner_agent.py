from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Load environment variables from backend/.env if present
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

AVAILABLE_MODELS: list[dict[str, str]] = [
    {
        "id": "google_genai:gemini-2.5-flash",
        "provider": "google_genai",
        "model_name": "gemini-2.5-flash",
        "label": "Gemini 2.5 Flash — Fast & Recommended",
    },
    {
        "id": "google_genai:gemini-2.5-pro",
        "provider": "google_genai",
        "model_name": "gemini-2.5-pro",
        "label": "Gemini 2.5 Pro — Deep Reasoning",
    },
    {
        "id": "google_genai:gemini-2.0-flash",
        "provider": "google_genai",
        "model_name": "gemini-2.0-flash",
        "label": "Gemini 2.0 Flash",
    },
    {
        "id": "openrouter:google/gemini-2.5-flash",
        "provider": "openrouter",
        "model_name": "google/gemini-2.5-flash",
        "label": "OpenRouter • Gemini 2.5 Flash",
    },
    {
        "id": "openrouter:anthropic/claude-3.5-sonnet",
        "provider": "openrouter",
        "model_name": "anthropic/claude-3.5-sonnet",
        "label": "OpenRouter • Claude 3.5 Sonnet",
    },
    {
        "id": "openrouter:openai/gpt-4o-mini",
        "provider": "openrouter",
        "model_name": "openai/gpt-4o-mini",
        "label": "OpenRouter • GPT-4o Mini",
    },
    {
        "id": "openrouter:deepseek/deepseek-chat",
        "provider": "openrouter",
        "model_name": "deepseek/deepseek-chat",
        "label": "OpenRouter • DeepSeek V3",
    },
    {
        "id": "openai:gpt-4o-mini",
        "provider": "openai",
        "model_name": "gpt-4o-mini",
        "label": "OpenAI • GPT-4o Mini",
    },
    {
        "id": "openai:gpt-4o",
        "provider": "openai",
        "model_name": "gpt-4o",
        "label": "OpenAI • GPT-4o",
    },
    {
        "id": "deterministic:deterministic-fallback",
        "provider": "deterministic",
        "model_name": "deterministic-fallback",
        "label": "Deterministic DOM Engine (Offline / Instant)",
    },
]


def _get_configured_providers() -> dict[str, bool]:
    return {
        "google_genai": bool((os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()),
        "openrouter": bool((os.getenv("OPENROUTER_API_KEY") or "").strip()),
        "openai": bool((os.getenv("OPENAI_API_KEY") or "").strip()),
        "deterministic": True,
    }


def list_available_models() -> dict[str, Any]:
    """Return catalog of selectable LLM providers/models and local API key readiness."""
    configured_providers = _get_configured_providers()
    env_provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if env_provider in ("google", "gemini"):
        env_provider = "google_genai"

    if env_provider and configured_providers.get(env_provider):
        default_provider = env_provider
    elif configured_providers["google_genai"]:
        default_provider = "google_genai"
    elif configured_providers["openrouter"]:
        default_provider = "openrouter"
    elif configured_providers["openai"]:
        default_provider = "openai"
    else:
        default_provider = "deterministic"

    env_model = (os.getenv("LLM_MODEL") or "").strip()
    if env_model:
        default_model = env_model
    elif default_provider == "google_genai":
        default_model = (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()
    elif default_provider == "openrouter":
        default_model = (os.getenv("OPENROUTER_MODEL") or "google/gemini-2.5-flash").strip()
    elif default_provider == "openai":
        default_model = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
    else:
        default_model = "deterministic-fallback"

    models_out = [
        {
            **entry,
            "configured": bool(configured_providers.get(entry["provider"], False)),
        }
        for entry in AVAILABLE_MODELS
    ]

    return {
        "default_provider": default_provider,
        "default_model": default_model,
        "configured_providers": configured_providers,
        "models": models_out,
    }


def resolve_chat_model(
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
) -> Optional[BaseChatModel]:
    """
    Create a model-agnostic LangChain BaseChatModel from explicit arguments or environment variables.

    Supports:
    - google_genai (GOOGLE_API_KEY or GEMINI_API_KEY)
    - openai (OPENAI_API_KEY)
    - openrouter (OPENROUTER_API_KEY via base_url='https://openrouter.ai/api/v1')
    - deterministic (returns None immediately for offline rule-based execution)
    Returns None if no valid API key is configured so agents can fall back cleanly.
    """
    parsed_model = (model_name or "").strip() or None
    parsed_provider = (provider or "").strip().lower() or None

    if parsed_model and not parsed_provider and ":" in parsed_model:
        prefix, rest = parsed_model.split(":", 1)
        if prefix.strip().lower() in (
            "google",
            "gemini",
            "google_genai",
            "openai",
            "openrouter",
            "deterministic",
            "local",
            "offline",
        ):
            parsed_provider = prefix.strip().lower()
            parsed_model = rest.strip() or None

    if parsed_provider in ("deterministic", "local", "offline") or parsed_model == "deterministic-fallback":
        return None

    raw_provider = (parsed_provider or os.getenv("LLM_PROVIDER") or "").strip().lower()
    explicit_key = (api_key or "").strip()

    if not raw_provider:
        if explicit_key.startswith("sk-or-"):
            raw_provider = "openrouter"
        elif explicit_key.startswith("AIza"):
            raw_provider = "google_genai"
        elif explicit_key.startswith("sk-"):
            raw_provider = "openai"
        elif os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            raw_provider = "google_genai"
        elif os.getenv("OPENROUTER_API_KEY"):
            raw_provider = "openrouter"
        elif os.getenv("OPENAI_API_KEY"):
            raw_provider = "openai"

    if not raw_provider or raw_provider in ("deterministic", "local", "offline"):
        return None

    try:
        if raw_provider in ("google", "gemini", "google_genai"):
            resolved_key = (explicit_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
            if not resolved_key:
                return None
            resolved_model = (parsed_model or os.getenv("LLM_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()
            return init_chat_model(
                resolved_model,
                model_provider="google_genai",
                api_key=resolved_key,
                temperature=temperature,
            )

        if raw_provider == "openrouter":
            resolved_key = (explicit_key or os.getenv("OPENROUTER_API_KEY") or "").strip()
            if not resolved_key:
                return None
            resolved_model = (
                parsed_model or os.getenv("LLM_MODEL") or os.getenv("OPENROUTER_MODEL") or "google/gemini-2.5-flash"
            ).strip()
            return init_chat_model(
                resolved_model,
                model_provider="openai",
                api_key=resolved_key,
                base_url=OPENROUTER_BASE_URL,
                temperature=temperature,
            )

        if raw_provider == "openai":
            resolved_key = (explicit_key or os.getenv("OPENAI_API_KEY") or "").strip()
            if not resolved_key:
                return None
            resolved_model = (parsed_model or os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
            return init_chat_model(
                resolved_model,
                model_provider="openai",
                api_key=resolved_key,
                temperature=temperature,
            )

        # Generic provider pass-through supported by LangChain init_chat_model
        resolved_model = (parsed_model or os.getenv("LLM_MODEL") or "").strip()
        if not resolved_model:
            return None
        kwargs: dict[str, Any] = {"model_provider": raw_provider, "temperature": temperature}
        if explicit_key:
            kwargs["api_key"] = explicit_key
        return init_chat_model(resolved_model, **kwargs)
    except ImportError as exc:
        logger.warning("[AgentSwarm LLM] Missing LangChain provider package for '%s': %s", raw_provider, exc)
        return None
    except Exception as exc:
        logger.warning("[AgentSwarm LLM] Failed to initialize provider '%s': %s", raw_provider, exc)
        return None


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
    generated_by: str = Field("langchain", description="Origin engine identifier")


class PlannerGraphState(TypedDict, total=False):
    """LangGraph state contract for PlannerAgent workflow."""

    target_url: str
    normalized_url: str
    dom_summary: Optional[str]
    objectives: list[str]
    prompt_input: dict[str, str]
    test_plan: TestPlanResult


class PlannerAgent:
    """
    Model-agnostic autonomous QA test planning agent powered by LangChain & LangGraph.

    Uses `ChatPromptTemplate | llm.with_structured_output(TestPlanResult)` inside a
    compiled LangGraph `StateGraph` (`prepare_input -> generate_plan -> finalize_plan`).
    Works with Gemini, OpenAI, OpenRouter, or any injected LangChain BaseChatModel.
    """

    SYSTEM_PROMPT = (
        "You are the Lead QA Planner Agent for AgentSwarm. Design concise, realistic, "
        "non-destructive end-to-end browser test plans using semantic accessibility hints. "
        "Only target authorized/staging workflows and never invent destructive actions."
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
            temperature=0.2,
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM_PROMPT),
                (
                    "human",
                    "Target URL: {target_url}\n\n"
                    "Testing Objectives:\n{objectives_text}\n\n"
                    "DOM / Route Summary:\n{dom_summary}\n\n"
                    "Generate 3 to 5 prioritized end-to-end test scenarios covering smoke navigation, "
                    "primary user workflow, authentication/session state, and negative edge cases.",
                ),
            ]
        )
        self._chain = (self._prompt | self._llm.with_structured_output(TestPlanResult)) if self._llm else None
        self._graph = self.build_graph()

    @staticmethod
    def _normalize_url(target_url: str) -> str:
        url = (target_url or "").strip()
        if not url:
            return "https://example.com"
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        return url

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    def _invoke_chain(self, prompt_input: dict[str, str]) -> TestPlanResult:
        if not self._chain:
            raise RuntimeError("LangChain structured output chain is not configured.")
        result = self._chain.invoke(prompt_input)
        if not isinstance(result, TestPlanResult):
            result = TestPlanResult.model_validate(result)
        
        return result

    def _node_prepare_input(self, state: PlannerGraphState) -> dict[str, Any]:
        normalized_url = self._normalize_url(state.get("target_url", ""))
        objectives = state.get("objectives") or []
        objectives_text = "\n".join(f"- {obj}" for obj in objectives) if objectives else "- Core smoke and user workflows"
        dom_summary = (state.get("dom_summary") or "No DOM summary provided")[:4000]
        return {
            "normalized_url": normalized_url,
            "prompt_input": {
                "target_url": normalized_url,
                "objectives_text": objectives_text,
                "dom_summary": dom_summary,
            },
        }

    def _node_generate_plan(self, state: PlannerGraphState) -> dict[str, Any]:
        normalized_url = state["normalized_url"]
        if self._chain:
            try:
                plan = self._invoke_chain(state["prompt_input"])
                return {"test_plan": plan}
            except Exception as exc:
                logger.warning("[PlannerAgent] LangChain invocation failed, using deterministic fallback: %s", exc)

        plan = self._build_fallback_plan(
            target_url=normalized_url,
            dom_summary=state.get("dom_summary"),
            objectives=state.get("objectives"),
        )
        return {"test_plan": plan}

    @staticmethod
    def _node_finalize_plan(state: PlannerGraphState) -> dict[str, Any]:
        plan = state["test_plan"]
        plan.target_url = state["normalized_url"]
        return {"test_plan": plan}

    def build_graph(self):
        """Build and compile the internal LangGraph workflow for test planning."""
        builder = StateGraph(PlannerGraphState)
        builder.add_node("prepare_input", self._node_prepare_input)
        builder.add_node("generate_plan", self._node_generate_plan)
        builder.add_node("finalize_plan", self._node_finalize_plan)

        builder.add_edge(START, "prepare_input")
        builder.add_edge("prepare_input", "generate_plan")
        builder.add_edge("generate_plan", "finalize_plan")
        builder.add_edge("finalize_plan", END)
        return builder.compile()

    def create_test_plan(
        self,
        target_url: str,
        dom_summary: Optional[str] = None,
        objectives: Optional[list[str]] = None,
    ) -> TestPlanResult:
        """Execute the compiled LangGraph planner workflow to produce a structured TestPlanResult."""
        result_state = self._graph.invoke(
            {
                "target_url": target_url,
                "dom_summary": dom_summary,
                "objectives": objectives or [],
            }
        )
        return result_state["test_plan"]

    @staticmethod
    def _extract_dom_candidates(dom_summary: Optional[str]) -> list[dict[str, str]]:
        """Extract interactive element hints from either raw HTML or BrowserToolkit summary text."""
        if not dom_summary:
            return []
        candidates: list[dict[str, str]] = []
        if "<" in dom_summary and ">" in dom_summary:
            try:
                from ..browser.browser_tools import BrowserToolkit

                for el in BrowserToolkit.extract_interactive_elements(dom_summary):
                    candidates.append(
                        {
                            "tag": el.tag,
                            "role": el.role,
                            "name": el.accessible_name or el.text or el.placeholder or el.element_id,
                            "locator": el.recommended_locator or (f"#{el.element_id}" if el.element_id else el.tag),
                            "id": el.element_id,
                        }
                    )
            except Exception:
                pass
        if not candidates:
            pattern = re.compile(
                r"^-\s*<([a-z0-9]+)>\s+role=([a-z0-9_-]+)(?:\s+name='([^']*)')?(?:\s+id=#([^\s]+))?\s*->\s*(.+)$",
                re.I,
            )
            for line in dom_summary.splitlines():
                m = pattern.match(line.strip())
                if m:
                    candidates.append(
                        {
                            "tag": m.group(1).lower(),
                            "role": m.group(2).lower(),
                            "name": (m.group(3) or "").strip(),
                            "id": (m.group(4) or "").strip(),
                            "locator": m.group(5).strip(),
                        }
                    )
        return candidates

    @classmethod
    def _build_fallback_plan(
        cls,
        target_url: str,
        dom_summary: Optional[str],
        objectives: Optional[list[str]],
    ) -> TestPlanResult:
        dom_lower = (dom_summary or "").lower()
        candidates = cls._extract_dom_candidates(dom_summary)
        buttons = [c for c in candidates if c["role"] == "button" or c["tag"] == "button"]
        links = [c for c in candidates if c["role"] == "link" or c["tag"] == "a"]
        inputs = [c for c in candidates if c["role"] in ("textbox", "combobox") or c["tag"] in ("input", "textarea", "select")]

        scenarios: list[TestScenario] = []

        # Scenario 1: Landing & Primary Element Smoke (using real DOM targets when discovered)
        scn1_steps: list[TestStepPlan] = [
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
                description="Verify primary document landmark and interactive controls are visible",
                target_hint="main, body, h1",
                expected_outcome="Primary application shell and heading render visibly",
            ),
        ]
        if links:
            first_link = links[0]
            label = first_link["name"] or "primary link"
            scn1_steps.append(
                TestStepPlan(
                    step_number=3,
                    action="assert_visible",
                    description=f"Verify discovered link '{label[:45]}' is visible and clickable",
                    target_hint=first_link["locator"],
                    expected_outcome=f"Link '{label[:45]}' is accessible in the viewport",
                )
            )
            scn1_steps.append(
                TestStepPlan(
                    step_number=4,
                    action="click",
                    description=f"Click discovered link '{label[:45]}'",
                    target_hint=first_link["locator"],
                    expected_outcome="Navigation or interaction executes without console or network errors",
                )
            )
        elif buttons:
            first_btn = buttons[0]
            label = first_btn["name"] or "primary button"
            scn1_steps.append(
                TestStepPlan(
                    step_number=3,
                    action="click",
                    description=f"Click primary discovered button '{label[:45]}'",
                    target_hint=first_btn["locator"],
                    expected_outcome="Button responds cleanly without runtime exception",
                )
            )

        scenarios.append(
            TestScenario(
                scenario_id="SCN-001",
                title="Landing Route Render & Discovered Navigation Smoke",
                category="smoke",
                priority="critical",
                description="Verify initial document load, live DOM interactive elements, and zero blocking errors.",
                preconditions=["Application URL is reachable", "Clean browser context"],
                steps=scn1_steps,
                expected_result="Application loads cleanly with accessible interactive controls.",
            )
        )

        # Scenario 2: Discovered Forms / Inputs OR Authentication Flow
        if inputs:
            form_steps: list[TestStepPlan] = [
                TestStepPlan(
                    step_number=1,
                    action="navigate",
                    description=f"Ensure target page {target_url} is loaded",
                    target_hint=target_url,
                    expected_outcome="Page form inputs are ready",
                )
            ]
            step_num = 2
            for inp in inputs[:3]:
                inp_label = inp["name"] or inp["id"] or "input field"
                sample_val = (
                    "qa.synthetic@agentswarm.dev"
                    if "email" in inp_label.lower()
                    else ("SyntheticPass!2026" if "pass" in inp_label.lower() else "AgentSwarm QA Test")
                )
                form_steps.append(
                    TestStepPlan(
                        step_number=step_num,
                        action="fill",
                        description=f"Populate '{inp_label[:40]}' input with synthetic QA value",
                        target_hint=inp["locator"],
                        input_value=sample_val,
                        expected_outcome=f"Input '{inp_label[:40]}' accepts synthetic text cleanly",
                    )
                )
                step_num += 1
            if buttons:
                submit_btn = buttons[0]
                btn_label = submit_btn["name"] or "Submit"
                form_steps.append(
                    TestStepPlan(
                        step_number=step_num,
                        action="click",
                        description=f"Click '{btn_label[:40]}' button to submit form",
                        target_hint=submit_btn["locator"],
                        expected_outcome="Submission triggers expected state change without 5xx error",
                    )
                )
            scenarios.append(
                TestScenario(
                    scenario_id="SCN-002",
                    title="Discovered Form Inputs & Submission Journey",
                    category="e2e",
                    priority="high",
                    description="Populate real discovered form inputs with synthetic QA data and verify interaction.",
                    preconditions=["Form elements present in live DOM"],
                    steps=form_steps,
                    expected_result="All discovered form controls accept synthetic inputs and submit cleanly.",
                )
            )
        elif len(buttons) >= 2 or "sign in" in dom_lower or "login" in dom_lower:
            btn_target = buttons[0]["locator"] if buttons else "button"
            btn_name = (buttons[0]["name"] or "Action") if buttons else "Action"
            scenarios.append(
                TestScenario(
                    scenario_id="SCN-002",
                    title="Interactive Controls & CTA Verification",
                    category="auth" if ("sign in" in dom_lower or "login" in dom_lower) else "e2e",
                    priority="high",
                    description="Exercise discovered interactive buttons and verify visual state transitions.",
                    preconditions=["Target page rendered"],
                    steps=[
                        TestStepPlan(
                            step_number=1,
                            action="navigate",
                            description=f"Open {target_url}",
                            target_hint=target_url,
                            expected_outcome="Page rendered",
                        ),
                        TestStepPlan(
                            step_number=2,
                            action="assert_visible",
                            description=f"Verify button '{btn_name[:40]}' is visible",
                            target_hint=btn_target,
                            expected_outcome="Interactive button is visible",
                        ),
                        TestStepPlan(
                            step_number=3,
                            action="click",
                            description=f"Click '{btn_name[:40]}' control",
                            target_hint=btn_target,
                            expected_outcome="UI transitions cleanly without console error",
                        ),
                    ],
                    expected_result="Discovered buttons respond cleanly.",
                )
            )

        # Scenario 3: Commerce / Core Interactive Workflow Verification
        is_commerce = any(token in dom_lower for token in ("checkout", "cart", "product", "price", "order"))
        if is_commerce or not scenarios or len(scenarios) < 2:
            cta_hint = (
                buttons[-1]["locator"]
                if buttons
                else "button[type='submit'], .cta-primary, getByRole('button', { name: /proceed|deploy|start/i })"
            )
            cta_label = (buttons[-1]["name"] or "primary CTA") if buttons else "primary CTA"
            scenarios.append(
                TestScenario(
                    scenario_id=f"SCN-{len(scenarios) + 1:03d}",
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
                            action="navigate",
                            description=f"Ensure application context at {target_url}",
                            target_hint=target_url,
                            expected_outcome="Application active",
                        ),
                        TestStepPlan(
                            step_number=2,
                            action="click",
                            description=f"Trigger '{cta_label[:40]}' action button",
                            target_hint=cta_hint,
                            expected_outcome="Workflow transitions to active processing or next step",
                        ),
                        TestStepPlan(
                            step_number=3,
                            action="assert_visible",
                            description="Confirm workflow status output updates visibly",
                            target_hint="main, body, [role='status']",
                            expected_outcome="Updated application state renders without selector timeout",
                        ),
                    ],
                    expected_result="Core interactive workflow advances and updates DOM state reliably.",
                )
            )

        if objectives:
            for objective in objectives[:3]:
                next_idx = len(scenarios) + 1
                obj_hint = buttons[0]["locator"] if buttons else (links[0]["locator"] if links else "main, body")
                scenarios.append(
                    TestScenario(
                        scenario_id=f"SCN-{next_idx:03d}",
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
                                target_hint=obj_hint,
                                expected_outcome="Objective criteria satisfied without errors",
                            ),
                        ],
                        expected_result=f"Objective verified: {objective}",
                    )
                )

        return TestPlanResult(
            target_url=target_url,
            summary=f"Generated {len(scenarios)} DOM-aware QA test scenarios for {target_url}.",
            scenarios=scenarios,
            generated_by="deterministic_fallback",
        )

    def run_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph parent-graph node entrypoint for test planning."""
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
