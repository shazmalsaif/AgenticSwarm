from __future__ import annotations

import logging
import re
from typing import Any, Literal, Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ..browser.browser_tools import BrowserToolkit
from .planner_agent import invoke_with_model_fallback, resolve_chat_model

logger = logging.getLogger(__name__)


class BrokenLocatorContext(BaseModel):
    """Context supplied to HealingAgent for synthesizing resilient replacement selectors."""

    failure_id: Optional[str] = None
    failed_selector: str
    action_description: Optional[str] = None
    element_role: Optional[str] = None
    accessible_name: Optional[str] = None
    element_text: Optional[str] = None
    dom_excerpt: Optional[str] = None
    blocking_overlay: Optional[str] = None
    target_url: str = ""
    error_message: Optional[str] = None


class FrameworkPatch(BaseModel):
    """Framework-specific healed locator and drop-in test code snippet."""

    framework: Literal["playwright", "cypress", "selenium"]
    healed_selector: str
    code_snippet: str
    explanation: str


class HealingSuggestion(BaseModel):
    """Synthesized semantic self-healing patch across Playwright, Cypress, and Selenium."""

    failure_id: str
    failed_selector: str
    primary_healed_locator: str
    locator_strategy: str = "aria_role"
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    playwright: FrameworkPatch
    cypress: FrameworkPatch
    selenium: FrameworkPatch


class HealingReport(BaseModel):
    """Consolidated self-healing report for all eligible selector failures."""

    target_url: str = ""
    total_healed: int
    suggestions: list[HealingSuggestion] = Field(default_factory=list)
    generated_by: str = Field("langchain", description="Origin engine identifier")


class HealingGraphState(TypedDict, total=False):
    target_url: str
    items: list[Any]
    coerced_items: list[BrokenLocatorContext]
    healing_report: HealingReport


class SwarmGraphState(TypedDict, total=False):
    """End-to-end multi-agent LangGraph state connecting all 5 AgentSwarm agents."""

    target_url: str
    dom_summary: Optional[str]
    dom_snapshot: Optional[str]
    app_context: Optional[str]
    objectives: list[str]
    persona_count: int
    browser_session: Optional[Any]
    max_steps_per_goal: int
    use_live_browser: bool
    record_traces: bool
    run_id: Optional[str]
    stream_id: Optional[str]
    test_plan: dict[str, Any]
    planner_error: Optional[str]
    personas: dict[str, Any]
    navigation_results: list[dict[str, Any]]
    raw_failures: list[dict[str, Any]]
    triage_report: dict[str, Any]
    healing_candidates: list[dict[str, Any]]
    needs_healing: bool
    healing_report: dict[str, Any]
    current_agent: str


class HealingAgent:
    """
    Model-agnostic self-healing locator synthesis agent powered by LangChain & LangGraph.

    Synthesizes accessible ARIA locators (`getByRole`/`getByLabel`/`getByText`) and explicitly waits
    for blocking overlays (`#intro-overlay` / dialogs) to clear rather than silently skipping hidden targets.
    """

    SYSTEM_PROMPT = (
        "You are the Self-Healing Locator Agent for AgentSwarm. Synthesize resilient, "
        "semantic ARIA-based locators (preferring getByRole / getByLabel / getByText) that survive CSS changes.\n"
        "STRICT SAFETY RULE: NEVER emit 'skip if not visible', 'if (await ...isVisible())', or swallowed '.catch()' blocks. "
        "If an overlay such as #intro-overlay may block interaction, explicitly wait for it to become hidden/detached "
        "(e.g. await page.locator('#intro-overlay').waitFor({ state: 'hidden', timeout: 10000 })) before interacting, "
        "so failures surface loudly instead of skipping."
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
                    "Target URL: {target_url}\n"
                    "Broken Selector: {failed_selector}\n"
                    "Blocking Overlay: {blocking_overlay}\n"
                    "Action Description: {action_description}\n"
                    "Element Role Hint: {element_role}\n"
                    "Accessible Name Hint: {accessible_name}\n"
                    "Element Text Hint: {element_text}\n"
                    "Error Message: {error_message}\n"
                    "DOM Excerpt:\n{dom_excerpt}\n",
                ),
            ]
        )
        self._chain = (self._prompt | self._llm.with_structured_output(HealingSuggestion)) if self._llm else None
        self._graph = self.build_healing_graph()

    @staticmethod
    def _coerce_context(context: BrokenLocatorContext | dict[str, Any], index: int = 1) -> BrokenLocatorContext:
        ctx = context.model_copy() if isinstance(context, BrokenLocatorContext) else BrokenLocatorContext.model_validate(context)
        if not ctx.failure_id:
            ctx.failure_id = f"HEAL-{index:03d}"
        return ctx

    @staticmethod
    def _has_overlay_context(ctx: BrokenLocatorContext) -> Optional[str]:
        if ctx.blocking_overlay:
            return ctx.blocking_overlay
        hay = f"{ctx.dom_excerpt or ''} {ctx.error_message or ''} {ctx.failed_selector}".lower()
        if "intro-overlay" in hay or "intro overlay" in hay:
            return "#intro-overlay"
        return None

    @classmethod
    def _enforce_no_silent_skip(cls, suggestion: HealingSuggestion, ctx: BrokenLocatorContext) -> HealingSuggestion:
        """Guarantee healed patches NEVER silently skip hidden buttons when blocking overlays may be present."""
        overlay = cls._has_overlay_context(ctx)
        forbidden = ("skip if", "if (await", ".isvisible())", ".catch(() =>", "pass  # skip")

        def _clean_pw(code: str, selector: str) -> str:
            low = code.lower()
            cleaned = code
            if any(tok in low for tok in forbidden):
                cleaned = (
                    f"const target = page.{selector} if not selector.startswith('page.') else {selector};\n"
                    f"await target.waitFor({{ state: 'visible', timeout: 9000 }});\n"
                    f"await target.click();"
                ) if False else (
                    f"const target = {selector if selector.startswith('page.') else f'page.{selector}'};\n"
                    f"await target.waitFor({{ state: 'visible', timeout: 9000 }});\n"
                    f"await target.click();"
                )
            if overlay and "waitFor" not in cleaned and overlay not in cleaned:
                cleaned = f"await page.locator('{overlay}').waitFor({{ state: 'hidden', timeout: 10000 }});\n" + cleaned
            elif overlay and "hidden" not in cleaned:
                cleaned = f"await page.locator('{overlay}').waitFor({{ state: 'hidden', timeout: 10000 }});\n" + cleaned
            return cleaned

        def _clean_cy(code: str, selector: str) -> str:
            low = code.lower()
            cleaned = f"{selector}.should('be.visible').click();" if any(t in low for t in forbidden) else code
            if overlay and overlay not in cleaned:
                cleaned = f"cy.get('{overlay}', {{ timeout: 10000 }}).should('not.be.visible');\n" + cleaned
            return cleaned

        def _clean_sel(code: str, selector: str) -> str:
            low = code.lower()
            cleaned = (
                f"target = WebDriverWait(driver, 9).until(EC.element_to_be_clickable(({selector})))\ntarget.click()"
                if any(t in low for t in forbidden)
                else code
            )
            if overlay and overlay not in cleaned:
                ov_id = overlay.lstrip("#")
                cleaned = (
                    f"WebDriverWait(driver, 10).until(EC.invisibility_of_element_located((By.ID, '{ov_id}')))\n"
                    + cleaned
                )
            return cleaned

        return suggestion.model_copy(
            update={
                "playwright": suggestion.playwright.model_copy(
                    update={"code_snippet": _clean_pw(suggestion.playwright.code_snippet, suggestion.playwright.healed_selector)}
                ),
                "cypress": suggestion.cypress.model_copy(
                    update={"code_snippet": _clean_cy(suggestion.cypress.code_snippet, suggestion.cypress.healed_selector)}
                ),
                "selenium": suggestion.selenium.model_copy(
                    update={"code_snippet": _clean_sel(suggestion.selenium.code_snippet, suggestion.selenium.healed_selector)}
                ),
            }
        )

    def _invoke_chain(self, ctx: BrokenLocatorContext) -> HealingSuggestion:
        prompt_input = {
            "failure_id": ctx.failure_id or "HEAL-001",
            "target_url": ctx.target_url,
            "failed_selector": ctx.failed_selector,
            "blocking_overlay": ctx.blocking_overlay or "None",
            "action_description": ctx.action_description or "Interact with UI element",
            "element_role": ctx.element_role or "N/A",
            "accessible_name": ctx.accessible_name or "N/A",
            "element_text": ctx.element_text or "N/A",
            "error_message": ctx.error_message or "N/A",
            "dom_excerpt": (ctx.dom_excerpt or "N/A")[:2500],
        }
        result, _ = invoke_with_model_fallback(
            self._prompt,
            HealingSuggestion,
            prompt_input,
            primary_chain=self._chain,
            provider=self._provider,
            model_name=self._model_name,
            api_key=self._api_key,
            temperature=0.1,
            allow_fallback=not self._custom_llm_injected,
        )
        if not isinstance(result, HealingSuggestion):
            result = HealingSuggestion.model_validate(result)
        result.failure_id = ctx.failure_id or result.failure_id
        result.failed_selector = ctx.failed_selector
        return self._enforce_no_silent_skip(result, ctx)

    def _node_parse_dom_context(self, state: HealingGraphState) -> dict[str, Any]:
        target_url = state.get("target_url", "")
        coerced: list[BrokenLocatorContext] = []
        for idx, raw in enumerate(state.get("items") or [], start=1):
            ctx = self._coerce_context(raw, index=idx)
            if target_url and not ctx.target_url:
                ctx.target_url = target_url
            if ctx.dom_excerpt and (not ctx.element_role or not ctx.accessible_name):
                role, name, _ = self._infer_target_from_dom(ctx)
                ctx.element_role = ctx.element_role or role
                ctx.accessible_name = ctx.accessible_name or name
            coerced.append(ctx)
        return {"coerced_items": coerced}

    def _node_synthesize_patches(self, state: HealingGraphState) -> dict[str, Any]:
        target_url = state.get("target_url", "")
        suggestions: list[HealingSuggestion] = []
        used_llm = False
        for ctx in state.get("coerced_items") or []:
            if self._chain:
                try:
                    suggestions.append(self._invoke_chain(ctx))
                    used_llm = True
                    continue
                except Exception as exc:
                    logger.warning("[HealingAgent] Item %s LangChain synthesis failed, using fallback: %s", ctx.failure_id, exc)
            suggestions.append(self._synthesize_deterministically(ctx))
        return {
            "healing_report": HealingReport(
                target_url=target_url,
                total_healed=len(suggestions),
                suggestions=suggestions,
                generated_by="langchain" if used_llm else "deterministic_fallback",
            )
        }

    def build_healing_graph(self):
        builder = StateGraph(HealingGraphState)
        builder.add_node("parse_dom_context", self._node_parse_dom_context)
        builder.add_node("synthesize_patches", self._node_synthesize_patches)
        builder.add_edge(START, "parse_dom_context")
        builder.add_edge("parse_dom_context", "synthesize_patches")
        builder.add_edge("synthesize_patches", END)
        return builder.compile()

    def heal_locator(self, context: BrokenLocatorContext | dict[str, Any]) -> HealingSuggestion:
        return self._graph.invoke({"items": [context], "target_url": ""})["healing_report"].suggestions[0]

    def heal_batch(self, items: list[BrokenLocatorContext | dict[str, Any]], target_url: str = "") -> HealingReport:
        return self._graph.invoke({"items": items, "target_url": target_url})["healing_report"]

    @staticmethod
    def _infer_target_from_dom(ctx: BrokenLocatorContext) -> tuple[str, str, str]:
        role = (ctx.element_role or "").strip().lower()
        name = (ctx.accessible_name or ctx.element_text or "").strip()
        html_tag = "button"

        if ctx.dom_excerpt and "<" in ctx.dom_excerpt:
            elements = BrowserToolkit.extract_interactive_elements(ctx.dom_excerpt)
            if elements:
                hint_tokens = set(
                    re.findall(r"[a-z0-9]+", f"{ctx.failed_selector} {ctx.action_description or ''} {ctx.element_text or ''}".lower())
                ) - {"btn", "button", "cta", "v1", "v2", "input", "legacy", "the", "and", "click"}
                best = max(
                    elements,
                    key=lambda el: (
                        sum(1 for tok in hint_tokens if tok in f"{el.accessible_name} {el.text} {el.placeholder} {el.element_id} {el.tag}".lower())
                        + (1 if (el.accessible_name or el.text) else 0)
                    ),
                )
                html_tag = best.tag if best.tag in ("button", "a", "input", "select", "textarea") else "button"
                role = role or best.role
                name = name or best.accessible_name or best.text or best.placeholder

        if not role:
            desc_low = f"{ctx.failed_selector} {ctx.action_description or ''}".lower()
            role, html_tag = (
                ("textbox", "input") if any(t in desc_low for t in ("email", "password", "input", "fill", "search"))
                else (("link", "a") if any(t in desc_low for t in ("link", "href", "navigate")) else ("button", "button"))
            )
        elif role == "link":
            html_tag = "a"
        elif role in ("textbox", "combobox"):
            html_tag = "input"

        if not name:
            if ctx.action_description:
                name = re.sub(r"^(click|press|fill|submit|trigger|open)\s+(the\s+)?", "", ctx.action_description.strip(), flags=re.I)[:40].strip() or "Submit"
            else:
                tokens = [t for t in re.findall(r"[a-z]+", ctx.failed_selector.lower()) if t not in ("btn", "button", "cta", "v1", "v2", "legacy", "el", "id")]
                name = " ".join(tokens).title() or "Continue"
        return role, name, html_tag

    def _synthesize_deterministically(self, ctx: BrokenLocatorContext) -> HealingSuggestion:
        role, accessible_name, html_tag = self._infer_target_from_dom(ctx)
        safe_regex = re.sub(r"[/\\^$*+?.()|[\]{}]", "", accessible_name).strip().lower() or "submit"
        escaped_xpath = accessible_name.replace("'", "")

        pw_locator = f"page.getByRole('{role}', {{ name: /{safe_regex}/i }})"
        pw_action = (
            f"const target = {pw_locator};\n"
            "await target.waitFor({ state: 'visible', timeout: 9000 });\n"
            + ("await target.fill('...');" if role == "textbox" else "await target.click();")
        )
        cy_locator = f"cy.findByRole('{role}', {{ name: /{safe_regex}/i }})"
        cy_action = f"{cy_locator}.should('be.visible')." + ("type('...');" if role == "textbox" else "click();")
        sel_xpath = (
            f"//input[contains(@aria-label, '{escaped_xpath}') or contains(@placeholder, '{escaped_xpath}')]"
            if html_tag == "input"
            else f"//{html_tag}[contains(., '{escaped_xpath}') or contains(@aria-label, '{escaped_xpath}')]"
        )
        sel_locator = f'By.XPATH, "{sel_xpath}"'
        sel_snippet = (
            f'element = WebDriverWait(driver, 9).until(EC.element_to_be_clickable((By.XPATH, "{sel_xpath}")))\n'
            + ("element.send_keys('...')" if role == "textbox" else "element.click()")
        )

        suggestion = HealingSuggestion(
            failure_id=ctx.failure_id or "HEAL-001",
            failed_selector=ctx.failed_selector,
            primary_healed_locator=pw_locator,
            locator_strategy="aria_role",
            confidence=0.94 if ctx.dom_excerpt else 0.86,
            rationale=(
                f"Replaced brittle selector '{ctx.failed_selector}' with accessible ARIA locator "
                f"and strict explicit visibility wait (never silently skips blocked targets)."
            ),
            playwright=FrameworkPatch(
                framework="playwright",
                healed_selector=pw_locator,
                code_snippet=pw_action,
                explanation="Uses Playwright ARIA role locator with explicit 9s visibility wait (fails loudly if blocked).",
            ),
            cypress=FrameworkPatch(
                framework="cypress",
                healed_selector=cy_locator,
                code_snippet=cy_action,
                explanation="Uses Cypress Testing Library ARIA role query with visibility assertion.",
            ),
            selenium=FrameworkPatch(
                framework="selenium",
                healed_selector=sel_locator,
                code_snippet=sel_snippet,
                explanation="Uses WebDriverWait clickable wait with accessible XPath matcher.",
            ),
        )
        return self._enforce_no_silent_skip(suggestion, ctx)

    def run_node(self, state: dict[str, Any]) -> dict[str, Any]:
        target_url = state.get("target_url", "")
        candidates: list[Any] = list(state.get("healing_candidates") or [])
        if not candidates and state.get("triage_report") and state.get("raw_failures"):
            diagnoses = state["triage_report"].get("diagnoses", []) if isinstance(state["triage_report"], dict) else []
            heal_ids = {d.get("failure_id") for d in diagnoses if isinstance(d, dict) and d.get("needs_healing")}
            for raw in state["raw_failures"]:
                if isinstance(raw, dict) and raw.get("failed_selector") and (not heal_ids or raw.get("failure_id") in heal_ids):
                    candidates.append(raw)
        report = self.heal_batch(candidates, target_url=target_url)
        return {"healing_report": report.model_dump(), "current_agent": "healing"}


def healing_node(state: dict[str, Any]) -> dict[str, Any]:
    return HealingAgent().run_node(state)


def build_swarm_graph(
    llm: Optional[BaseChatModel] = None,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    Build and compile the end-to-end AgentSwarm LangGraph workflow:
    planner -> persona -> navigator (single-session autonomous loop) -> triage --(needs_healing)--> healing -> END
    """
    from ..browser.trace_manager import get_stream_manager
    from .navigator_agent import NavigatorAgent
    from .persona_agent import PersonaAgent
    from .planner_agent import PlannerAgent
    from .triage_agent import TriageAgent

    planner = PlannerAgent(api_key=api_key, model_name=model_name, provider=provider, llm=llm)
    persona = PersonaAgent(api_key=api_key, model_name=model_name, provider=provider, llm=llm)
    navigator = NavigatorAgent(api_key=api_key, model_name=model_name, provider=provider, llm=llm)
    triage = TriageAgent(api_key=api_key, model_name=model_name, provider=provider, llm=llm)
    healing = HealingAgent(api_key=api_key, model_name=model_name, provider=provider, llm=llm)

    def _wrap_stage(stage_name: str, stage_idx: int, start_msg: str, fn):
        def _wrapped(state: SwarmGraphState) -> dict[str, Any]:
            stream_id = state.get("stream_id")
            stream_mgr = get_stream_manager() if stream_id else None
            if stream_mgr and stream_id:
                stream_mgr.update_stage(stream_id, stage_name, start_msg, stage_index=stage_idx)
                stream_mgr.push_log(stream_id, f"{stage_name.title()} Agent", start_msg, status="info")

            out = fn(state)

            if stream_mgr and stream_id and isinstance(out, dict):
                if stage_name == "planner" and out.get("test_plan"):
                    tp = out["test_plan"]
                    goals = tp.get("goals") or tp.get("scenarios") or []
                    gen_by = tp.get("generated_by", "agent")
                    titles = ", ".join(g.get("title", "Goal") for g in goals[:4])
                    stream_mgr.update_partial(stream_id, "test_plan", tp)
                    if str(gen_by).startswith("deterministic"):
                        reason = out.get("planner_error") or "Deterministic fallback."
                        stream_mgr.push_log(stream_id, "Planner Agent", f"Goal planning fallback ({reason}) -> {len(goals)} goal(s): {titles}", status="warning")
                    else:
                        stream_mgr.push_log(stream_id, "Planner Agent", f"Autonomously set {len(goals)} goal(s) via {gen_by}: {titles}", status="success")
                elif stage_name == "persona" and out.get("personas"):
                    pr = out["personas"]
                    p_list = pr.get("personas") or []
                    stream_mgr.update_partial(stream_id, "personas", pr)
                    stream_mgr.push_log(stream_id, "Persona Agent", f"Synthesized {len(p_list)} persona(s) via {pr.get('generated_by', 'agent')}.", status="success")
                elif stage_name == "navigator" and out.get("navigation_results") is not None:
                    nav = out["navigation_results"]
                    stream_mgr.update_partial(stream_id, "navigation_results", nav)
                    passed = sum(1 for nr in nav if nr.get("overall_status") == "passed")
                    stuck = sum(1 for nr in nav if nr.get("overall_status") == "stuck")
                    failed = len(nav) - passed - stuck
                    stream_mgr.push_log(
                        stream_id,
                        "Navigator Agent",
                        f"Finished {len(nav)} goal(s): {passed} passed, {failed} failed, {stuck} stuck.",
                        status="success" if passed == len(nav) else "warning",
                    )
                elif stage_name == "triage" and out.get("triage_report"):
                    tr = out["triage_report"]
                    stream_mgr.update_partial(stream_id, "triage_report", tr)
                    diags = tr.get("diagnoses") or []
                    stream_mgr.push_log(stream_id, "Triage Agent", f"Triaged {len(diags)} failure(s) ({tr.get('healing_candidates', 0)} healing candidate(s)).", status="info" if diags else "success")
                elif stage_name == "healing" and out.get("healing_report"):
                    hr = out["healing_report"]
                    stream_mgr.update_partial(stream_id, "healing_report", hr)
                    stream_mgr.push_log(stream_id, "Healing Agent", f"Synthesized {len(hr.get('suggestions') or [])} locator patch(es).", status="success")
            return out

        return _wrapped

    def _route_after_triage(state: SwarmGraphState) -> str:
        return "healing" if state.get("healing_candidates") else END

    builder = StateGraph(SwarmGraphState)
    builder.add_node("planner", _wrap_stage("planner", 0, "Planner Agent generating directional QA goals...", planner.run_node))
    builder.add_node("persona", _wrap_stage("persona", 1, "Persona Agent synthesizing QA test personas...", persona.run_node))
    builder.add_node("navigator", _wrap_stage("navigator", 2, "Navigator Agent running step-by-step autonomous loop...", navigator.run_node))
    builder.add_node("triage", _wrap_stage("triage", 3, "Triage Agent analyzing root causes...", triage.run_node))
    builder.add_node("healing", _wrap_stage("healing", 4, "Healing Agent synthesizing resilient ARIA locators...", healing.run_node))

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "persona")
    builder.add_edge("persona", "navigator")
    builder.add_edge("navigator", "triage")
    builder.add_conditional_edges("triage", _route_after_triage, {"healing": "healing", END: END})
    builder.add_edge("healing", END)
    return builder.compile()
