from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from typing import Any, Literal, Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

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
    """LangGraph state contract for HealingAgent locator patch synthesis."""

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
    page: Optional[Any]
    use_live_browser: bool
    record_traces: bool
    run_id: Optional[str]
    stream_id: Optional[str]
    test_plan: dict[str, Any]
    personas: dict[str, Any]
    navigation_results: list[dict[str, Any]]
    raw_failures: list[dict[str, Any]]
    triage_report: dict[str, Any]
    healing_candidates: list[dict[str, Any]]
    needs_healing: bool
    healing_report: dict[str, Any]
    current_agent: str


class _InteractiveElementParser(HTMLParser):
    """Lightweight HTML parser extracting interactive DOM candidates for fallback healing."""

    INTERACTIVE_TAGS = {"button", "a", "input", "select", "textarea"}

    def __init__(self) -> None:
        super().__init__()
        self.elements: list[dict[str, str]] = []
        self._current: Optional[dict[str, str]] = None
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        tag_lower = tag.lower()
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        if tag_lower in self.INTERACTIVE_TAGS or attr_map.get("role") in ("button", "link", "textbox", "menuitem"):
            self._current = {
                "tag": tag_lower,
                "role": attr_map.get("role", ""),
                "aria_label": attr_map.get("aria-label", ""),
                "placeholder": attr_map.get("placeholder", ""),
                "name": attr_map.get("name", ""),
                "type": attr_map.get("type", ""),
                "id": attr_map.get("id", ""),
                "text": "",
            }
            self._text_parts = []
            if tag_lower == "input":
                self.elements.append(self._current)
                self._current = None

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._current is not None and tag.lower() == self._current["tag"]:
            self._current["text"] = " ".join(" ".join(self._text_parts).split())
            self.elements.append(self._current)
            self._current = None
            self._text_parts = []


class HealingAgent:
    """
    Model-agnostic self-healing locator synthesis agent powered by LangChain & LangGraph.

    Uses `ChatPromptTemplate | llm.with_structured_output(HealingSuggestion)` inside a
    compiled LangGraph `StateGraph` (`parse_dom_context -> synthesize_patches`).
    """

    SYSTEM_PROMPT = (
        "You are the Self-Healing Locator Agent for AgentSwarm. Synthesize resilient, "
        "semantic ARIA-based locators (preferring getByRole / findByRole) that survive CSS and ID "
        "refactoring. Output working patches for Playwright, Cypress, and Selenium."
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
        if isinstance(context, BrokenLocatorContext):
            ctx = context.model_copy()
        else:
            ctx = BrokenLocatorContext.model_validate(context)
        if not ctx.failure_id:
            ctx.failure_id = f"HEAL-{index:03d}"
        return ctx

    def _invoke_chain(self, ctx: BrokenLocatorContext) -> HealingSuggestion:
        prompt_input = {
            "failure_id": ctx.failure_id or "HEAL-001",
            "target_url": ctx.target_url,
            "failed_selector": ctx.failed_selector,
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
        return result

    def _node_parse_dom_context(self, state: HealingGraphState) -> dict[str, Any]:
        target_url = state.get("target_url", "")
        raw_items = list(state.get("items") or [])
        coerced: list[BrokenLocatorContext] = []

        for idx, raw in enumerate(raw_items, start=1):
            ctx = self._coerce_context(raw, index=idx)
            if target_url and not ctx.target_url:
                ctx.target_url = target_url
            if ctx.dom_excerpt and (not ctx.element_role or not ctx.accessible_name):
                role, name, _ = self._infer_target_from_dom(ctx)
                if not ctx.element_role:
                    ctx.element_role = role
                if not ctx.accessible_name:
                    ctx.accessible_name = name
            coerced.append(ctx)

        return {"coerced_items": coerced}

    def _node_synthesize_patches(self, state: HealingGraphState) -> dict[str, Any]:
        target_url = state.get("target_url", "")
        coerced = state.get("coerced_items") or []
        suggestions: list[HealingSuggestion] = []
        used_llm = False

        for ctx in coerced:
            if self._chain:
                try:
                    suggestion = self._invoke_chain(ctx)
                    used_llm = True
                except Exception as exc:
                    logger.warning("[HealingAgent] Item %s LangChain synthesis failed, using fallback: %s", ctx.failure_id, exc)
                    suggestion = self._synthesize_deterministically(ctx)
            else:
                suggestion = self._synthesize_deterministically(ctx)
            suggestions.append(suggestion)

        report = HealingReport(
            target_url=target_url,
            total_healed=len(suggestions),
            suggestions=suggestions,
            generated_by="langchain" if used_llm else "deterministic_fallback",
        )
        return {"healing_report": report}

    def build_healing_graph(self):
        """Build and compile the internal LangGraph workflow for selector healing."""
        builder = StateGraph(HealingGraphState)
        builder.add_node("parse_dom_context", self._node_parse_dom_context)
        builder.add_node("synthesize_patches", self._node_synthesize_patches)

        builder.add_edge(START, "parse_dom_context")
        builder.add_edge("parse_dom_context", "synthesize_patches")
        builder.add_edge("synthesize_patches", END)
        return builder.compile()

    def heal_locator(self, context: BrokenLocatorContext | dict[str, Any]) -> HealingSuggestion:
        """Synthesize semantic replacement patches for a single broken selector via LangGraph."""
        result_state = self._graph.invoke({"items": [context], "target_url": ""})
        return result_state["healing_report"].suggestions[0]

    def heal_batch(
        self,
        items: list[BrokenLocatorContext | dict[str, Any]],
        target_url: str = "",
    ) -> HealingReport:
        """Execute the compiled LangGraph healing workflow across a batch of broken locators."""
        result_state = self._graph.invoke({"items": items, "target_url": target_url})
        return result_state["healing_report"]

    @staticmethod
    def _infer_target_from_dom(ctx: BrokenLocatorContext) -> tuple[str, str, str]:
        """Extract (aria_role, accessible_name, html_tag) from DOM excerpt or context hints."""
        role = (ctx.element_role or "").strip().lower()
        name = (ctx.accessible_name or ctx.element_text or "").strip()
        html_tag = "button"

        if ctx.dom_excerpt and "<" in ctx.dom_excerpt:
            parser = _InteractiveElementParser()
            try:
                parser.feed(ctx.dom_excerpt)
            except Exception:
                pass

            if parser.elements:
                hint_tokens = set(
                    re.findall(
                        r"[a-z0-9]+",
                        f"{ctx.failed_selector} {ctx.action_description or ''} {ctx.element_text or ''}".lower(),
                    )
                ) - {"btn", "button", "cta", "v1", "v2", "input", "legacy", "the", "and", "click"}

                best_el = parser.elements[0]
                best_score = -1
                for el in parser.elements:
                    haystack = " ".join(
                        [el["aria_label"], el["text"], el["placeholder"], el["name"], el["id"], el["tag"]]
                    ).lower()
                    score = sum(1 for tok in hint_tokens if tok in haystack)
                    if el["aria_label"] or el["text"]:
                        score += 1
                    if score > best_score:
                        best_score = score
                        best_el = el

                tag = best_el["tag"]
                html_tag = tag if tag in ("button", "a", "input", "select", "textarea") else "button"
                if not role:
                    if best_el["role"]:
                        role = best_el["role"]
                    elif tag == "a":
                        role = "link"
                    elif tag in ("input", "textarea"):
                        role = "textbox"
                    elif tag == "select":
                        role = "combobox"
                    else:
                        role = "button"
                if not name:
                    name = (
                        best_el["aria_label"]
                        or best_el["text"]
                        or best_el["placeholder"]
                        or best_el["name"]
                    ).strip()

        if not role:
            desc_lower = f"{ctx.failed_selector} {ctx.action_description or ''}".lower()
            if any(tok in desc_lower for tok in ("email", "password", "input", "fill", "search", "textbox")):
                role = "textbox"
                html_tag = "input"
            elif any(tok in desc_lower for tok in ("link", "href", "navigate")):
                role = "link"
                html_tag = "a"
            else:
                role = "button"
                html_tag = "button"
        elif role == "link":
            html_tag = "a"
        elif role in ("textbox", "combobox"):
            html_tag = "input"

        if not name:
            if ctx.action_description:
                cleaned = re.sub(
                    r"^(click|press|fill|submit|trigger|open)\s+(the\s+)?",
                    "",
                    ctx.action_description.strip(),
                    flags=re.I,
                )
                name = cleaned[:40].strip() or "Submit"
            else:
                tokens = [
                    tok
                    for tok in re.findall(r"[a-z]+", ctx.failed_selector.lower())
                    if tok not in ("btn", "button", "cta", "v1", "v2", "legacy", "el", "id")
                ]
                name = " ".join(tokens).title() or "Continue"

        return role, name, html_tag

    @staticmethod
    def _escape_regex_text(label: str) -> str:
        safe = re.sub(r"[/\\^$*+?.()|[\]{}]", "", label).strip()
        return safe.lower() or "submit"

    def _synthesize_deterministically(self, ctx: BrokenLocatorContext) -> HealingSuggestion:
        role, accessible_name, html_tag = self._infer_target_from_dom(ctx)
        regex_pattern = self._escape_regex_text(accessible_name)
        escaped_xpath_text = accessible_name.replace("'", "")

        pw_locator = f"page.getByRole('{role}', {{ name: /{regex_pattern}/i }})"
        pw_action = f"await {pw_locator}.fill('...');" if role == "textbox" else f"await {pw_locator}.click();"

        cy_locator = f"cy.findByRole('{role}', {{ name: /{regex_pattern}/i }})"
        cy_action = f"{cy_locator}.type('...');" if role == "textbox" else f"{cy_locator}.click();"

        if html_tag == "input":
            sel_xpath = (
                f"//input[contains(@aria-label, '{escaped_xpath_text}') "
                f"or contains(@placeholder, '{escaped_xpath_text}')]"
            )
        else:
            sel_xpath = (
                f"//{html_tag}[contains(., '{escaped_xpath_text}') "
                f"or contains(@aria-label, '{escaped_xpath_text}')]"
            )
        sel_locator = f'By.XPATH, "{sel_xpath}"'
        sel_snippet = (
            f'element = driver.find_element(By.XPATH, "{sel_xpath}")\n'
            + ("element.send_keys('...')" if role == "textbox" else "element.click()")
        )

        return HealingSuggestion(
            failure_id=ctx.failure_id or "HEAL-001",
            failed_selector=ctx.failed_selector,
            primary_healed_locator=pw_locator,
            locator_strategy="aria_role",
            confidence=0.94 if ctx.dom_excerpt else 0.86,
            rationale=(
                f"Replaced brittle selector '{ctx.failed_selector}' with semantic ARIA role '{role}' "
                f"matching accessible name '{accessible_name}', resilient to CSS class and ID changes."
            ),
            playwright=FrameworkPatch(
                framework="playwright",
                healed_selector=pw_locator,
                code_snippet=f"const target = {pw_locator};\n{pw_action}",
                explanation="Uses Playwright's built-in ARIA accessibility role locator.",
            ),
            cypress=FrameworkPatch(
                framework="cypress",
                healed_selector=cy_locator,
                code_snippet=cy_action,
                explanation="Uses Cypress Testing Library ARIA role query.",
            ),
            selenium=FrameworkPatch(
                framework="selenium",
                healed_selector=sel_locator,
                code_snippet=sel_snippet,
                explanation="Uses accessible text/aria-label XPath matcher compatible with Selenium 4.",
            ),
        )

    def run_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph parent-graph node entrypoint for locator self-healing."""
        target_url = state.get("target_url", "")
        candidates: list[Any] = list(state.get("healing_candidates") or [])

        if not candidates and state.get("triage_report") and state.get("raw_failures"):
            triage_report = state["triage_report"]
            diagnoses = triage_report.get("diagnoses", []) if isinstance(triage_report, dict) else []
            heal_ids = {d.get("failure_id") for d in diagnoses if isinstance(d, dict) and d.get("needs_healing")}
            for raw in state["raw_failures"]:
                if isinstance(raw, dict) and raw.get("failed_selector") and (
                    not heal_ids or raw.get("failure_id") in heal_ids
                ):
                    candidates.append(raw)

        report = self.heal_batch(candidates, target_url=target_url)
        return {
            "healing_report": report.model_dump(),
            "current_agent": "healing",
        }


def healing_node(state: dict[str, Any]) -> dict[str, Any]:
    """Functional LangGraph node wrapper around HealingAgent."""
    return HealingAgent().run_node(state)


def build_swarm_graph(
    llm: Optional[BaseChatModel] = None,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    Build and compile the end-to-end AgentSwarm LangGraph workflow:
    planner -> persona -> navigator -> triage --(needs_healing)--> healing -> END
                                              --(no healing)-----> END
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
                    scenarios = tp.get("scenarios") or []
                    gen_by = tp.get("generated_by", "agent")
                    titles = ", ".join(s.get("title", "Scenario") for s in scenarios[:4])
                    stream_mgr.update_partial(stream_id, "test_plan", tp)
                    stream_mgr.push_log(
                        stream_id,
                        "Planner Agent",
                        f"Planned {len(scenarios)} scenario(s) via {gen_by}: {titles}",
                        status="success",
                    )
                elif stage_name == "persona" and out.get("personas"):
                    pr = out["personas"]
                    p_list = pr.get("personas") or []
                    gen_by = pr.get("generated_by", "agent")
                    names = ", ".join(p.get("name", "Persona") for p in p_list[:4])
                    stream_mgr.update_partial(stream_id, "personas", pr)
                    stream_mgr.push_log(
                        stream_id,
                        "Persona Agent",
                        f"Synthesized {len(p_list)} persona(s) via {gen_by}: {names}",
                        status="success",
                    )
                elif stage_name == "navigator" and out.get("navigation_results") is not None:
                    nav = out["navigation_results"]
                    stream_mgr.update_partial(stream_id, "navigation_results", nav)
                    passed = sum(1 for nr in nav if nr.get("overall_status") == "passed")
                    stream_mgr.push_log(
                        stream_id,
                        "Navigator Agent",
                        f"Completed {len(nav)} scenario(s) ({passed} passed, {len(nav) - passed} failed).",
                        status="success" if passed == len(nav) else "warning",
                    )
                elif stage_name == "triage" and out.get("triage_report"):
                    tr = out["triage_report"]
                    stream_mgr.update_partial(stream_id, "triage_report", tr)
                    diags = tr.get("diagnoses") or []
                    stream_mgr.push_log(
                        stream_id,
                        "Triage Agent",
                        f"Triaged {len(diags)} failure(s) ({tr.get('healing_candidates', 0)} healing candidate(s)).",
                        status="info" if diags else "success",
                    )
                elif stage_name == "healing" and out.get("healing_report"):
                    hr = out["healing_report"]
                    stream_mgr.update_partial(stream_id, "healing_report", hr)
                    sugs = hr.get("suggestions") or []
                    stream_mgr.push_log(
                        stream_id,
                        "Healing Agent",
                        f"Synthesized {len(sugs)} self-healing ARIA locator patch(es).",
                        status="success",
                    )
            return out
        return _wrapped

    def _route_after_triage(state: SwarmGraphState) -> str:
        if state.get("healing_candidates"):
            return "healing"
        return END

    builder = StateGraph(SwarmGraphState)
    builder.add_node("planner", _wrap_stage("planner", 0, "Planner Agent generating scenarios from live DOM...", planner.run_node))
    builder.add_node("persona", _wrap_stage("persona", 1, "Persona Agent synthesizing QA test personas...", persona.run_node))
    builder.add_node("navigator", _wrap_stage("navigator", 2, "Navigator Agent executing browser steps & streaming frames...", navigator.run_node))
    builder.add_node("triage", _wrap_stage("triage", 3, "Triage Agent analyzing failures and root causes...", triage.run_node))
    builder.add_node("healing", _wrap_stage("healing", 4, "Healing Agent synthesizing resilient ARIA locators...", healing.run_node))

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "persona")
    builder.add_edge("persona", "navigator")
    builder.add_edge("navigator", "triage")
    builder.add_conditional_edges("triage", _route_after_triage, {"healing": "healing", END: END})
    builder.add_edge("healing", END)
    return builder.compile()
