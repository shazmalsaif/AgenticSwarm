from __future__ import annotations

import logging
import os
import re
from html.parser import HTMLParser
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
    generated_by: str = Field("gemini", description="Origin engine ('gemini' or 'deterministic_fallback')")


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
    Self-healing locator synthesis agent.

    Replaces brittle CSS/XPath selectors that broke during DOM refactors with resilient
    semantic ARIA role locators across Playwright, Cypress, and Selenium.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None) -> None:
        self.api_key = (api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
        self.model_name = (model_name or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()
        self._client: Optional[genai.Client] = genai.Client(api_key=self.api_key) if self.api_key else None

    @staticmethod
    def _coerce_context(context: BrokenLocatorContext | dict[str, Any], index: int = 1) -> BrokenLocatorContext:
        if isinstance(context, BrokenLocatorContext):
            ctx = context.model_copy()
        else:
            ctx = BrokenLocatorContext.model_validate(context)
        if not ctx.failure_id:
            ctx.failure_id = f"HEAL-{index:03d}"
        return ctx

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    def _invoke_gemini(self, ctx: BrokenLocatorContext) -> HealingSuggestion:
        if not self._client:
            raise RuntimeError("Gemini client is not configured.")

        prompt = (
            f"Failure ID: {ctx.failure_id}\n"
            f"Target URL: {ctx.target_url}\n"
            f"Broken Selector: {ctx.failed_selector}\n"
            f"Action Description: {ctx.action_description or 'Interact with UI element'}\n"
            f"Element Role Hint: {ctx.element_role}\n"
            f"Accessible Name Hint: {ctx.accessible_name}\n"
            f"Element Text Hint: {ctx.element_text}\n"
            f"Error Message: {ctx.error_message}\n"
            f"DOM Excerpt:\n{(ctx.dom_excerpt or 'N/A')[:2500]}\n"
        )
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are the Self-Healing Locator Agent for AgentSwarm. Synthesize resilient, "
                    "semantic ARIA-based locators (preferring getByRole / findByRole) that survive CSS and ID "
                    "refactoring. Output working patches for Playwright, Cypress, and Selenium."
                ),
                response_mime_type="application/json",
                response_schema=HealingSuggestion,
                temperature=0.1,
            ),
        )
        if response.parsed is not None and isinstance(response.parsed, HealingSuggestion):
            suggestion = response.parsed
            suggestion.failure_id = ctx.failure_id or suggestion.failure_id
            suggestion.failed_selector = ctx.failed_selector
            return suggestion
        suggestion = HealingSuggestion.model_validate_json(response.text)
        suggestion.failure_id = ctx.failure_id or suggestion.failure_id
        suggestion.failed_selector = ctx.failed_selector
        return suggestion

    def heal_locator(self, context: BrokenLocatorContext | dict[str, Any]) -> HealingSuggestion:
        """Generate semantic replacement patches for a single broken selector."""
        ctx = self._coerce_context(context)
        if self._client:
            try:
                return self._invoke_gemini(ctx)
            except Exception as exc:
                logger.warning("[HealingAgent] Gemini synthesis failed, using DOM/ARIA fallback: %s", exc)
        return self._synthesize_deterministically(ctx)

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
            # Derive clean fallback label from action description or selector tokens
            if ctx.action_description:
                cleaned = re.sub(r"^(click|press|fill|submit|trigger|open)\s+(the\s+)?", "", ctx.action_description.strip(), flags=re.I)
                name = cleaned[:40].strip() or "Submit"
            else:
                tokens = [
                    tok for tok in re.findall(r"[a-z]+", ctx.failed_selector.lower())
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
            sel_xpath = f"//input[contains(@aria-label, '{escaped_xpath_text}') or contains(@placeholder, '{escaped_xpath_text}')]"
        else:
            sel_xpath = f"//{html_tag}[contains(., '{escaped_xpath_text}') or contains(@aria-label, '{escaped_xpath_text}')]"
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

    def heal_batch(
        self,
        items: list[BrokenLocatorContext | dict[str, Any]],
        target_url: str = "",
    ) -> HealingReport:
        """Synthesize healing patches for a batch of broken locator contexts."""
        suggestions: list[HealingSuggestion] = []
        used_gemini = False

        for idx, raw in enumerate(items, start=1):
            ctx = self._coerce_context(raw, index=idx)
            if target_url and not ctx.target_url:
                ctx.target_url = target_url
            if self._client:
                try:
                    suggestion = self._invoke_gemini(ctx)
                    used_gemini = True
                except Exception as exc:
                    logger.warning("[HealingAgent] Batch item %s Gemini failed, falling back: %s", ctx.failure_id, exc)
                    suggestion = self._synthesize_deterministically(ctx)
            else:
                suggestion = self._synthesize_deterministically(ctx)
            suggestions.append(suggestion)

        return HealingReport(
            target_url=target_url,
            total_healed=len(suggestions),
            suggestions=suggestions,
            generated_by="gemini" if used_gemini else "deterministic_fallback",
        )

    def run_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph-compatible node entrypoint for locator self-healing."""
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
