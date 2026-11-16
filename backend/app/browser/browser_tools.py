from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any, Optional

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..agent.navigator_agent import BrowserAction

logger = logging.getLogger(__name__)


class InteractiveElement(BaseModel):
    """Structured metadata and ARIA locator recommendation for an interactive DOM element."""

    tag: str
    role: str
    accessible_name: str = ""
    text: str = ""
    element_id: str = ""
    classes: str = ""
    placeholder: str = ""
    name_attr: str = ""
    recommended_locator: str = ""


class _DomInteractiveParser(HTMLParser):
    """HTMLParser collecting interactive elements and heading/landmark text."""

    INTERACTIVE_TAGS = {"button", "a", "input", "select", "textarea"}
    VOID_TAGS = {"input", "br", "hr", "img", "meta", "link"}
    LANDMARK_TAGS = {"nav", "main", "header", "footer", "form", "h1", "h2", "h3"}

    def __init__(self) -> None:
        super().__init__()
        self.elements: list[dict[str, str]] = []
        self.landmarks: list[str] = []
        self._stack: list[dict[str, Any]] = []
        self._landmark_stack: list[tuple[str, list[str]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        tag_lower = tag.lower()
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        role = attr_map.get("role", "").lower()

        if tag_lower in self.LANDMARK_TAGS:
            if tag_lower == "form" and (attr_map.get("id") or attr_map.get("aria-label")):
                label = attr_map.get("aria-label") or attr_map.get("id") or "form"
                self.landmarks.append(f"form[{label}]")
            elif tag_lower in ("h1", "h2", "h3"):
                self._landmark_stack.append((tag_lower, []))

        is_interactive = tag_lower in self.INTERACTIVE_TAGS or role in (
            "button",
            "link",
            "textbox",
            "menuitem",
            "checkbox",
            "radio",
            "combobox",
            "switch",
            "tab",
        )
        if not is_interactive:
            return

        record: dict[str, Any] = {
            "tag": tag_lower,
            "role": role,
            "aria_label": attr_map.get("aria-label", ""),
            "placeholder": attr_map.get("placeholder", ""),
            "name_attr": attr_map.get("name", ""),
            "type": attr_map.get("type", "").lower(),
            "value": attr_map.get("value", ""),
            "title": attr_map.get("title", ""),
            "element_id": attr_map.get("id", ""),
            "classes": attr_map.get("class", ""),
            "text_parts": [],
        }
        if tag_lower in self.VOID_TAGS:
            record["text"] = ""
            self.elements.append(record)
        else:
            self._stack.append(record)

    def handle_data(self, data: str) -> None:
        if self._stack:
            self._stack[-1]["text_parts"].append(data)
        if self._landmark_stack:
            self._landmark_stack[-1][1].append(data)

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if self._landmark_stack and self._landmark_stack[-1][0] == tag_lower:
            ltag, parts = self._landmark_stack.pop()
            text = " ".join(" ".join(parts).split())
            if text:
                self.landmarks.append(f"{ltag}: {text[:80]}")

        for idx in range(len(self._stack) - 1, -1, -1):
            if self._stack[idx]["tag"] == tag_lower:
                rec = self._stack.pop(idx)
                rec["text"] = " ".join(" ".join(rec.pop("text_parts", [])).split())
                self.elements.append(rec)
                break


class BrowserToolkit:
    """DOM analysis, resilient selector resolution, and Playwright execution helpers."""

    @staticmethod
    def _infer_role(tag: str, explicit_role: str, input_type: str) -> str:
        if explicit_role:
            return explicit_role
        if tag == "a":
            return "link"
        if tag == "select":
            return "combobox"
        if tag == "textarea":
            return "textbox"
        if tag == "input":
            if input_type in ("submit", "button", "reset"):
                return "button"
            if input_type == "checkbox":
                return "checkbox"
            if input_type == "radio":
                return "radio"
            return "textbox"
        return "button"

    @staticmethod
    def _build_recommended_locator(role: str, accessible_name: str, element_id: str, tag: str) -> str:
        if accessible_name:
            safe = re.sub(r"[/\\^$*+?.()|[\]{}]", "", accessible_name).strip().lower()
            if safe:
                return f"page.getByRole('{role}', {{ name: /{safe}/i }})"
        if element_id:
            return f"#{element_id}"
        return tag

    @classmethod
    def extract_interactive_elements(cls, html_content: Optional[str]) -> list[InteractiveElement]:
        """Parse an HTML snapshot and return structured interactive elements with ARIA locators."""
        if not html_content or "<" not in html_content:
            return []

        parser = _DomInteractiveParser()
        try:
            parser.feed(html_content)
        except Exception as exc:
            logger.debug("[BrowserToolkit] HTML parse warning: %s", exc)

        results: list[InteractiveElement] = []
        for raw in parser.elements:
            role = cls._infer_role(raw["tag"], raw["role"], raw.get("type", ""))
            accessible_name = (
                raw["aria_label"]
                or raw.get("text", "")
                or raw["placeholder"]
                or raw.get("title", "")
                or (raw.get("value", "") if role == "button" else "")
                or raw["name_attr"]
            ).strip()
            rec_locator = cls._build_recommended_locator(
                role=role,
                accessible_name=accessible_name,
                element_id=raw["element_id"],
                tag=raw["tag"],
            )
            results.append(
                InteractiveElement(
                    tag=raw["tag"],
                    role=role,
                    accessible_name=accessible_name,
                    text=raw.get("text", ""),
                    element_id=raw["element_id"],
                    classes=raw["classes"],
                    placeholder=raw["placeholder"],
                    name_attr=raw["name_attr"],
                    recommended_locator=rec_locator,
                )
            )
        return results

    @classmethod
    def summarize_dom_for_agents(cls, html_content: Optional[str], max_elements: int = 35) -> str:
        """Produce a concise DOM summary for PlannerAgent, NavigatorAgent, and HealingAgent."""
        if not html_content or not html_content.strip():
            return "No DOM snapshot captured."
        if "<" not in html_content:
            return html_content.strip()[:2500]

        parser = _DomInteractiveParser()
        try:
            parser.feed(html_content)
        except Exception:
            pass
        elements = cls.extract_interactive_elements(html_content)[:max_elements]

        lines: list[str] = []
        if parser.landmarks:
            lines.append("Landmarks & Headings: " + " | ".join(parser.landmarks[:10]))
        if elements:
            lines.append(f"Interactive Elements ({len(elements)} shown):")
            for el in elements:
                id_part = f" id=#{el.element_id}" if el.element_id else ""
                name_part = f" name='{el.accessible_name}'" if el.accessible_name else ""
                lines.append(f"- <{el.tag}> role={el.role}{name_part}{id_part} -> {el.recommended_locator}")
        if not lines:
            stripped = re.sub(r"<[^>]+>", " ", html_content)
            return " ".join(stripped.split())[:2000]
        return "\n".join(lines)

    @staticmethod
    def _split_top_level_selectors(selector: str) -> list[str]:
        """Split comma-separated selectors without breaking commas inside (...) or {...}."""
        parts: list[str] = []
        buf: list[str] = []
        depth = 0
        quote: Optional[str] = None
        for ch in selector:
            if quote:
                buf.append(ch)
                if ch == quote:
                    quote = None
                continue
            if ch in ("'", '"'):
                quote = ch
                buf.append(ch)
            elif ch in ("(", "[", "{"):
                depth += 1
                buf.append(ch)
            elif ch in (")", "]", "}"):
                depth = max(0, depth - 1)
                buf.append(ch)
            elif ch == "," and depth == 0:
                piece = "".join(buf).strip()
                if piece:
                    parts.append(piece)
                buf = []
            else:
                buf.append(ch)
        tail = "".join(buf).strip()
        if tail:
            parts.append(tail)
        return parts

    @classmethod
    def _single_selector_matches_dom(cls, candidate: str, dom_snapshot: str) -> bool:
        cand = candidate.strip()
        if not cand:
            return False

        # Playwright getByRole('button', { name: /checkout/i }) style
        role_match = re.search(
            r"getByRole\(\s*['\"]([^'\"]+)['\"](?:\s*,\s*\{\s*name:\s*(?:/([^/]+)/([im]*)|['\"]([^'\"]+)['\"])\s*\})?\s*\)",
            cand,
        )
        if role_match:
            target_role = role_match.group(1).strip().lower()
            regex_pat = role_match.group(2)
            exact_name = role_match.group(4)
            elements = cls.extract_interactive_elements(dom_snapshot)
            for el in elements:
                if el.role != target_role:
                    continue
                haystack = f"{el.accessible_name} {el.text}".strip()
                if regex_pat:
                    try:
                        if re.search(regex_pat, haystack, flags=re.I):
                            return True
                    except re.error:
                        if regex_pat.lower() in haystack.lower():
                            return True
                elif exact_name:
                    if exact_name.strip().lower() in haystack.lower():
                        return True
                else:
                    return True
            return False

        # Playwright text= / getByText(...)
        text_match = re.match(r"^(?:text=|getByText\(['\"]([^'\"]+)['\"]\))", cand, flags=re.I)
        if text_match or cand.lower().startswith("text="):
            needle = cand.split("=", 1)[1].strip(" '\"") if cand.lower().startswith("text=") else (text_match.group(1) if text_match else "")
            if not needle:
                return False
            plain = " ".join(re.sub(r"<[^>]+>", " ", dom_snapshot).split()).lower()
            return needle.lower() in plain

        # CSS tokens inside candidate (#id, .class, [attr='val'], [attr*='val'], tag)
        id_matches = re.findall(r"#([A-Za-z0-9_-]+)", cand)
        cls_matches = re.findall(r"\.([A-Za-z0-9_-]+)", cand)
        attr_matches = re.findall(r"\[([A-Za-z0-9_-]+)\s*([*^$]?=)\s*['\"]([^'\"]+)['\"]\]", cand)

        for el_id in id_matches:
            if not re.search(rf'id=["\'][^"\']*\b{re.escape(el_id)}\b', dom_snapshot):
                return False

        for cls_name in cls_matches:
            if not re.search(rf'class=["\'][^"\']*\b{re.escape(cls_name)}\b', dom_snapshot):
                return False

        for attr_name, op, attr_val in attr_matches:
            esc_name = re.escape(attr_name)
            esc_val = re.escape(attr_val)
            if op == "*=":
                pattern = rf'{esc_name}=["\'][^"\']*{esc_val}[^"\']*["\']'
            elif op == "^=":
                pattern = rf'{esc_name}=["\']{esc_val}[^"\']*["\']'
            elif op == "$=":
                pattern = rf'{esc_name}=["\'][^"\']*{esc_val}["\']'
            else:
                pattern = rf'{esc_name}=["\']{esc_val}["\']'
            if not re.search(pattern, dom_snapshot, flags=re.I):
                return False

        if id_matches or cls_matches or attr_matches:
            tag_prefix = re.match(r"^([a-zA-Z][a-zA-Z0-9]*)", cand)
            if tag_prefix:
                tag_name = tag_prefix.group(1).lower()
                if f"<{tag_name}" not in dom_snapshot.lower():
                    return False
            return True

        # Bare tag name (e.g., "main", "body", "button", "input")
        tag_only = re.match(r"^[a-zA-Z][a-zA-Z0-9-]*$", cand)
        if tag_only:
            return f"<{cand.lower()}" in dom_snapshot.lower()

        return False

    @classmethod
    def selector_exists_in_dom(cls, selector: Optional[str], dom_snapshot: Optional[str]) -> bool:
        """Return True if a CSS, ARIA, or text selector resolves in an HTML snapshot."""
        if not selector or not dom_snapshot or "<" not in dom_snapshot:
            return False
        candidates = cls._split_top_level_selectors(selector)
        return any(cls._single_selector_matches_dom(part, dom_snapshot) for part in candidates)

    @classmethod
    def resolve_working_selector(
        cls,
        primary_selector: Optional[str],
        fallback_selectors: list[str],
        dom_snapshot: Optional[str] = None,
        page: Optional[Any] = None,
    ) -> tuple[Optional[str], bool]:
        """
        Resolve the first working selector from [primary_selector, *fallback_selectors].

        Returns `(selected_selector, used_fallback)`.
        """
        ordered: list[str] = []
        if primary_selector:
            ordered.append(primary_selector)
        for fb in fallback_selectors or []:
            if fb and fb not in ordered:
                ordered.append(fb)

        if not ordered:
            return None, False

        if page is not None and hasattr(page, "locator"):
            for idx, cand in enumerate(ordered):
                try:
                    locator = cls._to_playwright_locator(page, cand)
                    if hasattr(locator, "count") and locator.count() > 0:
                        return cand, idx > 0
                except Exception:
                    continue
            return primary_selector or ordered[0], False

        if dom_snapshot and "<" in dom_snapshot:
            for idx, cand in enumerate(ordered):
                if cls.selector_exists_in_dom(cand, dom_snapshot):
                    return cand, idx > 0
            return primary_selector or ordered[0], False

        return primary_selector or ordered[0], False

    @staticmethod
    def _to_playwright_locator(page: Any, selector: str) -> Any:
        role_match = re.search(
            r"getByRole\(\s*['\"]([^'\"]+)['\"](?:\s*,\s*\{\s*name:\s*(?:/([^/]+)/i?|['\"]([^'\"]+)['\"])\s*\})?\s*\)",
            selector,
        )
        if role_match and hasattr(page, "get_by_role"):
            role = role_match.group(1)
            regex_body = role_match.group(2)
            exact_name = role_match.group(3)
            if regex_body:
                return page.get_by_role(role, name=re.compile(regex_body, re.I))
            if exact_name:
                return page.get_by_role(role, name=exact_name)
            return page.get_by_role(role)
        return page.locator(selector)

    @classmethod
    def _run_selector_action(cls, page: Any, action_type: str, selector: str, value: Optional[str], timeout_ms: int) -> None:
        if action_type == "wait_for_selector" and "getByRole(" not in selector:
            page.wait_for_selector(selector, timeout=timeout_ms)
            return

        locator = cls._to_playwright_locator(page, selector)
        target = locator.first if hasattr(locator, "first") else locator

        if action_type == "click":
            target.click(timeout=timeout_ms)
        elif action_type == "fill":
            target.fill(value or "", timeout=timeout_ms)
        elif action_type in ("assert_visible", "wait_for_selector"):
            target.wait_for(state="visible", timeout=timeout_ms)

    @classmethod
    def execute_playwright_action(
        cls,
        page: Any,
        action: BrowserAction | dict[str, Any],
        target_url: str,
    ) -> tuple[Optional[int], Optional[str]]:
        """
        Execute a single BrowserAction against a Playwright Page with automatic fallback locators.

        Returns `(http_status, used_selector)`.
        """
        if isinstance(action, dict):
            action_type = str(action.get("action_type", "assert_visible"))
            primary = action.get("selector")
            fallbacks = list(action.get("fallback_selectors") or [])
            value = action.get("value")
            timeout_ms = int(action.get("timeout_ms") or 5000)
        else:
            action_type = action.action_type
            primary = action.selector
            fallbacks = list(action.fallback_selectors or [])
            value = action.value
            timeout_ms = int(action.timeout_ms)

        if action_type == "goto":
            resp = page.goto(value or target_url, timeout=timeout_ms)
            http_status = getattr(resp, "status", None) if resp else None
            return (int(http_status) if http_status is not None else None), None

        if action_type == "assert_url":
            expected = value or target_url
            current = str(getattr(page, "url", ""))
            if expected not in current:
                raise AssertionError(f"Expected URL to contain '{expected}', got '{current}'")
            return None, None

        default_sel = "input" if action_type == "fill" else ("button" if action_type == "click" else "body")
        candidates: list[str] = []
        if primary:
            candidates.append(primary)
        for fb in fallbacks:
            if fb and fb not in candidates:
                candidates.append(fb)
        if not candidates:
            candidates.append(default_sel)

        last_exc: Optional[Exception] = None
        for idx, cand in enumerate(candidates):
            cand_timeout = timeout_ms if idx == 0 else min(timeout_ms, 2000)
            try:
                cls._run_selector_action(page, action_type, cand, value, cand_timeout)
                return None, cand
            except Exception as exc:
                last_exc = exc

        if last_exc is not None:
            raise last_exc
        return None, primary


@tool
def inspect_dom_elements_tool(html_snapshot: str) -> str:
    """Inspect an HTML DOM snapshot and list interactive UI elements with recommended ARIA locators."""
    return BrowserToolkit.summarize_dom_for_agents(html_snapshot)


@tool
def check_selector_in_dom_tool(selector: str, html_snapshot: str) -> str:
    """Check whether a CSS, ARIA, or text selector resolves inside the provided HTML DOM snapshot."""
    found = BrowserToolkit.selector_exists_in_dom(selector, html_snapshot)
    return f"Selector '{selector}' matched DOM: {found}"


@tool
def suggest_resilient_locator_tool(broken_selector: str, action_description: str, html_snapshot: str) -> str:
    """Suggest a resilient Playwright ARIA locator replacement for a broken CSS/XPath selector."""
    from ..agent.healing_agent import BrokenLocatorContext, HealingAgent

    suggestion = HealingAgent().heal_locator(
        BrokenLocatorContext(
            failed_selector=broken_selector,
            action_description=action_description,
            dom_excerpt=html_snapshot[:2500] if html_snapshot else None,
        )
    )
    return (
        f"Primary Locator: {suggestion.primary_healed_locator}\n"
        f"Confidence: {suggestion.confidence}\n"
        f"Rationale: {suggestion.rationale}"
    )


def get_browser_tools() -> list[BaseTool]:
    """Return LangChain tools wrapping AgentSwarm DOM inspection and selector diagnostics."""
    return [
        inspect_dom_elements_tool,
        check_selector_in_dom_tool,
        suggest_resilient_locator_tool,
    ]
