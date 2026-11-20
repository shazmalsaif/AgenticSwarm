from __future__ import annotations

import json
import logging
import re
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any, Optional

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel

if TYPE_CHECKING:
    from ..agent.navigator_agent import BrowserAction

logger = logging.getLogger(__name__)

_FINGERPRINT_HELPER_JS = """
const computeFp = (url, title, overlays, els, visText) => {
  const ovSig = overlays.map(o => o.selector + ':' + (o.is_intro ? '1' : '0')).join('|');
  const elSig = els.slice(0, 30).map(e => e.role + ':' + e.name + ':' + (e.disabled ? 'D' : 'E') + ':' + (e.expanded || '') + ':' + (e.selected || '')).join('|');
  return [url, title, ovSig, String(els.length), elSig, visText.slice(0, 300)].join('||');
};
"""

_INTERACTIVE_COLLECT_JS = """
const isVisible = (el) => {
  if (!el || !el.isConnected) return false;
  const cs = window.getComputedStyle(el);
  if (!cs || cs.display === 'none' || cs.visibility === 'hidden' || parseFloat(cs.opacity || '1') <= 0.05) return false;
  const rect = el.getBoundingClientRect();
  return rect.width > 1 && rect.height > 1;
};

const getAccessibleName = (el) => {
  const aria = (el.getAttribute('aria-label') || '').trim();
  if (aria) return aria;
  const labelledBy = (el.getAttribute('aria-labelledby') || '').trim();
  if (labelledBy) {
    const parts = labelledBy.split(/\\s+/).map(id => {
      const ref = document.getElementById(id);
      return ref ? (ref.innerText || ref.textContent || '').trim() : '';
    }).filter(Boolean);
    if (parts.length) return parts.join(' ');
  }
  if (el.id) {
    try {
      const lbl = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
      if (lbl && (lbl.innerText || '').trim()) return lbl.innerText.trim();
    } catch (_) {}
  }
  const parentLbl = el.closest('label');
  if (parentLbl && (parentLbl.innerText || '').trim()) return parentLbl.innerText.trim();
  const inner = (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim();
  if (inner) return inner.slice(0, 80);
  const ph = (el.getAttribute('placeholder') || '').trim();
  if (ph) return ph;
  const ttl = (el.getAttribute('title') || '').trim();
  if (ttl) return ttl;
  if ((el.tagName || '').toLowerCase() === 'input' && ['submit','button','reset'].includes((el.type || '').toLowerCase())) {
    if ((el.value || '').trim()) return el.value.trim();
  }
  return (el.getAttribute('name') || '').trim();
};

const inferRole = (el) => {
  const explicit = (el.getAttribute('role') || '').trim().toLowerCase();
  if (explicit) return explicit;
  const tag = (el.tagName || '').toLowerCase();
  if (tag === 'a') return 'link';
  if (tag === 'select') return 'combobox';
  if (tag === 'textarea') return 'textbox';
  if (tag === 'input') {
    const t = (el.type || '').toLowerCase();
    if (['submit','button','reset'].includes(t)) return 'button';
    if (t === 'checkbox') return 'checkbox';
    if (t === 'radio') return 'radio';
    return 'textbox';
  }
  return 'button';
};

const vw = window.innerWidth || 1280;
const vh = window.innerHeight || 720;

const collectOverlays = () => {
  const found = [];
  const seen = new Set();
  const addOverlay = (el, explicitSel, isIntro) => {
    if (!el || seen.has(el) || !isVisible(el)) return;
    const cs = window.getComputedStyle(el);
    if (cs.pointerEvents === 'none') return;
    seen.add(el);
    const id = el.id ? '#' + el.id : '';
    const role = (el.getAttribute('role') || '').trim().toLowerCase();
    const sel = explicitSel || id || (role ? `[role="${role}"]` : el.tagName.toLowerCase());
    found.push({
      selector: sel,
      tag: el.tagName.toLowerCase(),
      id: el.id || '',
      role,
      is_intro: Boolean(isIntro || el.id === 'intro-overlay'),
      text: (el.innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 140),
      html: (el.outerHTML || '').slice(0, 900)
    });
  };

  const intro = document.getElementById('intro-overlay');
  if (intro) addOverlay(intro, '#intro-overlay', true);

  for (const el of document.querySelectorAll('[role="dialog"], [aria-modal="true"]')) {
    if (el.id === 'agentswarm-live-highlight') continue;
    addOverlay(el, el.id ? '#' + el.id : '[role="dialog"]', false);
  }

  for (const el of document.querySelectorAll('div, section, aside, dialog')) {
    if (el.id === 'agentswarm-live-highlight' || seen.has(el) || !isVisible(el)) continue;
    const cs = window.getComputedStyle(el);
    const zi = parseInt(cs.zIndex || '0', 10) || 0;
    if (cs.position === 'fixed' && zi >= 20 && cs.pointerEvents !== 'none') {
      const r = el.getBoundingClientRect();
      if (r.width >= vw * 0.5 && r.height >= vh * 0.5) {
        addOverlay(el, el.id ? '#' + el.id : '', el.id === 'intro-overlay');
      }
    }
  }
  return found;
};

const overlays = collectOverlays();
const rawEls = Array.from(document.querySelectorAll('a, button, input, textarea, select, [role="button"], [role="link"], [role="tab"], [role="textbox"], [role="checkbox"], [role="radio"], [role="combobox"], [role="switch"], [role="menuitem"]'));
const els = [];
for (const el of rawEls) {
  if (!isVisible(el)) continue;
  const rect = el.getBoundingClientRect();
  const role = inferRole(el);
  const name = getAccessibleName(el).replace(/\\s+/g, ' ').trim();
  const placeholder = (el.getAttribute('placeholder') || '').trim();
  const testid = (el.getAttribute('data-testid') || el.getAttribute('data-test') || '').trim();
  let labelText = '';
  if (el.id) {
    try {
      const lbl = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
      if (lbl) labelText = (lbl.innerText || '').trim();
    } catch (_) {}
  }
  if (!labelText) {
    const parentLbl = el.closest('label');
    if (parentLbl) labelText = (parentLbl.innerText || '').trim();
  }
  if (!labelText) labelText = (el.getAttribute('aria-label') || '').trim();
  els.push({
    tag: el.tagName.toLowerCase(),
    role,
    name: name.slice(0, 80),
    label: labelText.slice(0, 80),
    placeholder: placeholder.slice(0, 60),
    testid: testid.slice(0, 60),
    input_type: (el.type || '').toLowerCase(),
    disabled: Boolean(el.disabled || el.getAttribute('aria-disabled') === 'true'),
    expanded: el.getAttribute('aria-expanded') || '',
    selected: el.getAttribute('aria-selected') || el.getAttribute('aria-pressed') || '',
    href: (el.tagName.toLowerCase() === 'a' ? (el.getAttribute('href') || '') : '').slice(0, 120),
    in_viewport: rect.bottom > 0 && rect.top < vh && rect.right > 0 && rect.left < vw
  });
}
const visText = ((document.body && document.body.innerText) || '').replace(/\\s+/g, ' ').trim().slice(0, 750);
"""


class InteractiveElement(BaseModel):
    """Structured metadata and accessible ARIA locator recommendation for an interactive DOM element."""

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
    """HTMLParser collecting interactive elements and heading/landmark text from HTML strings."""

    INTERACTIVE_TAGS = {"button", "a", "input", "select", "textarea"}
    VOID_TAGS = {"input", "br", "hr", "img", "meta", "link"}
    LANDMARK_TAGS = {"nav", "main", "header", "footer", "form", "h1", "h2", "h3"}

    def __init__(self) -> None:
        super().__init__()
        self.elements: list[dict[str, Any]] = []
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
            "button", "link", "textbox", "menuitem", "checkbox", "radio", "combobox", "switch", "tab",
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
    """DOM inspection, accessible Playwright locator resolution, and explicit waits."""

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
            return f"getByRole('{role}', {{ name: {json.dumps(accessible_name)} }})"
        if element_id:
            return f"#{element_id}"
        return tag

    @classmethod
    def extract_interactive_elements(cls, html_content: Optional[str]) -> list[InteractiveElement]:
        """Parse an HTML snapshot string and return structured interactive elements."""
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
                    recommended_locator=cls._build_recommended_locator(
                        role=role,
                        accessible_name=accessible_name,
                        element_id=raw["element_id"],
                        tag=raw["tag"],
                    ),
                )
            )
        return results

    @classmethod
    def summarize_dom_for_agents(cls, html_content: Optional[str], max_elements: int = 35) -> str:
        """Produce a concise DOM summary from raw HTML for PlannerAgent / HealingAgent."""
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
            lines.append(f"Visible Interactive Elements ({len(elements)} shown):")
            for el in elements:
                name_part = f" name={json.dumps(el.accessible_name)}" if el.accessible_name else ""
                lines.append(f"- role={el.role}{name_part} ({el.tag})")
        if not lines:
            stripped = re.sub(r"<[^>]+>", " ", html_content)
            return " ".join(stripped.split())[:2000]
        return "\n".join(lines)

    @classmethod
    def compute_page_fingerprint(cls, page: Any) -> str:
        """Fast in-page DOM/interactive state fingerprint used to cache snapshots when nothing changed."""
        if page is None or not hasattr(page, "evaluate"):
            return ""
        script = f"""() => {{
            {_INTERACTIVE_COLLECT_JS}
            {_FINGERPRINT_HELPER_JS}
            return computeFp(window.location.href || '', document.title || '', overlays, els, visText);
        }}"""
        try:
            fp = page.evaluate(script)
            return str(fp or "")
        except Exception:
            return ""

    @classmethod
    def extract_live_page_state(cls, page: Any, max_elements: int = 45) -> dict[str, Any]:
        """Extract compact visible interactive elements, active overlays, headings, and DOM excerpt from a live Page."""
        if page is None:
            return {
                "url": "",
                "title": "",
                "blocking_overlays": [],
                "interactive_elements": [],
                "headings": [],
                "visible_text": "",
                "dom_excerpt": "",
                "dom_hash": "",
            }
        if not hasattr(page, "evaluate"):
            cur_url = str(getattr(page, "url", "") or "")
            return {
                "url": cur_url,
                "title": "",
                "blocking_overlays": [],
                "interactive_elements": [],
                "headings": [],
                "visible_text": "",
                "dom_excerpt": "",
                "dom_hash": cur_url,
            }

        script = f"""(maxEls) => {{
            {_INTERACTIVE_COLLECT_JS}
            {_FINGERPRINT_HELPER_JS}
            const headings = [];
            for (const h of document.querySelectorAll('h1, h2, h3')) {{
                if (!isVisible(h)) continue;
                const t = (h.innerText || h.textContent || '').replace(/\\s+/g, ' ').trim();
                if (t && !headings.includes(t.slice(0, 90))) headings.push(t.slice(0, 90));
                if (headings.length >= 12) break;
            }}
            let domExcerpt = '';
            if (overlays.length > 0) {{
                domExcerpt = '[BLOCKING OVERLAY]\\n' + (overlays[0].html || '') + '\\n[BODY EXCERPT]\\n' + ((document.body && document.body.outerHTML) || '').slice(0, 1000);
            }} else {{
                domExcerpt = ((document.documentElement && document.documentElement.outerHTML) || '').slice(0, 2000);
            }}
            const cleanOverlays = overlays.map(o => ({{
                selector: o.selector,
                tag: o.tag,
                id: o.id,
                role: o.role,
                is_intro: o.is_intro,
                text: o.text
            }}));
            return {{
                url: window.location.href || '',
                title: document.title || '',
                blocking_overlays: cleanOverlays,
                interactive_elements: els.slice(0, maxEls || 45),
                headings,
                visible_text: visText,
                dom_excerpt: domExcerpt.slice(0, 2000),
                dom_hash: computeFp(window.location.href || '', document.title || '', overlays, els, visText)
            }};
        }}"""
        try:
            raw = page.evaluate(script, max_elements)
            if isinstance(raw, dict):
                return raw
        except Exception as exc:
            logger.debug("[BrowserToolkit] Live page state extraction warning: %s", exc)

        cur_url = str(getattr(page, "url", "") or "")
        return {
            "url": cur_url,
            "title": "",
            "blocking_overlays": [],
            "interactive_elements": [],
            "headings": [],
            "visible_text": "",
            "dom_excerpt": "",
            "dom_hash": cur_url,
        }

    @staticmethod
    def format_live_state_for_llm(state: dict[str, Any]) -> str:
        """Render a compact, prompt-friendly view containing ONLY visible interactive elements and text."""
        url = str(state.get("url") or "")
        title = str(state.get("title") or "")
        overlays = state.get("blocking_overlays") or []
        headings = state.get("headings") or []
        elements = state.get("interactive_elements") or []
        vis_text = str(state.get("visible_text") or "").strip()

        lines: list[str] = [f"URL: {url}", f"Title: {title or '(none)'}"]
        if overlays:
            ov_desc = "; ".join(
                f"{o.get('selector') or o.get('tag')} ({o.get('text', '')[:60]})" for o in overlays
            )
            lines.append(f"ACTIVE BLOCKING OVERLAYS: {ov_desc}")
        if headings:
            lines.append("Visible Headings: " + " | ".join(str(h) for h in headings[:10]))

        lines.append(f"Visible Interactive Elements ({len(elements)}):")
        for idx, el in enumerate(elements[:40], start=1):
            role = el.get("role") or "button"
            name = el.get("name") or ""
            label = el.get("label") or ""
            placeholder = el.get("placeholder") or ""
            testid = el.get("testid") or ""
            flags: list[str] = []
            if el.get("in_viewport"):
                flags.append("visible")
            if el.get("disabled"):
                flags.append("disabled")
            if el.get("expanded"):
                flags.append(f"expanded={el['expanded']}")
            if el.get("selected"):
                flags.append(f"selected={el['selected']}")
            if el.get("href"):
                flags.append(f"href={el['href'][:45]}")
            parts = [f"[{idx}] role={role}"]
            if name:
                parts.append(f"name={json.dumps(name)}")
            if label and label != name:
                parts.append(f"label={json.dumps(label)}")
            if placeholder and placeholder != name:
                parts.append(f"placeholder={json.dumps(placeholder)}")
            if testid:
                parts.append(f"testid={json.dumps(testid)}")
            if flags:
                parts.append(f"({', '.join(flags)})")
            lines.append("- " + " ".join(parts))

        if vis_text:
            lines.append(f"Visible Text Excerpt: {vis_text[:500]}")
        return "\n".join(lines)

    @staticmethod
    def format_accessible_locator(
        strategy: Optional[str] = None,
        role: Optional[str] = None,
        name: Optional[str] = None,
        selector: Optional[str] = None,
    ) -> str:
        """Format a canonical human-readable Playwright accessible locator string."""
        strat = (strategy or "").strip().lower()
        clean_name = (name or "").strip()
        clean_role = (role or "button").strip().lower()

        if strat == "label" and clean_name:
            return f"getByLabel({json.dumps(clean_name)})"
        if strat == "text" and clean_name:
            return f"getByText({json.dumps(clean_name)})"
        if strat == "placeholder" and clean_name:
            return f"getByPlaceholder({json.dumps(clean_name)})"
        if strat in ("testid", "test_id") and clean_name:
            return f"getByTestId({json.dumps(clean_name)})"
        if strat == "role" or (role and clean_name):
            if clean_name:
                return f"getByRole('{clean_role}', {{ name: {json.dumps(clean_name)} }})"
            return f"getByRole('{clean_role}')"
        if clean_name:
            return f"getByText({json.dumps(clean_name)})"
        return (selector or "body").strip()

    @classmethod
    def build_playwright_locator(
        cls,
        page: Any,
        *,
        strategy: Optional[str] = None,
        role: Optional[str] = None,
        name: Optional[str] = None,
        selector: Optional[str] = None,
    ) -> tuple[Any, str]:
        """
        Resolve a native Playwright Locator via get_by_role / get_by_label / get_by_text /
        get_by_placeholder / get_by_test_id. NEVER converts into ':has-text' CSS selectors.
        """
        strat = (strategy or "").strip().lower()
        clean_name = (name or "").strip()
        clean_role = (role or "").strip().lower()
        clean_sel = (selector or "").strip()

        def _pick_exact_or_fuzzy(exact_loc: Any, fuzzy_loc: Any) -> Any:
            try:
                if hasattr(exact_loc, "count") and exact_loc.count() > 0:
                    return exact_loc
            except Exception:
                pass
            return fuzzy_loc

        if strat == "role" and clean_role:
            disp = cls.format_accessible_locator("role", clean_role, clean_name)
            if hasattr(page, "get_by_role"):
                if clean_name:
                    return _pick_exact_or_fuzzy(
                        page.get_by_role(clean_role, name=clean_name, exact=True),
                        page.get_by_role(clean_role, name=clean_name),
                    ), disp
                return page.get_by_role(clean_role), disp

        if strat == "label" and clean_name and hasattr(page, "get_by_label"):
            disp = cls.format_accessible_locator("label", None, clean_name)
            return _pick_exact_or_fuzzy(
                page.get_by_label(clean_name, exact=True),
                page.get_by_label(clean_name),
            ), disp

        if strat == "text" and clean_name and hasattr(page, "get_by_text"):
            disp = cls.format_accessible_locator("text", None, clean_name)
            return _pick_exact_or_fuzzy(
                page.get_by_text(clean_name, exact=True),
                page.get_by_text(clean_name),
            ), disp

        if strat == "placeholder" and clean_name and hasattr(page, "get_by_placeholder"):
            disp = cls.format_accessible_locator("placeholder", None, clean_name)
            return _pick_exact_or_fuzzy(
                page.get_by_placeholder(clean_name, exact=True),
                page.get_by_placeholder(clean_name),
            ), disp

        if strat in ("testid", "test_id") and clean_name and hasattr(page, "get_by_test_id"):
            disp = cls.format_accessible_locator("testid", None, clean_name)
            return page.get_by_test_id(clean_name), disp

        if clean_role and clean_name and hasattr(page, "get_by_role"):
            disp = cls.format_accessible_locator("role", clean_role, clean_name)
            return _pick_exact_or_fuzzy(
                page.get_by_role(clean_role, name=clean_name, exact=True),
                page.get_by_role(clean_role, name=clean_name),
            ), disp

        if clean_sel:
            # Never pass ':has-text(...)' CSS into page.locator; translate back to accessible locators.
            has_text_m = re.match(
                r"^(?:([a-zA-Z0-9_-]+):has-text|:has-text)\(\s*['\"]([^'\"]+)['\"]\s*\)$",
                clean_sel,
            )
            if has_text_m:
                tag_prefix = (has_text_m.group(1) or "").lower()
                text_val = has_text_m.group(2).strip()
                if tag_prefix in ("button", "a") and hasattr(page, "get_by_role"):
                    mapped_role = "link" if tag_prefix == "a" else "button"
                    return page.get_by_role(mapped_role, name=text_val), cls.format_accessible_locator(
                        "role", mapped_role, text_val
                    )
                if hasattr(page, "get_by_text"):
                    return page.get_by_text(text_val), cls.format_accessible_locator("text", None, text_val)

            loc = cls._to_playwright_locator(page, clean_sel)
            return loc, clean_sel

        if clean_name and hasattr(page, "get_by_text"):
            return page.get_by_text(clean_name), cls.format_accessible_locator("text", None, clean_name)

        return page.locator("body"), "body"

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

        lbl_match = re.search(r"getByLabel\(\s*['\"]([^'\"]+)['\"]\s*\)", selector)
        if lbl_match and hasattr(page, "get_by_label"):
            return page.get_by_label(lbl_match.group(1))

        txt_match = re.search(r"getByText\(\s*['\"]([^'\"]+)['\"]\s*\)", selector)
        if txt_match and hasattr(page, "get_by_text"):
            return page.get_by_text(txt_match.group(1))

        ph_match = re.search(r"getByPlaceholder\(\s*['\"]([^'\"]+)['\"]\s*\)", selector)
        if ph_match and hasattr(page, "get_by_placeholder"):
            return page.get_by_placeholder(ph_match.group(1))

        tid_match = re.search(r"getByTestId\(\s*['\"]([^'\"]+)['\"]\s*\)", selector)
        if tid_match and hasattr(page, "get_by_test_id"):
            return page.get_by_test_id(tid_match.group(1))

        has_text_m = re.match(r"^(?:([a-zA-Z0-9_-]+):has-text|:has-text)\(\s*['\"]([^'\"]+)['\"]\s*\)$", selector.strip())
        if has_text_m:
            tag_prefix = (has_text_m.group(1) or "").lower()
            text_val = has_text_m.group(2).strip()
            if tag_prefix in ("button", "a") and hasattr(page, "get_by_role"):
                return page.get_by_role("link" if tag_prefix == "a" else "button", name=text_val)
            if hasattr(page, "get_by_text"):
                return page.get_by_text(text_val)

        return page.locator(selector)

    @staticmethod
    def wait_visible_and_enabled(page: Any, locator: Any, timeout_ms: int = 9000) -> Any:
        """Explicitly wait for the target element to be visible and enabled (no fixed sleeps)."""
        target = locator.first if hasattr(locator, "first") else locator
        if hasattr(target, "wait_for"):
            target.wait_for(state="visible", timeout=timeout_ms)

        if hasattr(page, "wait_for_function") and hasattr(target, "element_handle"):
            try:
                handle = target.element_handle(timeout=min(timeout_ms, 3000))
                if handle is not None:
                    page.wait_for_function(
                        "el => !el.disabled && el.getAttribute('aria-disabled') !== 'true'",
                        arg=handle,
                        timeout=timeout_ms,
                    )
            except Exception as exc:
                if hasattr(target, "is_enabled") and not target.is_enabled():
                    raise RuntimeError(f"Element is disabled after {timeout_ms}ms wait: {exc}") from exc
        elif hasattr(target, "is_enabled") and not target.is_enabled():
            raise RuntimeError("Target element is visible but disabled.")

        if hasattr(target, "scroll_into_view_if_needed"):
            try:
                target.scroll_into_view_if_needed(timeout=min(timeout_ms, 3000))
            except Exception:
                pass
        return target

    @classmethod
    def execute_playwright_action(
        cls,
        page: Any,
        action: BrowserAction | dict[str, Any],
        target_url: str,
    ) -> tuple[Optional[int], Optional[str]]:
        """Execute one BrowserAction on a Playwright Page using native accessible locators and 9s explicit waits."""
        if isinstance(action, dict):
            action_type = str(action.get("action_type", "assert_visible")).lower()
            strategy = action.get("locator_strategy")
            role = action.get("role")
            name = action.get("name")
            selector = action.get("selector")
            value = action.get("value")
            timeout_ms = int(action.get("timeout_ms") or 9000)
        else:
            action_type = action.action_type
            strategy = getattr(action, "locator_strategy", None)
            role = getattr(action, "role", None)
            name = getattr(action, "name", None)
            selector = action.selector
            value = action.value
            timeout_ms = int(action.timeout_ms or 9000)

        timeout_ms = max(timeout_ms, 8000)

        if action_type == "goto":
            resp = page.goto(value or target_url, wait_until="domcontentloaded", timeout=max(timeout_ms, 20000))
            http_status = getattr(resp, "status", None) if resp else None
            return (int(http_status) if http_status is not None else None), None

        if action_type == "scroll":
            delta = 650
            if value and str(value).lower() == "top":
                if hasattr(page, "evaluate"):
                    page.evaluate("() => window.scrollTo({ top: 0, behavior: 'instant' })")
                return None, "window"
            if value and str(value).lower() == "bottom":
                if hasattr(page, "evaluate"):
                    page.evaluate("() => window.scrollTo({ top: document.body.scrollHeight, behavior: 'instant' })")
                return None, "window"
            if value and str(value).lstrip("-").isdigit():
                delta = int(value)
            if hasattr(page, "evaluate"):
                page.evaluate("(dy) => window.scrollBy(0, dy)", delta)
            return None, "window"

        if action_type == "assert_url":
            expected = value or target_url
            current = str(getattr(page, "url", ""))
            if expected not in current:
                raise AssertionError(f"Expected URL to contain '{expected}', got '{current}'")
            return None, None

        locator, used_label = cls.build_playwright_locator(
            page,
            strategy=strategy,
            role=role,
            name=name,
            selector=selector,
        )

        if action_type == "click":
            target = cls.wait_visible_and_enabled(page, locator, timeout_ms=timeout_ms)
            target.click(timeout=timeout_ms)
            return None, used_label

        if action_type == "fill":
            target = cls.wait_visible_and_enabled(page, locator, timeout_ms=timeout_ms)
            target.fill(value or "", timeout=timeout_ms)
            return None, used_label

        if action_type == "select":
            target = cls.wait_visible_and_enabled(page, locator, timeout_ms=timeout_ms)
            target.select_option(value or "", timeout=timeout_ms)
            return None, used_label

        if action_type == "press":
            target = cls.wait_visible_and_enabled(page, locator, timeout_ms=timeout_ms)
            target.press(value or "Enter", timeout=timeout_ms)
            return None, used_label

        target = locator.first if hasattr(locator, "first") else locator
        if hasattr(target, "wait_for"):
            target.wait_for(state="visible", timeout=timeout_ms)
        return None, used_label

    @classmethod
    def selector_exists_in_dom(cls, selector: Optional[str], dom_snapshot: Optional[str]) -> bool:
        """Return True if a CSS or accessible locator resolves in an HTML snapshot string."""
        if not selector or not dom_snapshot or "<" not in dom_snapshot:
            return False
        cand = selector.strip()
        role_match = re.search(
            r"getByRole\(\s*['\"]([^'\"]+)['\"](?:\s*,\s*\{\s*name:\s*(?:/([^/]+)/([im]*)|['\"]([^'\"]+)['\"])\s*\})?\s*\)",
            cand,
        )
        if role_match:
            target_role = role_match.group(1).strip().lower()
            regex_pat = role_match.group(2)
            exact_name = role_match.group(4)
            for el in cls.extract_interactive_elements(dom_snapshot):
                if el.role != target_role:
                    continue
                haystack = f"{el.accessible_name} {el.text}".strip()
                if regex_pat and re.search(regex_pat, haystack, flags=re.I):
                    return True
                if exact_name and exact_name.lower() in haystack.lower():
                    return True
                if not regex_pat and not exact_name:
                    return True
            return False

        txt_match = re.search(r"(?:getByText|getByLabel|getByPlaceholder)\(\s*['\"]([^'\"]+)['\"]\s*\)", cand)
        if txt_match:
            plain = " ".join(re.sub(r"<[^>]+>", " ", dom_snapshot).split()).lower()
            return txt_match.group(1).lower() in plain

        if cand.startswith("#"):
            return bool(re.search(rf'id=["\'][^"\']*\b{re.escape(cand[1:])}\b', dom_snapshot))
        return cand.lower() in dom_snapshot.lower()

    @classmethod
    def resolve_working_selector(
        cls,
        primary_selector: Optional[str],
        fallback_selectors: list[str],
        dom_snapshot: Optional[str] = None,
        page: Optional[Any] = None,
    ) -> tuple[Optional[str], bool]:
        """Resolve the first working selector from [primary_selector, *fallback_selectors]."""
        ordered = [s for s in [primary_selector, *(fallback_selectors or [])] if s]
        if not ordered:
            return None, False
        if page is not None and hasattr(page, "locator"):
            for idx, cand in enumerate(ordered):
                try:
                    loc = cls._to_playwright_locator(page, cand)
                    if hasattr(loc, "count") and loc.count() > 0:
                        return cand, idx > 0
                except Exception:
                    continue
        if dom_snapshot and "<" in dom_snapshot:
            for idx, cand in enumerate(ordered):
                if cls.selector_exists_in_dom(cand, dom_snapshot):
                    return cand, idx > 0
        return ordered[0], False


@tool
def inspect_dom_elements_tool(html_snapshot: str) -> str:
    """Inspect an HTML DOM snapshot and list interactive UI elements with accessible locators."""
    return BrowserToolkit.summarize_dom_for_agents(html_snapshot)


@tool
def check_selector_in_dom_tool(selector: str, html_snapshot: str) -> str:
    """Check whether an accessible or CSS selector resolves inside the provided HTML DOM snapshot."""
    found = BrowserToolkit.selector_exists_in_dom(selector, html_snapshot)
    return f"Selector '{selector}' matched DOM: {found}"


@tool
def suggest_resilient_locator_tool(broken_selector: str, action_description: str, html_snapshot: str) -> str:
    """Suggest a resilient Playwright ARIA locator replacement for a broken selector."""
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
