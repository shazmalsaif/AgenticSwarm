from __future__ import annotations

import base64
import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from ..agent.triage_agent import FailureEvidence

logger = logging.getLogger(__name__)


def _read_member(obj: Any, attr: str, default: Any = None) -> Any:
    if obj is None:
        return default
    val = getattr(obj, attr, default)
    if callable(val):
        try:
            return val()
        except TypeError:
            return default
    return val


class TelemetrySnapshot(BaseModel):
    """Point-in-time browser telemetry and DOM state captured during execution."""

    current_url: str = ""
    last_http_status: Optional[int] = None
    console_errors: list[str] = Field(default_factory=list)
    network_failures: list[str] = Field(default_factory=list)
    dom_excerpt: Optional[str] = None
    screenshot_b64: Optional[str] = None


class EvidenceCollector:
    """Captures browser console, network, screenshot, and DOM telemetry for TriageAgent."""

    def __init__(self) -> None:
        self.console_errors: list[str] = []
        self.network_failures: list[str] = []
        self.page_errors: list[str] = []
        self.last_http_status: Optional[int] = None
        self._attached_page_ids: set[int] = set()

    def attach_to_page(self, page: Any) -> None:
        """Attach Playwright event listeners to capture console, uncaught errors, and HTTP failures."""
        if page is None or not hasattr(page, "on"):
            return
        page_id = id(page)
        if page_id in self._attached_page_ids:
            return
        self._attached_page_ids.add(page_id)

        def _on_console(msg: Any) -> None:
            try:
                msg_type = str(_read_member(msg, "type", "") or "").lower()
                text = str(_read_member(msg, "text", str(msg)) or "").strip()
                if msg_type in ("error", "assert") and text:
                    self._append_unique(self.console_errors, f"[{msg_type}] {text}")
            except Exception as exc:
                logger.debug("[EvidenceCollector] Console handler error: %s", exc)

        def _on_pageerror(err: Any) -> None:
            try:
                text = str(err).strip()
                if text:
                    self._append_unique(self.page_errors, text)
                    self._append_unique(self.console_errors, f"[pageerror] {text}")
            except Exception as exc:
                logger.debug("[EvidenceCollector] PageError handler error: %s", exc)

        def _on_response(resp: Any) -> None:
            try:
                status_val = _read_member(resp, "status", None)
                if status_val is None:
                    return
                status_code = int(status_val)
                url = str(_read_member(resp, "url", "") or "")
                req = getattr(resp, "request", None)
                method = str(_read_member(req, "method", "GET") or "GET").upper()
                resource_type = str(_read_member(req, "resource_type", "") or "").lower()

                if status_code >= 400:
                    self.last_http_status = status_code
                    entry = f"HTTP {status_code} {method} {url}".strip()
                    self._append_unique(self.network_failures, entry)
                elif self.last_http_status is None or (
                    self.last_http_status < 400 and resource_type in ("document", "")
                ):
                    self.last_http_status = status_code
            except Exception as exc:
                logger.debug("[EvidenceCollector] Response handler error: %s", exc)

        def _on_requestfailed(req: Any) -> None:
            try:
                method = str(_read_member(req, "method", "GET") or "GET").upper()
                url = str(_read_member(req, "url", "") or "")
                failure = _read_member(req, "failure", "request_failed")
                if isinstance(failure, dict):
                    failure_text = str(failure.get("errorText") or failure)
                else:
                    failure_text = str(failure or "request_failed")
                entry = f"{method} {url} -> {failure_text}".strip()
                self._append_unique(self.network_failures, entry)
            except Exception as exc:
                logger.debug("[EvidenceCollector] RequestFailed handler error: %s", exc)

        for event_name, handler in (
            ("console", _on_console),
            ("pageerror", _on_pageerror),
            ("response", _on_response),
            ("requestfailed", _on_requestfailed),
        ):
            try:
                page.on(event_name, handler)
            except Exception as exc:
                logger.debug("[EvidenceCollector] Failed attaching '%s' listener: %s", event_name, exc)

    @staticmethod
    def _append_unique(target: list[str], value: str) -> None:
        if value and value not in target:
            target.append(value)

    def ingest_simulated_telemetry(
        self,
        http_status: Optional[int] = None,
        console_errors: Optional[list[str]] = None,
        network_failures: Optional[list[str]] = None,
    ) -> None:
        """Populate telemetry buffers from simulated steps or external runtime logs."""
        if http_status is not None:
            self.last_http_status = http_status
        for err in console_errors or []:
            self._append_unique(self.console_errors, str(err))
        for nf in network_failures or []:
            self._append_unique(self.network_failures, str(nf))

    @staticmethod
    def extract_dom_excerpt(
        page: Optional[Any] = None,
        fallback_dom: Optional[str] = None,
        max_chars: int = 2000,
    ) -> Optional[str]:
        """Extract a compact DOM excerpt from a live Playwright page or fallback HTML."""
        if page is not None and hasattr(page, "content"):
            try:
                html = page.content()
                if html:
                    return str(html)[:max_chars]
            except Exception as exc:
                logger.debug("[EvidenceCollector] page.content() failed: %s", exc)
        if fallback_dom:
            return str(fallback_dom)[:max_chars]
        return None

    @staticmethod
    def capture_screenshot_base64(page: Optional[Any] = None) -> Optional[str]:
        """Safely capture a base64-encoded JPEG data URL screenshot from a Playwright page."""
        if page is None or not hasattr(page, "screenshot"):
            return None
        try:
            raw = page.screenshot(type="jpeg", quality=75)
            if isinstance(raw, (bytes, bytearray)) and len(raw) > 0:
                return "data:image/jpeg;base64," + base64.b64encode(bytes(raw)).decode("ascii")
        except Exception as exc:
            logger.debug("[EvidenceCollector] Screenshot capture skipped: %s", exc)
        return None

    def snapshot(
        self,
        page: Optional[Any] = None,
        fallback_url: str = "",
        fallback_dom: Optional[str] = None,
    ) -> TelemetrySnapshot:
        """Build a point-in-time TelemetrySnapshot across current buffers and page state."""
        current_url = str(getattr(page, "url", "") or fallback_url) if page is not None else fallback_url
        return TelemetrySnapshot(
            current_url=current_url,
            last_http_status=self.last_http_status,
            console_errors=list(self.console_errors),
            network_failures=list(self.network_failures),
            dom_excerpt=self.extract_dom_excerpt(page=page, fallback_dom=fallback_dom),
            screenshot_b64=self.capture_screenshot_base64(page=page),
        )

    def build_failure_evidence(
        self,
        *,
        failure_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
        scenario_title: Optional[str] = None,
        step_number: Optional[int] = None,
        target_url: str = "",
        current_url: Optional[str] = None,
        error_message: str,
        failed_selector: Optional[str] = None,
        http_status: Optional[int] = None,
        dom_excerpt: Optional[str] = None,
        screenshot_b64: Optional[str] = None,
        blocking_overlay: Optional[str] = None,
        action_description: Optional[str] = None,
        page: Optional[Any] = None,
        fallback_dom: Optional[str] = None,
    ) -> FailureEvidence:
        """Build a validated FailureEvidence payload with live URL, overlay, screenshot, and DOM excerpt."""
        resolved_status = http_status if http_status is not None else self.last_http_status
        live_page_url = str(getattr(page, "url", "")) if page is not None else ""
        resolved_current_url = (current_url if current_url is not None else live_page_url) or target_url
        resolved_overlay = blocking_overlay
        if resolved_overlay is None and page is not None:
            try:
                from .browser_tools import BrowserToolkit

                state = BrowserToolkit.extract_live_page_state(page, max_elements=10)
                overlays = state.get("blocking_overlays") or []
                if overlays:
                    resolved_overlay = ", ".join(
                        str(o.get("selector") or o.get("tag") or "overlay") for o in overlays
                    )
                if not dom_excerpt and state.get("dom_excerpt"):
                    dom_excerpt = str(state["dom_excerpt"])
            except Exception:
                pass
        resolved_dom = dom_excerpt or self.extract_dom_excerpt(page=page, fallback_dom=fallback_dom)
        resolved_shot = screenshot_b64 or self.capture_screenshot_base64(page=page)
        return FailureEvidence(
            failure_id=failure_id,
            scenario_id=scenario_id,
            scenario_title=scenario_title,
            step_number=step_number,
            target_url=resolved_current_url or target_url,
            current_url=resolved_current_url,
            error_message=error_message,
            failed_selector=failed_selector,
            http_status=resolved_status,
            console_errors=list(self.console_errors),
            network_failures=list(self.network_failures),
            dom_excerpt=resolved_dom,
            screenshot_b64=resolved_shot,
            blocking_overlay=resolved_overlay,
            action_description=action_description,
        )

    def reset(self) -> None:
        """Clear collected console, network, and status telemetry buffers."""
        self.console_errors.clear()
        self.network_failures.clear()
        self.page_errors.clear()
        self.last_http_status = None
