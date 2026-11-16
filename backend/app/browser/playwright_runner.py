from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel

from .browser_tools import BrowserToolkit
from .evidence_collector import EvidenceCollector
from .trace_manager import TraceManager, get_trace_manager

if TYPE_CHECKING:
    from ..agent.navigator_agent import BrowserAction, NavigationRunResult, NavigationStepResult

logger = logging.getLogger(__name__)


class PlaywrightRunConfig(BaseModel):
    """Runtime execution configuration for AgentSwarm browser sessions."""

    headless: bool = True
    use_live_browser: bool = False
    record_traces: bool = True
    save_traces_to_disk: bool = False
    viewport_width: int = 1280
    viewport_height: int = 720
    default_timeout_ms: int = 5000


class PlaywrightRunner:
    """
    Unified browser runner coordinating Playwright execution, EvidenceCollector telemetry,
    BrowserToolkit fallback resolution, and TraceManager step artifacts.
    """

    def __init__(
        self,
        trace_manager: Optional[TraceManager] = None,
        default_config: Optional[PlaywrightRunConfig] = None,
    ) -> None:
        self.trace_manager = trace_manager or get_trace_manager()
        self.default_config = default_config or PlaywrightRunConfig()

    def capture_dom_snapshot(
        self,
        target_url: str,
        use_live_browser: bool = False,
        fallback_html: Optional[str] = None,
        headless: bool = True,
        timeout_ms: int = 10000,
    ) -> dict[str, Optional[str]]:
        """
        Capture HTML DOM snapshot and interactive summary for planning and locator healing.

        Never launches Chromium unless `use_live_browser=True`; degrades safely to `fallback_html`.
        """
        if not use_live_browser:
            summary = BrowserToolkit.summarize_dom_for_agents(fallback_html) if fallback_html else None
            return {"dom_snapshot": fallback_html, "dom_summary": summary}

        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=headless)
                try:
                    context = browser.new_context()
                    page = context.new_page()
                    page.goto(target_url, timeout=timeout_ms)
                    html = page.content()
                    summary = BrowserToolkit.summarize_dom_for_agents(html)
                    return {"dom_snapshot": html, "dom_summary": summary}
                finally:
                    browser.close()
        except Exception as exc:
            logger.warning("[PlaywrightRunner] Live snapshot capture failed for %s: %s", target_url, exc)
            summary = BrowserToolkit.summarize_dom_for_agents(fallback_html) if fallback_html else None
            return {"dom_snapshot": fallback_html, "dom_summary": summary}

    def execute_actions(
        self,
        scenario_id: str,
        scenario_title: str,
        target_url: str,
        actions: list[BrowserAction | dict[str, Any]],
        dom_snapshot: Optional[str] = None,
        page: Optional[Any] = None,
        run_id: Optional[str] = None,
        config: Optional[PlaywrightRunConfig] = None,
    ) -> NavigationRunResult:
        """Execute BrowserAction steps via Playwright (live or injected page) or simulated DOM checks."""
        from ..agent.navigator_agent import BrowserAction

        cfg = config or self.default_config
        coerced_actions = [
            act if isinstance(act, BrowserAction) else BrowserAction.model_validate(act)
            for act in actions
        ]

        if page is not None:
            return self._execute_on_page(
                scenario_id=scenario_id,
                scenario_title=scenario_title,
                target_url=target_url,
                actions=coerced_actions,
                page=page,
                context=getattr(page, "context", None),
                dom_snapshot=dom_snapshot,
                run_id=run_id,
                cfg=cfg,
            )

        if cfg.use_live_browser:
            try:
                from playwright.sync_api import sync_playwright

                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=cfg.headless)
                    try:
                        context = browser.new_context(
                            viewport={"width": cfg.viewport_width, "height": cfg.viewport_height}
                        )
                        live_page = context.new_page()
                        return self._execute_on_page(
                            scenario_id=scenario_id,
                            scenario_title=scenario_title,
                            target_url=target_url,
                            actions=coerced_actions,
                            page=live_page,
                            context=context,
                            dom_snapshot=dom_snapshot,
                            run_id=run_id,
                            cfg=cfg,
                        )
                    finally:
                        browser.close()
            except Exception as exc:
                logger.warning(
                    "[PlaywrightRunner] Live browser launch unavailable (%s); falling back to simulated runner.",
                    exc,
                )

        return self._execute_simulated(
            scenario_id=scenario_id,
            scenario_title=scenario_title,
            target_url=target_url,
            actions=coerced_actions,
            dom_snapshot=dom_snapshot,
            run_id=run_id,
            cfg=cfg,
        )

    def _execute_on_page(
        self,
        *,
        scenario_id: str,
        scenario_title: str,
        target_url: str,
        actions: list[BrowserAction],
        page: Any,
        context: Optional[Any],
        dom_snapshot: Optional[str],
        run_id: Optional[str],
        cfg: PlaywrightRunConfig,
    ) -> NavigationRunResult:
        from ..agent.navigator_agent import NavigationRunResult, NavigationStepResult

        trace = (
            self.trace_manager.start_trace(
                scenario_id=scenario_id,
                scenario_title=scenario_title,
                target_url=target_url,
                run_id=run_id,
                context=context,
                execution_mode="playwright",
            )
            if cfg.record_traces
            else None
        )
        trace_id = trace.trace_id if trace else None

        collector = EvidenceCollector()
        collector.attach_to_page(page)

        step_results: list[NavigationStepResult] = []
        failures: list[dict[str, Any]] = []
        failed_already = False

        for action in actions:
            if failed_already:
                current_url = str(getattr(page, "url", "") or target_url)
                step_results.append(
                    NavigationStepResult(
                        step_number=action.step_number,
                        action=action,
                        status="skipped",
                        current_url=current_url,
                    )
                )
                if trace_id:
                    self.trace_manager.record_step(
                        trace_id=trace_id,
                        step_number=action.step_number,
                        action_type=action.action_type,
                        status="skipped",
                        selector=action.selector,
                        value=action.value,
                        current_url=current_url,
                    )
                continue

            started = time.perf_counter()
            try:
                action_status, used_selector = BrowserToolkit.execute_playwright_action(
                    page=page,
                    action=action,
                    target_url=target_url,
                )
                if action_status is not None and (
                    collector.last_http_status is None or action_status >= 400
                ):
                    collector.ingest_simulated_telemetry(http_status=action_status)

                elapsed = round((time.perf_counter() - started) * 1000, 2)
                current_url = str(getattr(page, "url", "") or target_url)
                used_fallback = (
                    used_selector
                    if used_selector and action.selector and used_selector != action.selector
                    else None
                )
                step_results.append(
                    NavigationStepResult(
                        step_number=action.step_number,
                        action=action,
                        status="passed",
                        duration_ms=elapsed,
                        current_url=current_url,
                        http_status=action_status if action_status is not None else collector.last_http_status,
                        console_errors=list(collector.console_errors),
                        network_failures=list(collector.network_failures),
                        used_fallback_selector=used_fallback,
                    )
                )
                if trace_id:
                    self.trace_manager.record_step(
                        trace_id=trace_id,
                        step_number=action.step_number,
                        action_type=action.action_type,
                        status="passed",
                        duration_ms=elapsed,
                        selector=used_selector or action.selector,
                        value=action.value,
                        current_url=current_url,
                    )
            except Exception as exc:
                failed_already = True
                elapsed = round((time.perf_counter() - started) * 1000, 2)
                current_url = str(getattr(page, "url", "") or target_url)
                err_msg = str(exc)
                evidence = collector.build_failure_evidence(
                    failure_id=f"{scenario_id}-STEP-{action.step_number}",
                    scenario_id=scenario_id,
                    scenario_title=scenario_title,
                    step_number=action.step_number,
                    target_url=current_url,
                    error_message=err_msg,
                    failed_selector=action.selector,
                    action_description=action.description,
                    page=page,
                    fallback_dom=dom_snapshot,
                )
                step_results.append(
                    NavigationStepResult(
                        step_number=action.step_number,
                        action=action,
                        status="failed",
                        duration_ms=elapsed,
                        current_url=current_url,
                        error_message=err_msg,
                        failed_selector=action.selector,
                        http_status=evidence.http_status,
                        console_errors=list(evidence.console_errors),
                        network_failures=list(evidence.network_failures),
                        dom_excerpt=evidence.dom_excerpt,
                    )
                )
                failures.append(evidence.model_dump())
                if trace_id:
                    self.trace_manager.record_step(
                        trace_id=trace_id,
                        step_number=action.step_number,
                        action_type=action.action_type,
                        status="failed",
                        duration_ms=elapsed,
                        selector=action.selector,
                        value=action.value,
                        current_url=current_url,
                        error_message=err_msg,
                    )

        overall_status = "failed" if failures else "passed"
        if trace_id:
            self.trace_manager.stop_trace(
                trace_id=trace_id,
                overall_status=overall_status,
                context=context,
                save_to_disk=cfg.save_traces_to_disk,
            )

        return NavigationRunResult(
            scenario_id=scenario_id,
            scenario_title=scenario_title,
            target_url=target_url,
            overall_status=overall_status,
            steps=step_results,
            failures=failures,
            execution_mode="playwright",
            trace_id=trace_id,
        )

    def _execute_simulated(
        self,
        *,
        scenario_id: str,
        scenario_title: str,
        target_url: str,
        actions: list[BrowserAction],
        dom_snapshot: Optional[str],
        run_id: Optional[str],
        cfg: PlaywrightRunConfig,
    ) -> NavigationRunResult:
        from ..agent.navigator_agent import NavigationRunResult, NavigationStepResult

        trace = (
            self.trace_manager.start_trace(
                scenario_id=scenario_id,
                scenario_title=scenario_title,
                target_url=target_url,
                run_id=run_id,
                execution_mode="simulated",
            )
            if cfg.record_traces
            else None
        )
        trace_id = trace.trace_id if trace else None

        collector = EvidenceCollector()
        has_html = bool(dom_snapshot and "<" in dom_snapshot)
        step_results: list[NavigationStepResult] = []
        failures: list[dict[str, Any]] = []
        failed_already = False
        current_url = target_url

        for action in actions:
            if failed_already:
                step_results.append(
                    NavigationStepResult(
                        step_number=action.step_number,
                        action=action,
                        status="skipped",
                        current_url=current_url,
                    )
                )
                if trace_id:
                    self.trace_manager.record_step(
                        trace_id=trace_id,
                        step_number=action.step_number,
                        action_type=action.action_type,
                        status="skipped",
                        selector=action.selector,
                        value=action.value,
                        current_url=current_url,
                    )
                continue

            if action.action_type == "goto":
                if action.value:
                    current_url = action.value
                collector.ingest_simulated_telemetry(http_status=200)
                step_results.append(
                    NavigationStepResult(
                        step_number=action.step_number,
                        action=action,
                        status="passed",
                        duration_ms=42.0,
                        current_url=current_url,
                        http_status=200,
                    )
                )
                if trace_id:
                    self.trace_manager.record_step(
                        trace_id=trace_id,
                        step_number=action.step_number,
                        action_type=action.action_type,
                        status="passed",
                        duration_ms=42.0,
                        value=action.value,
                        current_url=current_url,
                    )
                continue

            if action.action_type == "assert_url":
                expected = action.value or target_url
                if expected not in current_url:
                    failed_already = True
                    err_msg = f"Expected URL to contain '{expected}', got '{current_url}'"
                    evidence = collector.build_failure_evidence(
                        failure_id=f"{scenario_id}-STEP-{action.step_number}",
                        scenario_id=scenario_id,
                        scenario_title=scenario_title,
                        step_number=action.step_number,
                        target_url=current_url,
                        error_message=err_msg,
                        http_status=collector.last_http_status or 200,
                        action_description=action.description,
                        fallback_dom=dom_snapshot,
                    )
                    step_results.append(
                        NavigationStepResult(
                            step_number=action.step_number,
                            action=action,
                            status="failed",
                            duration_ms=15.0,
                            current_url=current_url,
                            error_message=err_msg,
                            http_status=evidence.http_status,
                            dom_excerpt=evidence.dom_excerpt,
                        )
                    )
                    failures.append(evidence.model_dump())
                    if trace_id:
                        self.trace_manager.record_step(
                            trace_id=trace_id,
                            step_number=action.step_number,
                            action_type=action.action_type,
                            status="failed",
                            duration_ms=15.0,
                            value=action.value,
                            current_url=current_url,
                            error_message=err_msg,
                        )
                    continue

            used_fallback: Optional[str] = None
            selector_broken = False
            if action.action_type in ("click", "fill", "wait_for_selector", "assert_visible") and has_html:
                chosen, fallback_used = BrowserToolkit.resolve_working_selector(
                    primary_selector=action.selector,
                    fallback_selectors=action.fallback_selectors,
                    dom_snapshot=dom_snapshot,
                )
                if chosen and not BrowserToolkit.selector_exists_in_dom(chosen, dom_snapshot):
                    selector_broken = True
                elif fallback_used and chosen:
                    used_fallback = chosen

            if selector_broken:
                failed_already = True
                err_msg = f"Timeout {action.timeout_ms}ms waiting for selector '{action.selector}'"
                evidence = collector.build_failure_evidence(
                    failure_id=f"{scenario_id}-STEP-{action.step_number}",
                    scenario_id=scenario_id,
                    scenario_title=scenario_title,
                    step_number=action.step_number,
                    target_url=current_url,
                    error_message=err_msg,
                    failed_selector=action.selector,
                    http_status=collector.last_http_status or 200,
                    action_description=action.description,
                    fallback_dom=dom_snapshot,
                )
                step_results.append(
                    NavigationStepResult(
                        step_number=action.step_number,
                        action=action,
                        status="failed",
                        duration_ms=float(action.timeout_ms),
                        current_url=current_url,
                        error_message=err_msg,
                        failed_selector=action.selector,
                        http_status=evidence.http_status,
                        console_errors=list(evidence.console_errors),
                        network_failures=list(evidence.network_failures),
                        dom_excerpt=evidence.dom_excerpt,
                    )
                )
                failures.append(evidence.model_dump())
                if trace_id:
                    self.trace_manager.record_step(
                        trace_id=trace_id,
                        step_number=action.step_number,
                        action_type=action.action_type,
                        status="failed",
                        duration_ms=float(action.timeout_ms),
                        selector=action.selector,
                        value=action.value,
                        current_url=current_url,
                        error_message=err_msg,
                    )
            else:
                step_results.append(
                    NavigationStepResult(
                        step_number=action.step_number,
                        action=action,
                        status="passed",
                        duration_ms=42.0,
                        current_url=current_url,
                        used_fallback_selector=used_fallback,
                    )
                )
                if trace_id:
                    self.trace_manager.record_step(
                        trace_id=trace_id,
                        step_number=action.step_number,
                        action_type=action.action_type,
                        status="passed",
                        duration_ms=42.0,
                        selector=used_fallback or action.selector,
                        value=action.value,
                        current_url=current_url,
                    )

        overall_status = "failed" if failures else "passed"
        if trace_id:
            self.trace_manager.stop_trace(
                trace_id=trace_id,
                overall_status=overall_status,
            )

        return NavigationRunResult(
            scenario_id=scenario_id,
            scenario_title=scenario_title,
            target_url=target_url,
            overall_status=overall_status,
            steps=step_results,
            failures=failures,
            execution_mode="simulated",
            trace_id=trace_id,
        )
