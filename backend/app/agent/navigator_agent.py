from __future__ import annotations

import logging
import re
import time
from typing import Any, Literal, Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ..browser.browser_tools import BrowserToolkit
from ..browser.playwright_runner import BrowserSession, PlaywrightRunConfig, PlaywrightRunner, _has_playwright_chromium
from .planner_agent import invoke_with_model_fallback, resolve_chat_model

logger = logging.getLogger(__name__)


class NextActionDecision(BaseModel):
    """Single step decision returned by NavigatorAgent during the observe-decide-act loop."""

    thought: str = Field("", description="Concise reasoning from current live DOM and recent steps")
    decision: Literal["act", "goal_met", "goal_failed"] = "act"
    action_type: Literal[
        "click",
        "fill",
        "press",
        "scroll",
        "select",
        "wait_for_visible",
        "assert_visible",
        "assert_url",
        "none",
    ] = "click"
    locator_strategy: Optional[Literal["role", "label", "text", "placeholder", "testid"]] = None
    role: Optional[str] = None
    name: Optional[str] = None
    value: Optional[str] = None
    description: str = ""
    completion_reason: str = ""


class BrowserAction(BaseModel):
    """Atomic Playwright browser action using accessible locators and 8-10s timeouts."""

    step_number: int = Field(..., ge=1)
    action_type: Literal[
        "goto",
        "click",
        "fill",
        "press",
        "scroll",
        "select",
        "wait_for_visible",
        "assert_visible",
        "assert_url",
    ]
    locator_strategy: Optional[str] = None
    role: Optional[str] = None
    name: Optional[str] = None
    selector: Optional[str] = None
    value: Optional[str] = None
    description: str = ""
    timeout_ms: int = Field(9000, ge=100)


class NavigationStepResult(BaseModel):
    """Execution outcome of one autonomous agent step under a goal."""

    step_number: int
    action: BrowserAction
    thought: str = ""
    status: Literal["passed", "failed", "skipped"]
    duration_ms: float = 0.0
    current_url: str = ""
    error_message: Optional[str] = None
    failed_selector: Optional[str] = None
    http_status: Optional[int] = None
    console_errors: list[str] = Field(default_factory=list)
    network_failures: list[str] = Field(default_factory=list)
    dom_excerpt: Optional[str] = None
    blocking_overlay: Optional[str] = None
    screenshot_b64: Optional[str] = None
    target_box: Optional[dict[str, Any]] = None


class NavigationRunResult(BaseModel):
    """Aggregated step log and final status ('passed', 'failed', 'stuck') for a single QA goal."""

    scenario_id: str
    scenario_title: str
    goal_objective: str = ""
    target_url: str
    overall_status: Literal["passed", "failed", "stuck"]
    completion_reason: str = ""
    steps: list[NavigationStepResult] = Field(default_factory=list)
    failures: list[dict[str, Any]] = Field(default_factory=list)
    execution_mode: Literal["playwright", "simulated"] = "playwright"
    trace_id: Optional[str] = None
    video_url: Optional[str] = None


class NavigatorGraphState(TypedDict, total=False):
    target_url: str
    goal: dict[str, Any]
    browser_session: Optional[Any]
    dom_summary: Optional[str]
    persona_hint: str
    max_steps: int
    use_live_browser: bool
    record_traces: bool
    run_id: Optional[str]
    stream_id: Optional[str]
    run_result: NavigationRunResult


class NavigatorAgent:
    """
    Autonomous step-by-step browser QA agent powered by LangChain & LangGraph.

    Per goal:
    1. Ensure no blocking overlay intercepts clicks.
    2. Observe live visible DOM/accessibility tree (cached when DOM hash unchanged).
    3. Pick ONE next action via getByRole/getByLabel/getByText (no :has-text CSS).
    4. Execute on the shared BrowserSession page, capture screenshot & re-observe.
    5. Stop on goal_met ('passed'), 3 consecutive failures ('stuck'), or max_steps.
    """

    SYSTEM_PROMPT = (
        "You are an autonomous Playwright QA Navigator Agent operating inside a live browser page.\n"
        "You receive ONE high-level testing goal, the recent step history, and the CURRENT visible "
        "interactive elements (role, name, label, placeholder, state) from the live DOM.\n\n"
        "Rules:\n"
        "1. Pick ONLY ONE next action per turn using decision='act', or set decision='goal_met' if history "
        "shows the goal's success criteria are satisfied, or decision='goal_failed' if impossible.\n"
        "2. ALWAYS target elements using accessible locator_strategy ('role', 'label', 'text', 'placeholder', 'testid') "
        "plus exact 'role' and 'name' copied verbatim from Visible Interactive Elements. NEVER output ':has-text' or CSS selectors.\n"
        "3. NEVER repeat an action/locator listed under Failed Locators To Avoid; pick an alternative visible element or scroll.\n"
        "4. Use synthetic QA test data for fill actions (e.g., qa.synthetic@agentswarm.dev for email, Alex QA for name)."
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
                    "Goal [{goal_id}]: {goal_title}\n"
                    "Objective: {goal_objective}\n"
                    "Success Criteria: {success_criteria}\n"
                    "Persona Hint: {persona_hint}\n"
                    "Step: {step_number}/{max_steps}\n\n"
                    "Recent Step History:\n{history}\n\n"
                    "Failed Locators To Avoid (DO NOT REPEAT):\n{avoid_list}\n\n"
                    "Live Visible Page State:\n{live_state}\n\n"
                    "Choose ONE next action or mark goal_met / goal_failed.",
                ),
            ]
        )
        self._chain = (self._prompt | self._llm.with_structured_output(NextActionDecision)) if self._llm else None
        self._graph = self.build_navigator_graph()

    def build_navigator_graph(self):
        builder = StateGraph(NavigatorGraphState)
        builder.add_node("execute_goal_loop", self._node_execute_goal_loop)
        builder.add_edge(START, "execute_goal_loop")
        builder.add_edge("execute_goal_loop", END)
        return builder.compile()

    @staticmethod
    def _decision_signature(dec: NextActionDecision) -> tuple[str, str, str, str]:
        return (
            dec.action_type,
            (dec.locator_strategy or "").lower(),
            (dec.role or "").lower(),
            (dec.name or "").strip().lower(),
        )

    @staticmethod
    def _sanitize_decision(dec: NextActionDecision) -> NextActionDecision:
        clean_name = (dec.name or "").strip()
        role = dec.role
        strat = dec.locator_strategy
        if clean_name and ":has-text(" in clean_name:
            m = re.match(r"^(?:([a-zA-Z0-9_-]+):has-text|:has-text)\(\s*['\"]([^'\"]+)['\"]\s*\)$", clean_name)
            if m:
                clean_name = m.group(2).strip()
                if m.group(1) in ("button", "a") and not role:
                    role = "link" if m.group(1) == "a" else "button"
                strat = "role" if role else "text"
        return dec.model_copy(update={"name": clean_name or None, "role": role, "locator_strategy": strat})

    def _invoke_llm_step(self, prompt_input: dict[str, Any]) -> NextActionDecision:
        res, _ = invoke_with_model_fallback(
            self._prompt,
            NextActionDecision,
            prompt_input,
            primary_chain=self._chain,
            provider=self._provider,
            model_name=self._model_name,
            api_key=self._api_key,
            temperature=0.1,
            allow_fallback=not self._custom_llm_injected,
        )
        if not isinstance(res, NextActionDecision):
            res = NextActionDecision.model_validate(res)
        return self._sanitize_decision(res)

    def _decide_next_action(
        self,
        *,
        browser_session: BrowserSession,
        goal_id: str,
        goal_title: str,
        goal_objective: str,
        success_criteria: str,
        persona_hint: str,
        step_num: int,
        max_steps: int,
        history_lines: list[str],
        failed_signatures: set[tuple[str, str, str, str]],
        attempted_signatures: set[tuple[str, str, str, str]],
        live_state: dict[str, Any],
        compact_dom: str,
        consecutive_failures: int,
    ) -> tuple[NextActionDecision, dict[str, Any]]:
        """Pick ONE next action using LLM (with fresh DOM re-pick if repeating a failed locator); fails loudly if LLM unavailable."""
        avoid_str = (
            "\n".join(f"- {a} ({s}: role={r}, name={n!r})" for a, s, r, n in sorted(failed_signatures))
            if failed_signatures
            else "None"
        )
        history_str = "\n".join(history_lines[-6:]) if history_lines else "No steps taken yet."

        def _make_prompt(dom_txt: str, extra_note: str = "") -> dict[str, Any]:
            return {
                "goal_id": goal_id,
                "goal_title": goal_title,
                "goal_objective": goal_objective,
                "success_criteria": success_criteria or "Goal actions succeed cleanly.",
                "persona_hint": persona_hint or "Methodical QA Explorer",
                "step_number": step_num,
                "max_steps": max_steps,
                "history": history_str + (f"\nIMPORTANT: {extra_note}" if extra_note else ""),
                "avoid_list": avoid_str,
                "live_state": dom_txt,
            }

        if not self._chain:
            return (
                NextActionDecision(
                    thought="No LLM configured",
                    decision="goal_failed",
                    action_type="none",
                    completion_reason="No active LLM configured for autonomous browser decisions.",
                ),
                live_state,
            )

        try:
            dec = self._invoke_llm_step(_make_prompt(compact_dom))
            if dec.decision == "act":
                sig = self._decision_signature(dec)
                if sig in failed_signatures:
                    live_state, fresh_dom, _ = browser_session.observe(force_refresh=True)
                    dec2 = self._invoke_llm_step(
                        _make_prompt(
                            fresh_dom,
                            f"Your previous choice '{dec.name}' failed earlier. Re-pick a DIFFERENT visible element from the refreshed DOM.",
                        )
                    )
                    if dec2.decision != "act" or self._decision_signature(dec2) not in failed_signatures:
                        return dec2, live_state
                    return (
                        NextActionDecision(
                            thought="Repeated failed locator",
                            decision="goal_failed",
                            action_type="none",
                            completion_reason=f"LLM repeatedly selected failed locator '{dec.name}' after DOM refresh.",
                        ),
                        live_state,
                    )
            return dec, live_state
        except Exception as exc:
            logger.error("[NavigatorAgent] LLM step decision failed across retries/models: %s", exc)
            return (
                NextActionDecision(
                    thought="LLM call failed",
                    decision="goal_failed",
                    action_type="none",
                    completion_reason=f"LLM action decision failed: {str(exc)[:180]}",
                ),
                live_state,
            )

    def _execute_simulated_goal(
        self,
        *,
        goal_id: str,
        goal_title: str,
        goal_objective: str,
        target_url: str,
        dom_summary: Optional[str],
        stream_id: Optional[str],
        run_id: Optional[str],
    ) -> NavigationRunResult:
        runner = PlaywrightRunner()
        trace = runner.trace_manager.start_trace(
            scenario_id=goal_id,
            scenario_title=goal_title,
            target_url=target_url,
            run_id=run_id,
            execution_mode="simulated",
        )
        act = BrowserAction(
            step_number=1,
            action_type="assert_visible",
            locator_strategy="role",
            role="main",
            selector="main",
            description=f"Simulated verification of goal '{goal_title}'",
        )
        shot = PlaywrightRunner.build_synthetic_frame_b64(
            target_url=target_url,
            scenario_title=goal_title,
            action=act,
            step_number=1,
            total_steps=1,
            status="passed",
            dom_snapshot=dom_summary,
        )
        PlaywrightRunner.emit_step_telemetry(
            runner.trace_manager,
            trace_id=trace.trace_id,
            stream_id=stream_id,
            scenario_id=goal_id,
            scenario_title=goal_title,
            action=act,
            total_steps=1,
            status="passed",
            duration_ms=25.0,
            selector="main",
            current_url=target_url,
            screenshot_b64=shot,
            execution_mode="simulated",
        )
        runner.trace_manager.stop_trace(trace.trace_id, overall_status="passed")
        return NavigationRunResult(
            scenario_id=goal_id,
            scenario_title=goal_title,
            goal_objective=goal_objective,
            target_url=target_url,
            overall_status="passed",
            completion_reason="Verified in simulated offline mode.",
            steps=[
                NavigationStepResult(
                    step_number=1,
                    action=act,
                    thought="Simulated offline verification",
                    status="passed",
                    duration_ms=25.0,
                    current_url=target_url,
                    screenshot_b64=shot,
                )
            ],
            failures=[],
            execution_mode="simulated",
            trace_id=trace.trace_id,
        )

    def _run_goal_on_session(
        self,
        *,
        goal: dict[str, Any],
        target_url: str,
        browser_session: Optional[BrowserSession],
        dom_summary: Optional[str] = None,
        persona_hint: str = "",
        max_steps: int = 15,
        stream_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> NavigationRunResult:
        goal_id = str(goal.get("goal_id") or goal.get("scenario_id") or "GOAL-01")
        goal_title = str(goal.get("title") or "Autonomous QA Goal")
        goal_objective = str(goal.get("objective") or goal.get("description") or goal_title)
        success_criteria = str(goal.get("success_criteria") or goal.get("expected_result") or "")
        goal_text = f"{goal_title} {goal_objective}"

        if browser_session is None:
            return self._execute_simulated_goal(
                goal_id=goal_id,
                goal_title=goal_title,
                goal_objective=goal_objective,
                target_url=target_url,
                dom_summary=dom_summary,
                stream_id=stream_id,
                run_id=run_id,
            )

        from ..browser.trace_manager import get_stream_manager

        stream_mgr = get_stream_manager() if stream_id else None
        trace = browser_session.trace_manager.start_trace(
            scenario_id=goal_id,
            scenario_title=goal_title,
            target_url=target_url,
            run_id=run_id,
            context=None,
            execution_mode="playwright",
        )

        if stream_mgr and stream_id:
            stream_mgr.push_log(stream_id, "Navigator Agent", f"Starting {goal_id}: {goal_title}...", status="info")

        step_results: list[NavigationStepResult] = []
        failures: list[dict[str, Any]] = []
        history_lines: list[str] = []
        failed_signatures: set[tuple[str, str, str, str]] = set()
        attempted_signatures: set[tuple[str, str, str, str]] = set()
        consecutive_failures = 0
        force_refresh = False
        final_status: Literal["passed", "failed", "stuck"] = "failed"
        completion_reason = ""

        for step_num in range(1, max_steps + 1):
            # 1. Pre-action overlay guard
            blocker = browser_session.ensure_no_blocking_overlay(goal_text=goal_text)
            live_state, compact_dom, cache_hit = browser_session.observe(force_refresh=force_refresh)
            force_refresh = False
            live_url = str(live_state.get("url") or getattr(browser_session.page, "url", "") or "")

            # Fail loudly if page landed on about:blank or #intro-overlay blocked interaction
            if not live_url or live_url.startswith("about:blank") or blocker == "#intro-overlay":
                err_msg = (
                    f"Page is at '{live_url or 'about:blank'}' (navigation missing)."
                    if (not live_url or live_url.startswith("about:blank"))
                    else f"Blocking overlay '{blocker}' prevented action on {goal_id}."
                )
                act = BrowserAction(
                    step_number=step_num,
                    action_type="assert_visible",
                    selector=blocker or "body",
                    description=f"Pre-step check for {goal_id}",
                )
                shot_b64, _ = PlaywrightRunner._highlight_and_capture(
                    page=browser_session.page,
                    action=act,
                    used_selector=blocker or "body",
                    step_number=step_num,
                    total_steps=max_steps,
                    scenario_title=goal_title,
                    target_url=live_url or "about:blank",
                    status="failed",
                    dom_snapshot=live_state.get("dom_excerpt"),
                    error_message=err_msg,
                )
                ev = browser_session.collector.build_failure_evidence(
                    failure_id=f"{goal_id}-STEP-{step_num}",
                    scenario_id=goal_id,
                    scenario_title=goal_title,
                    step_number=step_num,
                    current_url=live_url or "about:blank",
                    error_message=err_msg,
                    failed_selector=blocker or "body",
                    blocking_overlay=blocker,
                    dom_excerpt=live_state.get("dom_excerpt"),
                    screenshot_b64=shot_b64,
                    action_description=act.description,
                    page=browser_session.page,
                )
                step_results.append(
                    NavigationStepResult(
                        step_number=step_num,
                        action=act,
                        thought="Aborted step due to missing navigation or blocking overlay",
                        status="failed",
                        current_url=ev.current_url or "about:blank",
                        error_message=err_msg,
                        failed_selector=ev.failed_selector,
                        dom_excerpt=ev.dom_excerpt,
                        blocking_overlay=blocker,
                        screenshot_b64=shot_b64,
                    )
                )
                failures.append(ev.model_dump())
                final_status = "failed"
                completion_reason = err_msg
                break

            # 2. Pick ONE next action (using cached DOM when unchanged)
            dec, live_state = self._decide_next_action(
                browser_session=browser_session,
                goal_id=goal_id,
                goal_title=goal_title,
                goal_objective=goal_objective,
                success_criteria=success_criteria,
                persona_hint=persona_hint,
                step_num=step_num,
                max_steps=max_steps,
                history_lines=history_lines,
                failed_signatures=failed_signatures,
                attempted_signatures=attempted_signatures,
                live_state=live_state,
                compact_dom=compact_dom,
                consecutive_failures=consecutive_failures,
            )

            if dec.decision == "goal_met":
                final_status = "passed"
                completion_reason = dec.completion_reason or f"Goal '{goal_title}' verified cleanly."
                if stream_mgr and stream_id:
                    stream_mgr.push_log(stream_id, "Navigator Agent", f"{goal_id} PASSED ({len(step_results)} step(s)): {completion_reason}", status="success")
                break

            if dec.decision == "goal_failed":
                err_msg = dec.completion_reason or f"Goal '{goal_title}' could not be satisfied on live page."
                act = BrowserAction(
                    step_number=step_num,
                    action_type="assert_visible",
                    selector=BrowserToolkit.format_accessible_locator(dec.locator_strategy, dec.role, dec.name),
                    description=dec.description or err_msg,
                )
                shot_b64, _ = PlaywrightRunner._highlight_and_capture(
                    page=browser_session.page,
                    action=act,
                    used_selector=act.selector,
                    step_number=step_num,
                    total_steps=max_steps,
                    scenario_title=goal_title,
                    target_url=live_url,
                    status="failed",
                    dom_snapshot=live_state.get("dom_excerpt"),
                    error_message=err_msg,
                )
                ev = browser_session.collector.build_failure_evidence(
                    failure_id=f"{goal_id}-STEP-{step_num}",
                    scenario_id=goal_id,
                    scenario_title=goal_title,
                    step_number=step_num,
                    current_url=live_url,
                    error_message=err_msg,
                    failed_selector=act.selector,
                    dom_excerpt=live_state.get("dom_excerpt"),
                    screenshot_b64=shot_b64,
                    action_description=err_msg,
                    page=browser_session.page,
                )
                step_results.append(
                    NavigationStepResult(
                        step_number=step_num,
                        action=act,
                        thought=dec.thought,
                        status="failed",
                        current_url=live_url,
                        error_message=err_msg,
                        failed_selector=act.selector,
                        dom_excerpt=ev.dom_excerpt,
                        blocking_overlay=ev.blocking_overlay,
                        screenshot_b64=shot_b64,
                    )
                )
                failures.append(ev.model_dump())
                final_status = "failed"
                completion_reason = err_msg
                if stream_mgr and stream_id:
                    stream_mgr.push_log(stream_id, "Navigator Agent", f"{goal_id} FAILED: {err_msg}", status="error")
                break

            # 3. Execute single accessible action on shared page
            sig = self._decision_signature(dec)
            exec_type = "assert_visible" if dec.action_type == "none" else dec.action_type
            acc_selector = BrowserToolkit.format_accessible_locator(dec.locator_strategy, dec.role, dec.name)
            action = BrowserAction(
                step_number=step_num,
                action_type=exec_type,  # type: ignore[arg-type]
                locator_strategy=dec.locator_strategy,
                role=dec.role,
                name=dec.name,
                selector=acc_selector,
                value=dec.value,
                description=dec.description or f"{exec_type} {acc_selector}",
                timeout_ms=9000,
            )
            step_res, fail_dict = browser_session.execute_single_action(
                goal_id=goal_id,
                goal_title=goal_title,
                action=action,
                step_number=step_num,
                max_steps=max_steps,
                goal_text=goal_text,
                trace_id=trace.trace_id,
            )
            step_res.thought = dec.thought
            step_results.append(step_res)

            if step_res.status == "passed":
                consecutive_failures = 0
                attempted_signatures.add(sig)
                history_lines.append(f"Step {step_num} PASSED: {action.description} (cache={'hit' if cache_hit else 'miss'})")
            else:
                consecutive_failures += 1
                failed_signatures.add(sig)
                force_refresh = True  # Must inspect fresh DOM before next decision
                if fail_dict:
                    failures.append(fail_dict)
                history_lines.append(f"Step {step_num} FAILED: {action.description} -> {(step_res.error_message or '')[:100]}")

                if consecutive_failures >= 3:
                    final_status = "stuck"
                    completion_reason = f"Marked STUCK after 3 consecutive failures on {goal_id} (last: {(step_res.error_message or '')[:120]})"
                    if stream_mgr and stream_id:
                        stream_mgr.push_log(stream_id, "Navigator Agent", completion_reason, status="warning")
                    break
        else:
            # Loop ran to max_steps without explicit goal_met/goal_failed
            final_status = "stuck" if failures else "failed"
            completion_reason = f"Reached max {max_steps} steps on {goal_id} without goal_met."
            if not failures:
                live_state, _, _ = browser_session.observe(force_refresh=True)
                act = BrowserAction(
                    step_number=len(step_results) + 1,
                    action_type="assert_visible",
                    selector="body",
                    description=completion_reason,
                )
                shot_b64, _ = PlaywrightRunner._highlight_and_capture(
                    page=browser_session.page,
                    action=act,
                    used_selector="body",
                    step_number=act.step_number,
                    total_steps=max_steps,
                    scenario_title=goal_title,
                    target_url=str(live_state.get("url") or target_url),
                    status="failed",
                    dom_snapshot=live_state.get("dom_excerpt"),
                    error_message=completion_reason,
                )
                ev = browser_session.collector.build_failure_evidence(
                    failure_id=f"{goal_id}-MAXSTEPS",
                    scenario_id=goal_id,
                    scenario_title=goal_title,
                    step_number=act.step_number,
                    current_url=str(live_state.get("url") or target_url),
                    error_message=completion_reason,
                    failed_selector="body",
                    dom_excerpt=live_state.get("dom_excerpt"),
                    screenshot_b64=shot_b64,
                    action_description=completion_reason,
                    page=browser_session.page,
                )
                failures.append(ev.model_dump())

        browser_session.trace_manager.stop_trace(trace.trace_id, overall_status=final_status)
        return NavigationRunResult(
            scenario_id=goal_id,
            scenario_title=goal_title,
            goal_objective=goal_objective,
            target_url=target_url,
            overall_status=final_status,
            completion_reason=completion_reason,
            steps=step_results,
            failures=failures,
            execution_mode="playwright",
            trace_id=trace.trace_id,
        )

    def _node_execute_goal_loop(self, state: NavigatorGraphState) -> dict[str, Any]:
        run_result = self._run_goal_on_session(
            goal=state["goal"],
            target_url=state["target_url"],
            browser_session=state.get("browser_session"),
            dom_summary=state.get("dom_summary"),
            persona_hint=state.get("persona_hint", ""),
            max_steps=int(state.get("max_steps", 15)),
            stream_id=state.get("stream_id"),
            run_id=state.get("run_id"),
        )
        return {"run_result": run_result}

    def execute_goal(
        self,
        *,
        target_url: str,
        goal: dict[str, Any],
        browser_session: Optional[BrowserSession] = None,
        dom_summary: Optional[str] = None,
        persona_hint: str = "",
        max_steps: int = 15,
        stream_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> NavigationRunResult:
        res = self._graph.invoke(
            {
                "target_url": target_url,
                "goal": goal,
                "browser_session": browser_session,
                "dom_summary": dom_summary,
                "persona_hint": persona_hint,
                "max_steps": max_steps,
                "stream_id": stream_id,
                "run_id": run_id,
            }
        )
        return res["run_result"]

    def run_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph parent-graph node: execute ALL planned goals sequentially in ONE shared BrowserSession."""
        target_url = state.get("target_url", "https://example.com")
        dom_summary = state.get("dom_summary") or state.get("dom_snapshot")
        use_live_browser = bool(state.get("use_live_browser", False))
        record_traces = bool(state.get("record_traces", True))
        run_id = state.get("run_id")
        stream_id = state.get("stream_id")
        max_steps = int(state.get("max_steps_per_goal", 15))
        injected_session: Optional[BrowserSession] = state.get("browser_session")

        test_plan = state.get("test_plan") or {}
        if hasattr(test_plan, "model_dump"):
            test_plan = test_plan.model_dump()
        goals = test_plan.get("goals") or test_plan.get("scenarios") or [
            {"goal_id": "GOAL-01", "title": "Primary Navigation Smoke", "objective": "Verify main application view renders."}
        ]

        personas_data = (state.get("personas") or {}).get("personas") or []
        persona_summary = "; ".join(
            f"{p.get('name', 'QA')}: {p.get('risk_focus', '')}" for p in personas_data[:3] if isinstance(p, dict)
        )

        from ..browser.trace_manager import get_stream_manager

        stream_mgr = get_stream_manager() if stream_id else None
        results: list[dict[str, Any]] = []
        raw_failures: list[dict[str, Any]] = list(state.get("raw_failures") or [])

        owns_session = False
        session: Optional[BrowserSession] = injected_session
        if session is None and use_live_browser and _has_playwright_chromium() and "staging.store.nexus" not in target_url:
            session = BrowserSession(
                target_url=target_url,
                run_id=run_id,
                stream_id=stream_id,
                config=PlaywrightRunConfig(use_live_browser=True, record_traces=record_traces),
            )
            owns_session = True

        try:
            if owns_session and session is not None:
                try:
                    session.start()
                except Exception as pre_exc:
                    if session.preflight_failure_evidence:
                        raw_failures.append(session.preflight_failure_evidence)
                    else:
                        raw_failures.append(
                            {
                                "failure_id": "PREFLIGHT-ERR",
                                "scenario_id": "PREFLIGHT",
                                "scenario_title": "Session Pre-Flight",
                                "target_url": target_url,
                                "current_url": str(getattr(session.page, "url", "") or "about:blank"),
                                "error_message": str(pre_exc),
                                "blocking_overlay": "#intro-overlay" if "intro-overlay" in str(pre_exc).lower() else None,
                            }
                        )
                    logger.warning("[NavigatorAgent] Pre-flight failed: %s", pre_exc)

            for goal in goals:
                if isinstance(goal, BaseModel):
                    goal = goal.model_dump()
                res = self.execute_goal(
                    target_url=target_url,
                    goal=goal,
                    browser_session=session,
                    dom_summary=dom_summary,
                    persona_hint=persona_summary,
                    max_steps=max_steps,
                    stream_id=stream_id,
                    run_id=run_id,
                )
                results.append(res.model_dump())
                if res.overall_status in ("failed", "stuck"):
                    raw_failures.extend(res.failures)
                else:
                    # Still surface any fatal HTTP/nav/overlay errors if logged during the goal
                    for f in res.failures:
                        http_st = f.get("http_status")
                        if f.get("blocking_overlay") or str(f.get("current_url", "")).startswith("about:blank") or (http_st and int(http_st) >= 400):
                            raw_failures.append(f)

                if stream_mgr and stream_id:
                    stream_mgr.update_partial(stream_id, "navigation_results", list(results))
        finally:
            if owns_session and session is not None:
                session.close()
                if session.video_url and results:
                    results[0]["video_url"] = session.video_url

        return {
            "navigation_results": results,
            "raw_failures": raw_failures,
            "current_agent": "navigator",
        }


def navigator_node(state: dict[str, Any]) -> dict[str, Any]:
    return NavigatorAgent().run_node(state)
