from __future__ import annotations

import base64
import html as html_lib
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel

from .browser_tools import BrowserToolkit
from .evidence_collector import EvidenceCollector
from .trace_manager import TraceManager, get_stream_manager, get_trace_manager

if TYPE_CHECKING:
    from ..agent.navigator_agent import BrowserAction, NavigationStepResult

logger = logging.getLogger(__name__)

_DEMO_STAGING_HTML = """<!DOCTYPE html>
<html lang="en">
<head><title>Nexus Store Staging — Checkout</title></head>
<body>
  <header>
    <nav>
      <a href="/catalog" aria-label="Catalog">Catalog</a>
      <a href="/cart" aria-label="Shopping Cart">Cart (2)</a>
      <button id="btn-signin" type="button">Sign In</button>
    </nav>
  </header>
  <main>
    <h1>Enterprise Precision Laptop (M4 Pro)</h1>
    <p>SKU: PROD-9941 • Price: $1,499.00</p>
    <form id="checkout-form">
      <input id="checkout_email" name="email" type="email" placeholder="Work Email" aria-label="Work Email" />
      <input id="promo_code" name="promo_code" type="text" placeholder="Promo Code" aria-label="Promo Code" />
      <button id="apply_promo_btn" type="button">Apply Promo</button>
      <button id="add_to_cart_btn" type="button">Add to Cart</button>
      <button id="checkout_btn_v2" type="submit">Proceed to Checkout</button>
    </form>
  </main>
</body>
</html>"""


def _playwright_browser_dirs() -> list[Path]:
    """Return candidate directories where Playwright stores cached browser binaries."""
    dirs: list[Path] = []
    env_path = os.getenv("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    if env_path and env_path != "0":
        dirs.append(Path(env_path))
    dirs.append(Path("/ms-playwright"))
    local_app = os.getenv("LOCALAPPDATA", "").strip()
    if local_app:
        dirs.append(Path(local_app) / "ms-playwright")
    dirs.extend([Path.home() / ".cache" / "ms-playwright", Path.home() / "AppData" / "Local" / "ms-playwright"])
    return dirs


def _find_bundled_chromium_executable() -> Optional[str]:
    """Locate Playwright's own cached chromium / chromium_headless_shell executable if installed."""
    for base in _playwright_browser_dirs():
        if not base.exists():
            continue
        for pat in ("chromium*", "chromium_headless_shell*"):
            for sub in base.glob(pat):
                for rel in (
                    sub / "chrome-win" / "chrome.exe",
                    sub / "chrome-win64" / "chrome.exe",
                    sub / "chrome-linux" / "chrome",
                    sub / "chrome-linux64" / "chrome",
                    sub / "chrome-linux" / "headless_shell",
                    sub / "chrome-linux64" / "headless_shell",
                    sub / "chrome-headless-shell-linux64" / "chrome-headless-shell",
                    sub / "chrome-mac" / "Chromium.app" / "Contents" / "MacOS" / "Chromium",
                ):
                    if rel.exists():
                        return str(rel)
    return None


def _has_bundled_playwright_chromium() -> bool:
    return _find_bundled_chromium_executable() is not None


def _resolve_browser_executable() -> Optional[str]:
    """Locate bundled Playwright Chromium or system Google Chrome / Microsoft Edge executable."""
    bundled = _find_bundled_chromium_executable()
    if bundled:
        return bundled

    local_app = os.getenv("LOCALAPPDATA", "").strip()
    system_browsers = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path(local_app) / "Google" / "Chrome" / "Application" / "chrome.exe" if local_app else Path(""),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path("/usr/bin/google-chrome"),
        Path("/usr/bin/chromium"),
        Path("/usr/bin/chromium-browser"),
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    ]
    for path in system_browsers:
        if str(path) and path.exists() and path.is_file():
            return str(path)
    return None


def _has_playwright_chromium() -> bool:
    return _has_bundled_playwright_chromium() or _resolve_browser_executable() is not None


def _has_playwright_ffmpeg() -> bool:
    for base in _playwright_browser_dirs():
        if not base.exists():
            continue
        for sub in base.glob("ffmpeg*"):
            for fname in ("ffmpeg-win64.exe", "ffmpeg-win32.exe", "ffmpeg-linux", "ffmpeg-mac"):
                if (sub / fname).exists():
                    return True
    return False


def _launch_chromium(playwright_instance: Any, headless: bool = True) -> Any:
    args = ["--disable-blink-features=AutomationControlled"]
    if os.name != "nt":
        args.extend(["--no-sandbox", "--disable-dev-shm-usage"])

    launch_kwargs: dict[str, Any] = {
        "headless": headless,
        "args": args,
    }
    if not _has_bundled_playwright_chromium():
        exe_path = _resolve_browser_executable()
        if exe_path:
            launch_kwargs["executable_path"] = exe_path
    return playwright_instance.chromium.launch(**launch_kwargs)


class PlaywrightRunConfig(BaseModel):
    """Runtime execution configuration for single-session AgentSwarm browser runs."""

    headless: bool = True
    use_live_browser: bool = False
    record_traces: bool = True
    record_video: bool = True
    save_traces_to_disk: bool = False
    viewport_width: int = 1280
    viewport_height: int = 720
    default_timeout_ms: int = 9000
    navigation_timeout_ms: int = 20000
    preflight_timeout_ms: int = 10000


class BrowserSession:
    """
    Single continuous Playwright browser/context/page session shared across an entire test run.

    Opens ONE browser context & page, performs ONE goto at startup, runs ONE mandatory
    pre-flight check to clear #intro-overlay, caches live DOM state when unchanged,
    and executes individual autonomous agent actions without reloading between goals.
    """

    def __init__(
        self,
        target_url: str,
        *,
        run_id: Optional[str] = None,
        stream_id: Optional[str] = None,
        config: Optional[PlaywrightRunConfig] = None,
        trace_manager: Optional[TraceManager] = None,
        injected_page: Optional[Any] = None,
    ) -> None:
        self.target_url = target_url
        self.run_id = run_id
        self.stream_id = stream_id
        self.config = config or PlaywrightRunConfig()
        self.trace_manager = trace_manager or get_trace_manager()
        self.collector = EvidenceCollector()
        self._injected_page = injected_page

        self._pw: Optional[Any] = None
        self._browser: Optional[Any] = None
        self.context: Optional[Any] = None
        self.page: Optional[Any] = None
        self._owns_browser = injected_page is None
        self._videos_dir = self.trace_manager.artifacts_dir / "videos"
        self._use_video = False

        self.goto_count: int = 0
        self.preflight_done: bool = False
        self.preflight_error: Optional[str] = None
        self.preflight_failure_evidence: Optional[dict[str, Any]] = None
        self.video_url: Optional[str] = None

        # DOM observation cache
        self._cached_dom_hash: Optional[str] = None
        self._cached_state: Optional[dict[str, Any]] = None
        self._cached_llm_summary: Optional[str] = None
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    def __enter__(self) -> "BrowserSession":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def start(self) -> "BrowserSession":
        """Open single browser/context/page, navigate once to target_url, and run pre-flight check."""
        stream_mgr = get_stream_manager()
        if self._injected_page is not None:
            self.page = self._injected_page
            self.context = getattr(self.page, "context", None)
        else:
            from playwright.sync_api import sync_playwright

            self._use_video = bool(self.config.record_video and _has_playwright_ffmpeg())
            if self._use_video:
                self._videos_dir.mkdir(parents=True, exist_ok=True)
            self._pw = sync_playwright().start()
            self._browser = _launch_chromium(self._pw, headless=self.config.headless)
            ctx_kwargs: dict[str, Any] = {
                "viewport": {"width": self.config.viewport_width, "height": self.config.viewport_height},
                "user_agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                ),
            }
            if self._use_video:
                ctx_kwargs["record_video_dir"] = str(self._videos_dir)
                ctx_kwargs["record_video_size"] = {
                    "width": self.config.viewport_width,
                    "height": self.config.viewport_height,
                }
            self.context = self._browser.new_context(**ctx_kwargs)
            self.page = self.context.new_page()

        if hasattr(self.page, "set_default_timeout"):
            self.page.set_default_timeout(self.config.default_timeout_ms)
        if hasattr(self.page, "set_default_navigation_timeout"):
            self.page.set_default_navigation_timeout(self.config.navigation_timeout_ms)

        self.collector.attach_to_page(self.page)
        if self.config.record_traces and self.context is not None and hasattr(self.context, "tracing"):
            try:
                self.context.tracing.start(screenshots=True, snapshots=True, sources=True)
            except Exception as exc:
                logger.debug("[BrowserSession] Context tracing start skipped: %s", exc)

        if self.stream_id:
            stream_mgr.push_log(
                self.stream_id,
                "Browser Session",
                f"Opening single continuous session -> {self.target_url} (goto #1)...",
                status="info",
            )

        # Single initial goto for the entire session
        resp = self.page.goto(self.target_url, wait_until="domcontentloaded", timeout=self.config.navigation_timeout_ms)
        self.goto_count += 1
        http_status = getattr(resp, "status", None) if resp else None
        if http_status is not None and self.collector.last_http_status is None:
            self.collector.last_http_status = int(http_status)

        self._wait_for_initial_mount()
        self.run_preflight()
        return self

    def _wait_for_initial_mount(self) -> None:
        """Explicitly wait for rendered content & interactive controls (zero fixed sleeps)."""
        if self.page is None:
            return
        try:
            body = self.page.locator("body").first
            if hasattr(body, "wait_for"):
                body.wait_for(state="visible", timeout=self.config.preflight_timeout_ms)
        except Exception:
            pass
        if hasattr(self.page, "wait_for_function"):
            try:
                self.page.wait_for_function(
                    "() => document.readyState === 'complete' || document.querySelectorAll('a,button,input').length > 0",
                    timeout=self.config.preflight_timeout_ms,
                )
            except Exception:
                pass
        if hasattr(self.page, "wait_for_load_state"):
            try:
                self.page.wait_for_load_state("load", timeout=min(self.config.preflight_timeout_ms, 5000))
            except Exception:
                pass

    def _dismiss_intro_overlay(self) -> tuple[bool, Optional[str]]:
        """
        Attempt mandatory dismissal of #intro-overlay or entry splash gate.
        Returns `(dismissed_cleanly, error_detail)`. Never marks intro skip optional if overlay persists.
        """
        if self.page is None:
            return False, "No active Playwright page"

        def _is_intro_still_visible() -> bool:
            state = BrowserToolkit.extract_live_page_state(self.page, max_elements=12)
            return any(bool(o.get("is_intro")) or o.get("selector") == "#intro-overlay" for o in (state.get("blocking_overlays") or []))

        if not _is_intro_still_visible():
            return True, None

        btn_candidates: list[Any] = []
        if hasattr(self.page, "get_by_role"):
            btn_candidates.append(self.page.get_by_role("button", name="Skip intro animation"))
            btn_candidates.append(self.page.get_by_role("button", name=re.compile(r"skip\s*intro|skip\s*animation|enter", re.I)))
        if hasattr(self.page, "get_by_text"):
            btn_candidates.append(self.page.get_by_text(re.compile(r"skip\s*intro|click\s+anywhere\s+to\s+enter", re.I)))

        clicked = False
        for cand in btn_candidates:
            try:
                if hasattr(cand, "count") and cand.count() == 0:
                    continue
                target = BrowserToolkit.wait_visible_and_enabled(
                    self.page,
                    cand,
                    timeout_ms=self.config.default_timeout_ms,
                )
                target.click(timeout=self.config.default_timeout_ms)
                clicked = True
                break
            except Exception:
                continue

        intro_loc = self.page.locator("#intro-overlay").first
        if clicked:
            try:
                if hasattr(intro_loc, "wait_for"):
                    intro_loc.wait_for(state="hidden", timeout=self.config.preflight_timeout_ms)
            except Exception:
                pass

        # Second attempt if overlay uses a click-anywhere handler instead of a separate button click
        if _is_intro_still_visible():
            try:
                if hasattr(intro_loc, "count") and intro_loc.count() > 0:
                    intro_loc.click(timeout=min(self.config.default_timeout_ms, 5000))
                    intro_loc.wait_for(state="hidden", timeout=min(self.config.preflight_timeout_ms, 5000))
            except Exception:
                pass

        if _is_intro_still_visible():
            return False, "#intro-overlay remained visible and blocked page interaction after dismissal attempts."

        self.invalidate_dom_cache()
        return True, None

    def run_preflight(self) -> dict[str, Any]:
        """Run once after initial goto: dismiss #intro-overlay loudly (never optional) and capture Frame 0."""
        stream_mgr = get_stream_manager()
        ok, err_detail = self._dismiss_intro_overlay()
        self.invalidate_dom_cache()
        live_state, _, _ = self.observe(force_refresh=True)
        shot_b64, _ = PlaywrightRunner._highlight_and_capture(
            page=self.page,
            action=None,
            used_selector=None,
            step_number=1,
            total_steps=1,
            scenario_title="Pre-Flight Check",
            target_url=str(live_state.get("url") or self.target_url),
            status="passed" if ok else "failed",
            dom_snapshot=live_state.get("dom_excerpt"),
            error_message=err_detail,
        )

        if not ok:
            self.preflight_done = False
            self.preflight_error = err_detail or "#intro-overlay still blocking"
            ev = self.collector.build_failure_evidence(
                failure_id="PREFLIGHT-OVERLAY",
                scenario_id="PREFLIGHT",
                scenario_title="Pre-Flight Intro Dismissal",
                step_number=1,
                current_url=str(live_state.get("url") or getattr(self.page, "url", "")),
                error_message=self.preflight_error,
                failed_selector="getByRole('button', { name: 'Skip intro animation' })",
                blocking_overlay="#intro-overlay",
                dom_excerpt=live_state.get("dom_excerpt"),
                screenshot_b64=shot_b64,
                action_description="Dismiss #intro-overlay before starting goals",
                page=self.page,
            )
            self.preflight_failure_evidence = ev.model_dump()
            if self.stream_id:
                stream_mgr.push_frame(
                    self.stream_id,
                    {
                        "scenario_id": "PREFLIGHT",
                        "scenario_title": "Pre-Flight Check",
                        "step_number": 1,
                        "total_steps": 1,
                        "action_type": "click",
                        "selector": "getByRole('button', { name: 'Skip intro animation' })",
                        "description": self.preflight_error,
                        "status": "failed",
                        "duration_ms": 0.0,
                        "current_url": ev.current_url,
                        "error_message": self.preflight_error,
                        "screenshot_b64": shot_b64,
                        "execution_mode": "playwright",
                    },
                )
                stream_mgr.push_log(self.stream_id, "Pre-Flight", self.preflight_error, status="error")
            raise RuntimeError(self.preflight_error)

        self.preflight_done = True
        self.preflight_error = None
        el_count = len(live_state.get("interactive_elements") or [])
        if self.stream_id:
            if shot_b64:
                stream_mgr.push_frame(
                    self.stream_id,
                    {
                        "scenario_id": "PREFLIGHT",
                        "scenario_title": "Pre-Flight & Intro Dismissal",
                        "step_number": 1,
                        "total_steps": 1,
                        "action_type": "goto",
                        "selector": None,
                        "value": str(live_state.get("url") or self.target_url),
                        "description": f"Loaded {self.target_url} & cleared blocking overlays ({el_count} visible interactive elements)",
                        "status": "passed",
                        "duration_ms": 0.0,
                        "current_url": str(live_state.get("url") or self.target_url),
                        "screenshot_b64": shot_b64,
                        "target_box": None,
                        "execution_mode": "playwright",
                    },
                )
            stream_mgr.push_log(
                self.stream_id,
                "Pre-Flight",
                f"Page ready at {live_state.get('url') or self.target_url}; intro cleared ({el_count} visible interactive elements).",
                status="success",
            )
        return live_state

    def ensure_no_blocking_overlay(self, goal_text: str = "") -> Optional[str]:
        """
        Pre-action guard: if a blocking overlay/dialog covers interaction, handle it first.
        Never ignores #intro-overlay. Returns remaining blocking overlay descriptor if still blocked.
        """
        if self.page is None:
            return None
        state = BrowserToolkit.extract_live_page_state(self.page, max_elements=10)
        overlays = state.get("blocking_overlays") or []
        if not overlays:
            return None

        has_intro = any(bool(o.get("is_intro")) or o.get("selector") == "#intro-overlay" for o in overlays)
        if has_intro:
            ok, _ = self._dismiss_intro_overlay()
            if not ok:
                return "#intro-overlay"
            state = BrowserToolkit.extract_live_page_state(self.page, max_elements=10)
            overlays = state.get("blocking_overlays") or []
            if not overlays:
                return None

        goal_low = (goal_text or "").lower()
        is_modal_goal = any(tok in goal_low for tok in ("modal", "dialog", "popup", "drawer", "overlay"))
        if is_modal_goal:
            return None

        # Non-goal blocking modal/cookie banner: try dismissing once cleanly
        if hasattr(self.page, "get_by_role"):
            try:
                close_btn = self.page.get_by_role("button", name=re.compile(r"^(close|dismiss|accept|got it|continue|ok)$", re.I))
                if hasattr(close_btn, "count") and close_btn.count() > 0:
                    target = BrowserToolkit.wait_visible_and_enabled(self.page, close_btn, timeout_ms=2500)
                    target.click(timeout=3000)
                    self._wait_for_dom_paint()
                    self.invalidate_dom_cache()
            except Exception:
                pass
        state_after = BrowserToolkit.extract_live_page_state(self.page, max_elements=10)
        rem = state_after.get("blocking_overlays") or []
        if not rem:
            return None
        return ", ".join(str(o.get("selector") or o.get("tag") or "overlay") for o in rem)

    def _wait_for_dom_paint(self) -> None:
        """Explicit animation-frame + load-state settle without fixed sleep timers."""
        if self.page is None:
            return
        if hasattr(self.page, "wait_for_load_state"):
            try:
                self.page.wait_for_load_state("domcontentloaded", timeout=2500)
            except Exception:
                pass
        if hasattr(self.page, "evaluate"):
            try:
                self.page.evaluate(
                    "() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"
                )
            except Exception:
                pass

    def invalidate_dom_cache(self) -> None:
        self._cached_dom_hash = None

    def observe(self, force_refresh: bool = False) -> tuple[dict[str, Any], str, bool]:
        """Return `(live_state, compact_llm_summary, cache_hit)` reusing cached snapshot when DOM is unchanged."""
        if not force_refresh and self._cached_dom_hash and self._cached_state is not None and self._cached_llm_summary is not None:
            fp = BrowserToolkit.compute_page_fingerprint(self.page)
            if fp and fp == self._cached_dom_hash:
                self.cache_hits += 1
                return self._cached_state, self._cached_llm_summary, True

        state = BrowserToolkit.extract_live_page_state(self.page, max_elements=45)
        summary = BrowserToolkit.format_live_state_for_llm(state)
        self._cached_dom_hash = str(state.get("dom_hash") or "")
        self._cached_state = state
        self._cached_llm_summary = summary
        self.cache_misses += 1
        return state, summary, False

    def execute_single_action(
        self,
        *,
        goal_id: str,
        goal_title: str,
        action: BrowserAction,
        step_number: int,
        max_steps: int,
        goal_text: str = "",
        trace_id: Optional[str] = None,
    ) -> tuple[NavigationStepResult, Optional[dict[str, Any]]]:
        """Execute ONE autonomous step on the shared live Page and emit telemetry + failure evidence."""
        from ..agent.navigator_agent import NavigationStepResult

        started = time.perf_counter()
        stream_mgr = get_stream_manager()

        live_url = str(getattr(self.page, "url", "") or "")
        if not live_url or live_url.startswith("about:blank"):
            err_msg = f"Page is not navigated to target app (current url='{live_url or 'about:blank'}')."
            self.invalidate_dom_cache()
            state, _, _ = self.observe(force_refresh=True)
            shot_b64, _ = PlaywrightRunner._highlight_and_capture(
                page=self.page,
                action=action,
                used_selector=action.selector,
                step_number=step_number,
                total_steps=max_steps,
                scenario_title=goal_title,
                target_url=live_url,
                status="failed",
                dom_snapshot=state.get("dom_excerpt"),
                error_message=err_msg,
            )
            ev = self.collector.build_failure_evidence(
                failure_id=f"{goal_id}-STEP-{step_number}",
                scenario_id=goal_id,
                scenario_title=goal_title,
                step_number=step_number,
                current_url=live_url or "about:blank",
                error_message=err_msg,
                failed_selector=action.selector,
                dom_excerpt=state.get("dom_excerpt"),
                screenshot_b64=shot_b64,
                action_description=action.description,
                page=self.page,
            )
            step_res = NavigationStepResult(
                step_number=step_number,
                action=action,
                status="failed",
                duration_ms=0.0,
                current_url=live_url or "about:blank",
                error_message=err_msg,
                failed_selector=action.selector,
                dom_excerpt=ev.dom_excerpt,
                screenshot_b64=shot_b64,
                blocking_overlay=ev.blocking_overlay,
            )
            PlaywrightRunner.emit_step_telemetry(
                self.trace_manager,
                trace_id=trace_id,
                stream_id=self.stream_id,
                scenario_id=goal_id,
                scenario_title=goal_title,
                action=action,
                total_steps=max_steps,
                status="failed",
                duration_ms=0.0,
                selector=action.selector,
                current_url=live_url or "about:blank",
                error_message=err_msg,
                screenshot_b64=shot_b64,
                execution_mode="playwright",
            )
            return step_res, ev.model_dump()

        # Pre-action blocking overlay guard
        blocker = self.ensure_no_blocking_overlay(goal_text=goal_text)
        if blocker:
            err_msg = f"Blocking overlay '{blocker}' intercepts pointer events before step {step_number}."
            self.invalidate_dom_cache()
            state, _, _ = self.observe(force_refresh=True)
            shot_b64, _ = PlaywrightRunner._highlight_and_capture(
                page=self.page,
                action=action,
                used_selector=action.selector,
                step_number=step_number,
                total_steps=max_steps,
                scenario_title=goal_title,
                target_url=live_url,
                status="failed",
                dom_snapshot=state.get("dom_excerpt"),
                error_message=err_msg,
            )
            ev = self.collector.build_failure_evidence(
                failure_id=f"{goal_id}-STEP-{step_number}",
                scenario_id=goal_id,
                scenario_title=goal_title,
                step_number=step_number,
                current_url=live_url,
                error_message=err_msg,
                failed_selector=action.selector,
                blocking_overlay=blocker,
                dom_excerpt=state.get("dom_excerpt"),
                screenshot_b64=shot_b64,
                action_description=action.description,
                page=self.page,
            )
            step_res = NavigationStepResult(
                step_number=step_number,
                action=action,
                status="failed",
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                current_url=live_url,
                error_message=err_msg,
                failed_selector=action.selector,
                dom_excerpt=ev.dom_excerpt,
                screenshot_b64=shot_b64,
                blocking_overlay=blocker,
            )
            PlaywrightRunner.emit_step_telemetry(
                self.trace_manager,
                trace_id=trace_id,
                stream_id=self.stream_id,
                scenario_id=goal_id,
                scenario_title=goal_title,
                action=action,
                total_steps=max_steps,
                status="failed",
                duration_ms=step_res.duration_ms,
                selector=action.selector,
                current_url=live_url,
                error_message=err_msg,
                screenshot_b64=shot_b64,
                execution_mode="playwright",
            )
            return step_res, ev.model_dump()

        pre_box: Optional[dict[str, Any]] = None
        if action.action_type not in ("goto", "scroll", "assert_url"):
            try:
                pre_loc, _ = BrowserToolkit.build_playwright_locator(
                    self.page,
                    strategy=action.locator_strategy,
                    role=action.role,
                    name=action.name,
                    selector=action.selector,
                )
                first_pre = pre_loc.first if hasattr(pre_loc, "first") else pre_loc
                if hasattr(first_pre, "bounding_box"):
                    raw_pre = first_pre.bounding_box(timeout=600)
                    if isinstance(raw_pre, dict) and raw_pre.get("width", 0) > 0:
                        pre_box = {
                            "x_pct": round(max(0.0, min(95.0, (raw_pre["x"] / 1280.0) * 100.0)), 1),
                            "y_pct": round(max(0.0, min(95.0, (raw_pre["y"] / 720.0) * 100.0)), 1),
                            "w_pct": round(max(2.0, min(90.0, (raw_pre["width"] / 1280.0) * 100.0)), 1),
                            "h_pct": round(max(2.0, min(90.0, (raw_pre["height"] / 720.0) * 100.0)), 1),
                            "label": (action.selector or action.name or action.action_type)[:48],
                        }
            except Exception:
                pass

        try:
            action_status, used_selector = BrowserToolkit.execute_playwright_action(
                page=self.page,
                action=action,
                target_url=self.target_url,
            )
            if action.action_type in ("click", "fill", "press", "select", "scroll"):
                self._wait_for_dom_paint()

            elapsed = round((time.perf_counter() - started) * 1000, 2)
            current_url = str(getattr(self.page, "url", "") or live_url)
            shot_b64, target_box = PlaywrightRunner._highlight_and_capture(
                page=self.page,
                action=action,
                used_selector=used_selector,
                step_number=step_number,
                total_steps=max_steps,
                scenario_title=goal_title,
                target_url=current_url,
                status="passed",
                fallback_box=pre_box,
            )
            step_res = NavigationStepResult(
                step_number=step_number,
                action=action,
                status="passed",
                duration_ms=elapsed,
                current_url=current_url,
                http_status=action_status if action_status is not None else self.collector.last_http_status,
                console_errors=list(self.collector.console_errors),
                network_failures=list(self.collector.network_failures),
                screenshot_b64=shot_b64,
                target_box=target_box,
            )
            PlaywrightRunner.emit_step_telemetry(
                self.trace_manager,
                trace_id=trace_id,
                stream_id=self.stream_id,
                scenario_id=goal_id,
                scenario_title=goal_title,
                action=action,
                total_steps=max_steps,
                status="passed",
                duration_ms=elapsed,
                selector=used_selector or action.selector,
                current_url=current_url,
                screenshot_b64=shot_b64,
                target_box=target_box,
                execution_mode="playwright",
            )
            if self.stream_id:
                stream_mgr.update_stage(self.stream_id, "navigator", f"{goal_id} step {step_number}: {action.description[:60]}", stage_index=2)
            return step_res, None
        except Exception as exc:
            self.invalidate_dom_cache()
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            current_url = str(getattr(self.page, "url", "") or "")
            err_msg = str(exc)
            state, _, _ = self.observe(force_refresh=True)
            overlays = state.get("blocking_overlays") or []
            overlay_desc = (
                ", ".join(str(o.get("selector") or o.get("tag") or "overlay") for o in overlays)
                if overlays
                else None
            )
            shot_b64, target_box = PlaywrightRunner._highlight_and_capture(
                page=self.page,
                action=action,
                used_selector=action.selector,
                step_number=step_number,
                total_steps=max_steps,
                scenario_title=goal_title,
                target_url=current_url or "about:blank",
                status="failed",
                dom_snapshot=state.get("dom_excerpt"),
                error_message=err_msg,
                fallback_box=pre_box,
            )
            ev = self.collector.build_failure_evidence(
                failure_id=f"{goal_id}-STEP-{step_number}",
                scenario_id=goal_id,
                scenario_title=goal_title,
                step_number=step_number,
                current_url=current_url or "about:blank",
                error_message=err_msg,
                failed_selector=action.selector or BrowserToolkit.format_accessible_locator(action.locator_strategy, action.role, action.name),
                blocking_overlay=overlay_desc,
                dom_excerpt=state.get("dom_excerpt"),
                screenshot_b64=shot_b64,
                action_description=action.description,
                page=self.page,
            )
            step_res = NavigationStepResult(
                step_number=step_number,
                action=action,
                status="failed",
                duration_ms=elapsed,
                current_url=ev.current_url or current_url,
                error_message=err_msg,
                failed_selector=ev.failed_selector,
                http_status=ev.http_status,
                console_errors=list(ev.console_errors),
                network_failures=list(ev.network_failures),
                dom_excerpt=ev.dom_excerpt,
                screenshot_b64=shot_b64,
                target_box=target_box,
                blocking_overlay=ev.blocking_overlay,
            )
            PlaywrightRunner.emit_step_telemetry(
                self.trace_manager,
                trace_id=trace_id,
                stream_id=self.stream_id,
                scenario_id=goal_id,
                scenario_title=goal_title,
                action=action,
                total_steps=max_steps,
                status="failed",
                duration_ms=elapsed,
                selector=ev.failed_selector,
                current_url=ev.current_url or current_url,
                error_message=err_msg,
                screenshot_b64=shot_b64,
                target_box=target_box,
                execution_mode="playwright",
            )
            return step_res, ev.model_dump()

    def close(self) -> None:
        """Stop session tracing once, close single context/browser, and save video if enabled."""
        if self.config.record_traces and self.context is not None and hasattr(self.context, "tracing"):
            try:
                if self.config.save_traces_to_disk:
                    self.trace_manager.artifacts_dir.mkdir(parents=True, exist_ok=True)
                    self.context.tracing.stop(path=str(self.trace_manager.artifacts_dir / f"{self.run_id or 'run'}_session.zip"))
                else:
                    self.context.tracing.stop()
            except Exception as exc:
                logger.debug("[BrowserSession] Context tracing stop skipped: %s", exc)

        if self._owns_browser:
            raw_video = getattr(self.page, "video", None) if (self._use_video and self.page is not None) else None
            if self.context is not None:
                try:
                    self.context.close()
                except Exception:
                    pass
            if self._use_video and raw_video is not None:
                try:
                    src_path = Path(raw_video.path())
                    if src_path.exists():
                        dest_name = f"{self.run_id or 'run'}_session_{src_path.name}"
                        dest_path = self._videos_dir / dest_name
                        if src_path != dest_path:
                            shutil.move(str(src_path), str(dest_path))
                        self.video_url = f"/api/test-runs/videos/{dest_name}"
                except Exception as exc:
                    logger.debug("[BrowserSession] Video move skipped: %s", exc)
            if self._browser is not None:
                try:
                    self._browser.close()
                except Exception:
                    pass
            if self._pw is not None:
                try:
                    self._pw.stop()
                except Exception:
                    pass


class PlaywrightRunner:
    """Shared telemetry/screenshot helpers and standalone DOM snapshot capture."""

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
        stream_id: Optional[str] = None,
    ) -> dict[str, Optional[str]]:
        """Capture a live post-preflight DOM snapshot & summary using a single BrowserSession or httpx fallback."""
        stream_mgr = get_stream_manager()
        if not use_live_browser:
            html = fallback_html or (_DEMO_STAGING_HTML if "staging.store.nexus" in target_url else None)
            summary = BrowserToolkit.summarize_dom_for_agents(html) if html else None
            return {"dom_snapshot": html, "dom_summary": summary}

        if "staging.store.nexus" in target_url and not fallback_html:
            return {
                "dom_snapshot": _DEMO_STAGING_HTML,
                "dom_summary": BrowserToolkit.summarize_dom_for_agents(_DEMO_STAGING_HTML),
            }

        if _has_playwright_chromium():
            try:
                with BrowserSession(
                    target_url,
                    stream_id=stream_id,
                    config=PlaywrightRunConfig(headless=headless, use_live_browser=True, record_video=False, record_traces=False),
                ) as session:
                    live_state, summary, _ = session.observe(force_refresh=True)
                    return {
                        "dom_snapshot": str(live_state.get("dom_excerpt") or ""),
                        "dom_summary": summary,
                    }
            except Exception as exc:
                logger.warning("[PlaywrightRunner] Live browser snapshot fallback to HTTP fetch: %s", exc)
                if stream_id:
                    stream_mgr.push_log(stream_id, "DOM Inspector", f"Live snapshot HTTP fallback ({exc})", status="warning")

        try:
            import httpx

            resp = httpx.get(
                target_url,
                follow_redirects=True,
                timeout=8.0,
                headers={"User-Agent": "AgentSwarm-QA-Bot/1.0"},
            )
            if resp.text and "<" in resp.text:
                return {
                    "dom_snapshot": resp.text,
                    "dom_summary": BrowserToolkit.summarize_dom_for_agents(resp.text),
                }
        except Exception as exc:
            logger.warning("[PlaywrightRunner] httpx snapshot capture failed for %s: %s", target_url, exc)

        summary = BrowserToolkit.summarize_dom_for_agents(fallback_html) if fallback_html else None
        return {"dom_snapshot": fallback_html, "dom_summary": summary}

    @classmethod
    def _highlight_and_capture(
        cls,
        *,
        page: Any,
        action: Optional[BrowserAction],
        used_selector: Optional[str],
        step_number: int,
        total_steps: int,
        scenario_title: str,
        target_url: str,
        status: str,
        dom_snapshot: Optional[str] = None,
        error_message: Optional[str] = None,
        fallback_box: Optional[dict[str, Any]] = None,
    ) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        """Extract bounding box, highlight element briefly for screenshot, and return `(screenshot_b64, target_box)`."""
        target_box: Optional[dict[str, Any]] = fallback_box
        overlay_id = "agentswarm-live-highlight"

        if page is not None and action is not None and action.action_type not in ("goto", "scroll", "assert_url"):
            try:
                loc, _ = BrowserToolkit.build_playwright_locator(
                    page,
                    strategy=action.locator_strategy,
                    role=action.role,
                    name=action.name,
                    selector=used_selector or action.selector,
                )
                first_loc = loc.first if hasattr(loc, "first") else loc
                if hasattr(first_loc, "bounding_box"):
                    raw_box = first_loc.bounding_box(timeout=600)
                    if isinstance(raw_box, dict) and raw_box.get("width", 0) > 0:
                        target_box = {
                            "x_pct": round(max(0.0, min(95.0, (raw_box["x"] / 1280.0) * 100.0)), 1),
                            "y_pct": round(max(0.0, min(95.0, (raw_box["y"] / 720.0) * 100.0)), 1),
                            "w_pct": round(max(2.0, min(90.0, (raw_box["width"] / 1280.0) * 100.0)), 1),
                            "h_pct": round(max(2.0, min(90.0, (raw_box["height"] / 720.0) * 100.0)), 1),
                            "label": (used_selector or action.selector or action.name or "")[:48],
                        }
                        if hasattr(page, "evaluate"):
                            color = "#22c55e" if status == "passed" else "#ef4444"
                            page.evaluate(
                                """([box, color, id]) => {
                                    const old = document.getElementById(id);
                                    if (old) old.remove();
                                    const el = document.createElement('div');
                                    el.id = id;
                                    el.style.cssText = `position:fixed;left:${box.x}px;top:${box.y}px;width:${box.width}px;height:${box.height}px;border:3px solid ${color};background:rgba(34,197,94,0.14);border-radius:6px;z-index:2147483647;pointer-events:none;box-shadow:0 0 18px rgba(34,197,94,0.45);`;
                                    document.body.appendChild(el);
                                }""",
                                [raw_box, color, overlay_id],
                            )
            except Exception:
                pass

        screenshot_b64: Optional[str] = None
        if page is not None and hasattr(page, "screenshot"):
            try:
                raw = page.screenshot(type="jpeg", quality=78)
                if isinstance(raw, (bytes, bytearray)) and len(raw) > 0:
                    screenshot_b64 = "data:image/jpeg;base64," + base64.b64encode(bytes(raw)).decode("ascii")
            except Exception as exc:
                logger.debug("[PlaywrightRunner] Screenshot skipped: %s", exc)

        if page is not None and hasattr(page, "evaluate"):
            try:
                page.evaluate(f"() => {{ const el = document.getElementById('{overlay_id}'); if (el) el.remove(); }}")
            except Exception:
                pass

        if not screenshot_b64:
            screenshot_b64 = cls.build_synthetic_frame_b64(
                target_url=str(getattr(page, "url", "") or target_url),
                scenario_title=scenario_title,
                action=action,
                step_number=step_number,
                total_steps=total_steps,
                status=status,
                dom_snapshot=dom_snapshot,
                error_message=error_message,
            )
        return screenshot_b64, target_box

    @staticmethod
    def build_synthetic_frame_b64(
        *,
        target_url: str,
        scenario_title: str,
        action: Optional[BrowserAction],
        step_number: int,
        total_steps: int,
        status: str,
        dom_snapshot: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> str:
        """Render a fallback 1280x720 SVG frame when running in offline simulated mode."""
        accent = "#16a34a" if status == "passed" else ("#dc2626" if status == "failed" else "#737373")
        accent_bg = "#ecfdf5" if status == "passed" else ("#fef2f2" if status == "failed" else "#f5f5f5")
        safe_url = html_lib.escape((target_url or "")[:78])
        safe_title = html_lib.escape((scenario_title or "Autonomous QA Goal")[:64])
        act_desc = action.description if action else "Viewport check"
        safe_desc = html_lib.escape(act_desc[:76])
        err_banner = ""
        if error_message:
            err_banner = (
                '<rect x="90" y="548" width="1100" height="44" rx="8" fill="#fef2f2" stroke="#fca5a5" stroke-width="1.5"/>'
                f'<text x="110" y="575" font-family="monospace" font-size="13" fill="#991b1b">ERROR: {html_lib.escape(error_message[:95])}</text>'
            )
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
  <rect width="1280" height="720" fill="#fafaf9"/>
  <rect x="44" y="28" width="1192" height="664" rx="14" fill="#ffffff" stroke="#e4e4e7" stroke-width="2"/>
  <text x="86" y="88" font-family="sans-serif" font-size="24" font-weight="700" fill="#09090b">{safe_title}</text>
  <text x="86" y="120" font-family="monospace" font-size="14" fill="#52525b">{safe_url} — {safe_desc}</text>
  <rect x="950" y="54" width="250" height="36" rx="8" fill="{accent_bg}" stroke="{accent}"/>
  <text x="970" y="77" font-family="monospace" font-size="13" font-weight="700" fill="{accent}">SIMULATED STEP {step_number}/{max(total_steps, 1)} • {status.upper()}</text>
  <line x1="86" y1="146" x2="1194" y2="146" stroke="#e4e4e7" stroke-width="1.5"/>
  {err_banner}
</svg>'''
        return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")

    @staticmethod
    def emit_step_telemetry(
        trace_manager: TraceManager,
        *,
        trace_id: Optional[str],
        stream_id: Optional[str],
        scenario_id: str,
        scenario_title: str,
        action: BrowserAction,
        total_steps: int,
        status: str,
        duration_ms: float,
        selector: Optional[str],
        current_url: str,
        error_message: Optional[str] = None,
        screenshot_b64: Optional[str] = None,
        target_box: Optional[dict[str, Any]] = None,
        execution_mode: str = "playwright",
    ) -> None:
        if trace_id:
            trace_manager.record_step(
                trace_id=trace_id,
                step_number=action.step_number,
                action_type=action.action_type,
                status=status,  # type: ignore[arg-type]
                duration_ms=duration_ms,
                selector=selector,
                value=action.value,
                description=action.description,
                current_url=current_url,
                error_message=error_message,
                screenshot_b64=screenshot_b64,
                target_box=target_box,
            )
        if stream_id and status != "skipped":
            stream_mgr = get_stream_manager()
            stream_mgr.push_frame(
                stream_id,
                {
                    "scenario_id": scenario_id,
                    "scenario_title": scenario_title,
                    "step_number": action.step_number,
                    "total_steps": total_steps,
                    "action_type": action.action_type,
                    "selector": selector,
                    "value": action.value,
                    "description": action.description,
                    "status": status,
                    "duration_ms": duration_ms,
                    "current_url": current_url,
                    "error_message": error_message,
                    "screenshot_b64": screenshot_b64,
                    "target_box": target_box,
                    "execution_mode": execution_mode,
                },
            )
            target_desc = selector or action.value or action.description or action.action_type
            log_status = "success" if status == "passed" else "error"
            msg = f"{scenario_id} Step {action.step_number}: {action.action_type.upper()} {target_desc} -> {status.upper()} ({duration_ms:.0f}ms)"
            if error_message:
                msg += f" | {error_message[:100]}"
            stream_mgr.push_log(stream_id, "Navigator Agent", msg, status=log_status)
