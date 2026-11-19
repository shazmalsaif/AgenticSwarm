from __future__ import annotations

import base64
import concurrent.futures
import html as html_lib
import logging
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel

from .browser_tools import BrowserToolkit
from .evidence_collector import EvidenceCollector
from .trace_manager import TraceManager, get_stream_manager, get_trace_manager

if TYPE_CHECKING:
    from ..agent.navigator_agent import BrowserAction, NavigationRunResult, NavigationStepResult

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


def _resolve_browser_executable() -> Optional[str]:
    """Locate a local Chromium, Google Chrome, or Microsoft Edge executable for Playwright."""
    candidates: list[Path] = []
    env_path = os.getenv("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    if env_path and env_path != "0":
        candidates.append(Path(env_path))
    local_app = os.getenv("LOCALAPPDATA", "").strip()
    if local_app:
        candidates.append(Path(local_app) / "ms-playwright")
    candidates.append(Path.home() / ".cache" / "ms-playwright")
    candidates.append(Path.home() / "AppData" / "Local" / "ms-playwright")

    for base in candidates:
        if not base.exists():
            continue
        for sub in base.glob("chromium*"):
            for rel in (
                sub / "chrome-win" / "chrome.exe",
                sub / "chrome-win64" / "chrome.exe",
                sub / "chrome-linux" / "chrome",
                sub / "chrome-mac" / "Chromium.app" / "Contents" / "MacOS" / "Chromium",
            ):
                if rel.exists():
                    return str(rel)

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
    """Return True if Playwright Chromium, Chrome, or Edge is available on disk."""
    return _resolve_browser_executable() is not None


def _launch_chromium(playwright_instance: Any, headless: bool = True) -> Any:
    """Launch Playwright Chromium using local cache or installed Chrome/Edge binary."""
    exe_path = _resolve_browser_executable()
    launch_kwargs: dict[str, Any] = {
        "headless": headless,
        "args": ["--disable-blink-features=AutomationControlled"],
    }
    if exe_path:
        launch_kwargs["executable_path"] = exe_path
    return playwright_instance.chromium.launch(**launch_kwargs)


class PlaywrightRunConfig(BaseModel):
    """Runtime execution configuration for AgentSwarm browser sessions."""

    headless: bool = True
    use_live_browser: bool = False
    record_traces: bool = True
    record_video: bool = True
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
        timeout_ms: int = 15000,
        stream_id: Optional[str] = None,
    ) -> dict[str, Optional[str]]:
        """
        Capture HTML DOM snapshot, interactive summary, and an initial live browser screenshot.

        Launches installed Chrome/Edge via Playwright when `use_live_browser=True`; falls back to httpx.
        """
        stream_mgr = get_stream_manager()
        if not use_live_browser:
            html = fallback_html or (_DEMO_STAGING_HTML if "staging.store.nexus" in target_url else None)
            summary = BrowserToolkit.summarize_dom_for_agents(html) if html else None
            return {"dom_snapshot": html, "dom_summary": summary}

        if "staging.store.nexus" in target_url and not fallback_html:
            summary = BrowserToolkit.summarize_dom_for_agents(_DEMO_STAGING_HTML)
            if stream_id:
                stream_mgr.push_log(
                    stream_id,
                    "DOM Inspector",
                    "Loaded Nexus Store Staging demo DOM snapshot.",
                    status="info",
                )
            return {"dom_snapshot": _DEMO_STAGING_HTML, "dom_summary": summary}

        if _has_playwright_chromium():
            if stream_id:
                stream_mgr.push_log(
                    stream_id,
                    "DOM Inspector",
                    f"Launching headless browser to open {target_url}...",
                    status="info",
                )

            def _pw_fetch() -> tuple[Optional[str], Optional[str]]:
                from playwright.sync_api import sync_playwright

                with sync_playwright() as p:
                    browser = _launch_chromium(p, headless=headless)
                    try:
                        context = browser.new_context(
                            viewport={"width": 1280, "height": 720},
                            user_agent=(
                                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                            ),
                        )
                        page = context.new_page()
                        page.goto(target_url, wait_until="domcontentloaded", timeout=min(timeout_ms, 12000))
                        try:
                            page.wait_for_timeout(1200)
                        except Exception:
                            pass
                        html_text = page.content()
                        shot_b64: Optional[str] = None
                        try:
                            raw = page.screenshot(type="jpeg", quality=80)
                            if isinstance(raw, (bytes, bytearray)) and len(raw) > 0:
                                shot_b64 = "data:image/jpeg;base64," + base64.b64encode(bytes(raw)).decode("ascii")
                        except Exception:
                            pass
                        return html_text, shot_b64
                    finally:
                        browser.close()

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(_pw_fetch)
                    html, initial_shot = fut.result(timeout=20.0)
                    if html:
                        summary = BrowserToolkit.summarize_dom_for_agents(html)
                        elements = BrowserToolkit.extract_interactive_elements(html)
                        if stream_id:
                            if initial_shot:
                                stream_mgr.push_frame(
                                    stream_id,
                                    {
                                        "scenario_id": "LIVE-DOM",
                                        "scenario_title": "Initial Live Browser Load",
                                        "step_number": 1,
                                        "total_steps": 1,
                                        "action_type": "goto",
                                        "selector": None,
                                        "value": target_url,
                                        "description": f"Loaded live website {target_url}",
                                        "status": "passed",
                                        "duration_ms": 1200.0,
                                        "current_url": target_url,
                                        "screenshot_b64": initial_shot,
                                        "target_box": None,
                                        "execution_mode": "playwright",
                                    },
                                )
                            stream_mgr.push_log(
                                stream_id,
                                "DOM Inspector",
                                f"Captured live rendered DOM from {target_url} ({len(elements)} interactive elements found).",
                                status="success",
                            )
                        return {"dom_snapshot": html, "dom_summary": summary}
            except Exception as exc:
                logger.warning("[PlaywrightRunner] Playwright snapshot timed out/failed for %s: %s", target_url, exc)
                if stream_id:
                    stream_mgr.push_log(
                        stream_id,
                        "DOM Inspector",
                        f"Headless browser snapshot fallback to HTTP fetch ({exc})",
                        status="warning",
                    )

        # Fast httpx live HTML fetch fallback
        try:
            import httpx

            resp = httpx.get(
                target_url,
                follow_redirects=True,
                timeout=6.0,
                headers={"User-Agent": "AgentSwarm-QA-Bot/1.0"},
            )
            if resp.text and "<" in resp.text:
                html = resp.text
                summary = BrowserToolkit.summarize_dom_for_agents(html)
                if stream_id:
                    elements = BrowserToolkit.extract_interactive_elements(html)
                    stream_mgr.push_log(
                        stream_id,
                        "DOM Inspector",
                        f"Fetched HTML via HTTP ({len(elements)} interactive elements extracted).",
                        status="info",
                    )
                return {"dom_snapshot": html, "dom_summary": summary}
        except Exception as exc:
            logger.warning("[PlaywrightRunner] httpx snapshot capture failed for %s: %s", target_url, exc)

        summary = BrowserToolkit.summarize_dom_for_agents(fallback_html) if fallback_html else None
        return {"dom_snapshot": fallback_html, "dom_summary": summary}

    @staticmethod
    def _compute_target_box(
        action: BrowserAction,
        step_number: int,
        total_steps: int,
        dom_snapshot: Optional[str] = None,
    ) -> dict[str, Any]:
        """Compute deterministic viewport percentage coordinates for the cursor & highlight overlay."""
        action_type = action.action_type
        if action_type == "goto":
            return {"x_pct": 38.0, "y_pct": 8.0, "w_pct": 44.0, "h_pct": 6.0, "label": "URL Bar"}
        if action_type == "assert_url":
            return {"x_pct": 42.0, "y_pct": 8.0, "w_pct": 36.0, "h_pct": 6.0, "label": "URL Assertion"}

        elements = BrowserToolkit.extract_interactive_elements(dom_snapshot) if dom_snapshot and "<" in dom_snapshot else []
        matched_idx = None
        sel_lower = (action.selector or "").lower()
        for idx, el in enumerate(elements[:9]):
            hay = f"{el.recommended_locator} {el.element_id} {el.accessible_name} {el.tag}".lower()
            if sel_lower and (sel_lower in hay or (el.element_id and el.element_id.lower() in sel_lower)):
                matched_idx = idx
                break

        slot = matched_idx if matched_idx is not None else ((step_number - 1) % 6)
        if action_type == "fill":
            x_pct = 24.0 + (slot % 2) * 28.0
            y_pct = 36.0 + (slot // 2) * 14.0
            w_pct, h_pct = 24.0, 7.5
        elif action_type == "click":
            x_pct = 28.0 + (slot % 3) * 22.0
            y_pct = 52.0 + (slot // 3) * 14.0
            w_pct, h_pct = 18.0, 8.0
        else:
            x_pct = 20.0 + (slot % 2) * 25.0
            y_pct = 28.0 + (slot // 2) * 14.0
            w_pct, h_pct = 32.0, 12.0

        return {
            "x_pct": round(min(max(x_pct, 8.0), 82.0), 1),
            "y_pct": round(min(max(y_pct, 14.0), 82.0), 1),
            "w_pct": w_pct,
            "h_pct": h_pct,
            "label": (action.selector or action.description or action_type)[:48],
        }

    @classmethod
    def _build_synthetic_frame_b64(
        cls,
        *,
        target_url: str,
        scenario_title: str,
        action: BrowserAction,
        step_number: int,
        total_steps: int,
        status: str,
        dom_snapshot: Optional[str] = None,
        error_message: Optional[str] = None,
        target_box: Optional[dict[str, Any]] = None,
    ) -> str:
        """Render a crisp 1280x720 visual SVG browser viewport frame for live streaming & video playback."""
        box = target_box or cls._compute_target_box(action, step_number, total_steps, dom_snapshot)
        elements = BrowserToolkit.extract_interactive_elements(dom_snapshot) if dom_snapshot and "<" in dom_snapshot else []

        bx = box["x_pct"] * 12.8
        by = box["y_pct"] * 7.2
        bw = box["w_pct"] * 12.8
        bh = box["h_pct"] * 7.2

        accent = "#16a34a" if status == "passed" else ("#dc2626" if status == "failed" else "#737373")
        accent_bg = "#ecfdf5" if status == "passed" else ("#fef2f2" if status == "failed" else "#f5f5f5")
        safe_url = html_lib.escape((target_url or "")[:78])
        safe_title = html_lib.escape((scenario_title or "Autonomous QA Session")[:64])
        safe_action = html_lib.escape(action.action_type.upper())
        safe_target = html_lib.escape((action.selector or action.value or action.description or "")[:58])
        safe_desc = html_lib.escape((action.description or f"{action.action_type} step")[:76])

        # Render discovered interactive elements inside the virtual browser page
        el_svgs: list[str] = []
        for idx, el in enumerate(elements[:6]):
            col = idx % 3
            row = idx // 3
            ex = 110 + col * 360
            ey = 250 + row * 125
            ew = 320
            eh = 64
            label = html_lib.escape((el.accessible_name or el.text or el.element_id or el.tag)[:28])
            tag_badge = html_lib.escape(el.tag.upper())
            if el.tag in ("input", "textarea", "select"):
                el_svgs.append(
                    f'<rect x="{ex}" y="{ey}" width="{ew}" height="{eh}" rx="8" fill="#ffffff" stroke="#d4d4d8" stroke-width="1.5"/>'
                    f'<text x="{ex + 14}" y="{ey + 24}" font-family="monospace" font-size="11" fill="#71717a">{tag_badge}</text>'
                    f'<text x="{ex + 14}" y="{ey + 45}" font-family="sans-serif" font-size="15" fill="#27272a">{label}</text>'
                )
            else:
                fill_col = "#18181b" if el.tag == "button" else "#f4f4f5"
                txt_col = "#ffffff" if el.tag == "button" else "#18181b"
                el_svgs.append(
                    f'<rect x="{ex}" y="{ey}" width="{ew}" height="{eh}" rx="8" fill="{fill_col}" stroke="#d4d4d8" stroke-width="1.5"/>'
                    f'<text x="{ex + 16}" y="{ey + 38}" font-family="sans-serif" font-size="15" font-weight="600" fill="{txt_col}">{label}</text>'
                )

        if not el_svgs:
            el_svgs.append(
                '<rect x="110" y="250" width="500" height="72" rx="10" fill="#ffffff" stroke="#e4e4e7" stroke-width="1.5"/>'
                '<text x="132" y="293" font-family="sans-serif" font-size="16" fill="#27272a">Main Application Viewport</text>'
                '<rect x="640" y="250" width="280" height="72" rx="10" fill="#18181b"/>'
                '<text x="680" y="293" font-family="sans-serif" font-size="16" font-weight="600" fill="#ffffff">Interactive Target Control</text>'
            )

        err_banner = ""
        if error_message:
            safe_err = html_lib.escape(error_message[:95])
            err_banner = (
                '<rect x="90" y="548" width="1100" height="44" rx="8" fill="#fef2f2" stroke="#fca5a5" stroke-width="1.5"/>'
                f'<text x="110" y="575" font-family="monospace" font-size="13" fill="#991b1b">ERROR: {safe_err}</text>'
            )

        cx = bx + bw * 0.55
        cy = by + bh * 0.55
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
  <rect width="1280" height="720" fill="#fafaf9"/>
  <!-- Simulated Canvas Header -->
  <rect x="44" y="28" width="1192" height="664" rx="14" fill="#ffffff" stroke="#e4e4e7" stroke-width="2"/>
  <text x="86" y="88" font-family="sans-serif" font-size="24" font-weight="700" fill="#09090b">{safe_title}</text>
  <text x="86" y="120" font-family="monospace" font-size="14" fill="#52525b">{safe_url} — {safe_desc}</text>
  <rect x="970" y="54" width="230" height="36" rx="8" fill="{accent_bg}" stroke="{accent}"/>
  <text x="990" y="77" font-family="monospace" font-size="13" font-weight="700" fill="{accent}">SIMULATED STEP {step_number}/{max(total_steps, 1)} • {status.upper()}</text>
  <line x1="86" y1="146" x2="1194" y2="146" stroke="#e4e4e7" stroke-width="1.5"/>

  {''.join(el_svgs)}
  {err_banner}
</svg>'''
        encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        return f"data:image/svg+xml;base64,{encoded}"

    @classmethod
    def _highlight_and_capture(
        cls,
        *,
        page: Any,
        action: BrowserAction,
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
        """Extract bounding box, inject a clean highlight border on live Playwright page, and capture JPEG frame."""
        target_box: Optional[dict[str, Any]] = fallback_box
        sel = used_selector or action.selector
        overlay_id = "agentswarm-live-highlight"

        if page is not None and sel and action.action_type not in ("goto", "assert_url"):
            try:
                loc = BrowserToolkit._to_playwright_locator(page, sel)
                first_loc = loc.first if hasattr(loc, "first") else loc
                if hasattr(first_loc, "bounding_box"):
                    raw_box = first_loc.bounding_box(timeout=800)
                    if isinstance(raw_box, dict) and raw_box.get("width", 0) > 0:
                        target_box = {
                            "x_pct": round(max(0.0, min(95.0, (raw_box["x"] / 1280.0) * 100.0)), 1),
                            "y_pct": round(max(0.0, min(95.0, (raw_box["y"] / 720.0) * 100.0)), 1),
                            "w_pct": round(max(2.0, min(90.0, (raw_box["width"] / 1280.0) * 100.0)), 1),
                            "h_pct": round(max(2.0, min(90.0, (raw_box["height"] / 720.0) * 100.0)), 1),
                            "label": sel[:48],
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
                raw = page.screenshot(type="jpeg", quality=80)
                if isinstance(raw, (bytes, bytearray)) and len(raw) > 0:
                    screenshot_b64 = "data:image/jpeg;base64," + base64.b64encode(bytes(raw)).decode("ascii")
            except Exception as exc:
                logger.debug("[PlaywrightRunner] Live screenshot capture skipped: %s", exc)

        if page is not None and hasattr(page, "evaluate"):
            try:
                page.evaluate(f"() => {{ const el = document.getElementById('{overlay_id}'); if (el) el.remove(); }}")
            except Exception:
                pass

        if not screenshot_b64:
            if target_box is None:
                target_box = cls._compute_target_box(action, step_number, total_steps, dom_snapshot)
            screenshot_b64 = cls._build_synthetic_frame_b64(
                target_url=str(getattr(page, "url", "") or target_url),
                scenario_title=scenario_title,
                action=action,
                step_number=step_number,
                total_steps=total_steps,
                status=status,
                dom_snapshot=dom_snapshot,
                error_message=error_message,
                target_box=target_box,
            )
        return screenshot_b64, target_box

    def execute_actions(
        self,
        scenario_id: str,
        scenario_title: str,
        target_url: str,
        actions: list[BrowserAction | dict[str, Any]],
        dom_snapshot: Optional[str] = None,
        page: Optional[Any] = None,
        run_id: Optional[str] = None,
        stream_id: Optional[str] = None,
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
                stream_id=stream_id,
                cfg=cfg,
            )

        if cfg.use_live_browser and _has_playwright_chromium() and "staging.store.nexus" not in target_url:
            try:
                from playwright.sync_api import sync_playwright

                def _run_pw() -> NavigationRunResult:
                    videos_dir = self.trace_manager.artifacts_dir / "videos"
                    if cfg.record_video:
                        videos_dir.mkdir(parents=True, exist_ok=True)
                    with sync_playwright() as p:
                        browser = _launch_chromium(p, headless=cfg.headless)
                        try:
                            ctx_kwargs: dict[str, Any] = {
                                "viewport": {"width": cfg.viewport_width, "height": cfg.viewport_height},
                                "user_agent": (
                                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                                ),
                            }
                            if cfg.record_video:
                                ctx_kwargs["record_video_dir"] = str(videos_dir)
                                ctx_kwargs["record_video_size"] = {
                                    "width": cfg.viewport_width,
                                    "height": cfg.viewport_height,
                                }
                            context = browser.new_context(**ctx_kwargs)
                            live_page = context.new_page()
                            result = self._execute_on_page(
                                scenario_id=scenario_id,
                                scenario_title=scenario_title,
                                target_url=target_url,
                                actions=coerced_actions,
                                page=live_page,
                                context=context,
                                dom_snapshot=dom_snapshot,
                                run_id=run_id,
                                stream_id=stream_id,
                                cfg=cfg,
                            )
                            raw_video = getattr(live_page, "video", None)
                            context.close()
                            if cfg.record_video and raw_video is not None:
                                try:
                                    src_path = Path(raw_video.path())
                                    if src_path.exists():
                                        dest_name = f"{run_id or 'run'}_{scenario_id}_{src_path.name}"
                                        dest_path = videos_dir / dest_name
                                        if src_path != dest_path:
                                            shutil.move(str(src_path), str(dest_path))
                                        video_url = f"/api/test-runs/videos/{dest_name}"
                                        result.video_url = video_url
                                        if result.trace_id:
                                            self.trace_manager.stop_trace(
                                                trace_id=result.trace_id,
                                                overall_status=result.overall_status,
                                                video_url=video_url,
                                            )
                                except Exception as v_exc:
                                    logger.debug("[PlaywrightRunner] Video save skipped: %s", v_exc)
                            return result
                        finally:
                            browser.close()

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(_run_pw).result(timeout=60.0)
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
            stream_id=stream_id,
            cfg=cfg,
        )

    def _emit_step_telemetry(
        self,
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
        execution_mode: str = "simulated",
    ) -> None:
        if trace_id:
            self.trace_manager.record_step(
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
            msg = f"{scenario_id} Step {action.step_number}/{total_steps}: {action.action_type.upper()} {target_desc} -> {status.upper()} ({duration_ms:.0f}ms)"
            if error_message:
                msg += f" | {error_message[:100]}"
            stream_mgr.push_log(stream_id, "Navigator Agent", msg, status=log_status)

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
        stream_id: Optional[str] = None,
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
        total_steps = len(actions)

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
                self._emit_step_telemetry(
                    trace_id=trace_id,
                    stream_id=stream_id,
                    scenario_id=scenario_id,
                    scenario_title=scenario_title,
                    action=action,
                    total_steps=total_steps,
                    status="skipped",
                    duration_ms=0.0,
                    selector=action.selector,
                    current_url=current_url,
                    execution_mode="playwright",
                )
                continue

            started = time.perf_counter()
            pre_box: Optional[dict[str, Any]] = None
            if action.selector and action.action_type not in ("goto", "assert_url"):
                try:
                    pre_loc = BrowserToolkit._to_playwright_locator(page, action.selector)
                    first_pre = pre_loc.first if hasattr(pre_loc, "first") else pre_loc
                    if hasattr(first_pre, "scroll_into_view_if_needed"):
                        first_pre.scroll_into_view_if_needed(timeout=800)
                    if hasattr(first_pre, "bounding_box"):
                        raw_pre = first_pre.bounding_box(timeout=800)
                        if isinstance(raw_pre, dict) and raw_pre.get("width", 0) > 0:
                            pre_box = {
                                "x_pct": round(max(0.0, min(95.0, (raw_pre["x"] / 1280.0) * 100.0)), 1),
                                "y_pct": round(max(0.0, min(95.0, (raw_pre["y"] / 720.0) * 100.0)), 1),
                                "w_pct": round(max(2.0, min(90.0, (raw_pre["width"] / 1280.0) * 100.0)), 1),
                                "h_pct": round(max(2.0, min(90.0, (raw_pre["height"] / 720.0) * 100.0)), 1),
                                "label": action.selector[:48],
                            }
                except Exception:
                    pass

            exec_action = (
                action.model_copy(update={"timeout_ms": max(action.timeout_ms, 12000)})
                if action.action_type == "goto"
                else action
            )
            try:
                action_status, used_selector = BrowserToolkit.execute_playwright_action(
                    page=page,
                    action=exec_action,
                    target_url=target_url,
                )
                if action.action_type in ("goto", "click", "fill") and hasattr(page, "wait_for_timeout"):
                    try:
                        page.wait_for_timeout(500)
                    except Exception:
                        pass
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
                shot_b64, target_box = self._highlight_and_capture(
                    page=page,
                    action=action,
                    used_selector=used_selector,
                    step_number=action.step_number,
                    total_steps=total_steps,
                    scenario_title=scenario_title,
                    target_url=current_url,
                    status="passed",
                    dom_snapshot=dom_snapshot,
                    fallback_box=pre_box,
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
                        screenshot_b64=shot_b64,
                        target_box=target_box,
                    )
                )
                self._emit_step_telemetry(
                    trace_id=trace_id,
                    stream_id=stream_id,
                    scenario_id=scenario_id,
                    scenario_title=scenario_title,
                    action=action,
                    total_steps=total_steps,
                    status="passed",
                    duration_ms=elapsed,
                    selector=used_selector or action.selector,
                    current_url=current_url,
                    screenshot_b64=shot_b64,
                    target_box=target_box,
                    execution_mode="playwright",
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
                shot_b64, target_box = self._highlight_and_capture(
                    page=page,
                    action=action,
                    used_selector=action.selector,
                    step_number=action.step_number,
                    total_steps=total_steps,
                    scenario_title=scenario_title,
                    target_url=current_url,
                    status="failed",
                    dom_snapshot=dom_snapshot,
                    error_message=err_msg,
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
                        screenshot_b64=shot_b64,
                        target_box=target_box,
                    )
                )
                failures.append(evidence.model_dump())
                self._emit_step_telemetry(
                    trace_id=trace_id,
                    stream_id=stream_id,
                    scenario_id=scenario_id,
                    scenario_title=scenario_title,
                    action=action,
                    total_steps=total_steps,
                    status="failed",
                    duration_ms=elapsed,
                    selector=action.selector,
                    current_url=current_url,
                    error_message=err_msg,
                    screenshot_b64=shot_b64,
                    target_box=target_box,
                    execution_mode="playwright",
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
        stream_id: Optional[str] = None,
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
        total_steps = len(actions)

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
                self._emit_step_telemetry(
                    trace_id=trace_id,
                    stream_id=stream_id,
                    scenario_id=scenario_id,
                    scenario_title=scenario_title,
                    action=action,
                    total_steps=total_steps,
                    status="skipped",
                    duration_ms=0.0,
                    selector=action.selector,
                    current_url=current_url,
                    execution_mode="simulated",
                )
                continue

            if action.action_type == "goto":
                if action.value:
                    current_url = action.value
                collector.ingest_simulated_telemetry(http_status=200)
                t_box = self._compute_target_box(action, action.step_number, total_steps, dom_snapshot)
                shot_b64 = self._build_synthetic_frame_b64(
                    target_url=current_url,
                    scenario_title=scenario_title,
                    action=action,
                    step_number=action.step_number,
                    total_steps=total_steps,
                    status="passed",
                    dom_snapshot=dom_snapshot,
                    target_box=t_box,
                )
                step_results.append(
                    NavigationStepResult(
                        step_number=action.step_number,
                        action=action,
                        status="passed",
                        duration_ms=42.0,
                        current_url=current_url,
                        http_status=200,
                        screenshot_b64=shot_b64,
                        target_box=t_box,
                    )
                )
                self._emit_step_telemetry(
                    trace_id=trace_id,
                    stream_id=stream_id,
                    scenario_id=scenario_id,
                    scenario_title=scenario_title,
                    action=action,
                    total_steps=total_steps,
                    status="passed",
                    duration_ms=42.0,
                    selector=action.selector,
                    current_url=current_url,
                    screenshot_b64=shot_b64,
                    target_box=t_box,
                    execution_mode="simulated",
                )
                if stream_id:
                    time.sleep(0.05)
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
                    t_box = self._compute_target_box(action, action.step_number, total_steps, dom_snapshot)
                    shot_b64 = self._build_synthetic_frame_b64(
                        target_url=current_url,
                        scenario_title=scenario_title,
                        action=action,
                        step_number=action.step_number,
                        total_steps=total_steps,
                        status="failed",
                        dom_snapshot=dom_snapshot,
                        error_message=err_msg,
                        target_box=t_box,
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
                            screenshot_b64=shot_b64,
                            target_box=t_box,
                        )
                    )
                    failures.append(evidence.model_dump())
                    self._emit_step_telemetry(
                        trace_id=trace_id,
                        stream_id=stream_id,
                        scenario_id=scenario_id,
                        scenario_title=scenario_title,
                        action=action,
                        total_steps=total_steps,
                        status="failed",
                        duration_ms=15.0,
                        selector=action.selector,
                        current_url=current_url,
                        error_message=err_msg,
                        screenshot_b64=shot_b64,
                        target_box=t_box,
                        execution_mode="simulated",
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

            t_box = self._compute_target_box(action, action.step_number, total_steps, dom_snapshot)
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
                shot_b64 = self._build_synthetic_frame_b64(
                    target_url=current_url,
                    scenario_title=scenario_title,
                    action=action,
                    step_number=action.step_number,
                    total_steps=total_steps,
                    status="failed",
                    dom_snapshot=dom_snapshot,
                    error_message=err_msg,
                    target_box=t_box,
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
                        screenshot_b64=shot_b64,
                        target_box=t_box,
                    )
                )
                failures.append(evidence.model_dump())
                self._emit_step_telemetry(
                    trace_id=trace_id,
                    stream_id=stream_id,
                    scenario_id=scenario_id,
                    scenario_title=scenario_title,
                    action=action,
                    total_steps=total_steps,
                    status="failed",
                    duration_ms=float(action.timeout_ms),
                    selector=action.selector,
                    current_url=current_url,
                    error_message=err_msg,
                    screenshot_b64=shot_b64,
                    target_box=t_box,
                    execution_mode="simulated",
                )
            else:
                shot_b64 = self._build_synthetic_frame_b64(
                    target_url=current_url,
                    scenario_title=scenario_title,
                    action=action,
                    step_number=action.step_number,
                    total_steps=total_steps,
                    status="passed",
                    dom_snapshot=dom_snapshot,
                    target_box=t_box,
                )
                step_results.append(
                    NavigationStepResult(
                        step_number=action.step_number,
                        action=action,
                        status="passed",
                        duration_ms=42.0,
                        current_url=current_url,
                        used_fallback_selector=used_fallback,
                        screenshot_b64=shot_b64,
                        target_box=t_box,
                    )
                )
                self._emit_step_telemetry(
                    trace_id=trace_id,
                    stream_id=stream_id,
                    scenario_id=scenario_id,
                    scenario_title=scenario_title,
                    action=action,
                    total_steps=total_steps,
                    status="passed",
                    duration_ms=42.0,
                    selector=used_fallback or action.selector,
                    current_url=current_url,
                    screenshot_b64=shot_b64,
                    target_box=t_box,
                    execution_mode="simulated",
                )
            if stream_id:
                time.sleep(0.05)

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
