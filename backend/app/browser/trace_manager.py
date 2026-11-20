from __future__ import annotations

import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TraceEvent(BaseModel):
    """Single browser step event recorded inside an execution trace."""

    step_number: int
    action_type: str
    selector: Optional[str] = None
    value: Optional[str] = None
    description: str = ""
    status: Literal["passed", "failed", "skipped"]
    duration_ms: float = 0.0
    current_url: str = ""
    error_message: Optional[str] = None
    screenshot_b64: Optional[str] = None
    target_box: Optional[dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TraceArtifact(BaseModel):
    """Trace timeline and optional Playwright trace.zip reference for one scenario run."""

    trace_id: str
    run_id: Optional[str] = None
    scenario_id: str
    scenario_title: str
    target_url: str
    execution_mode: Literal["playwright", "simulated"] = "simulated"
    trace_zip_path: Optional[str] = None
    video_url: Optional[str] = None
    events: list[TraceEvent] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    overall_status: Literal["passed", "failed", "stuck", "running"] = "running"


class TraceManager:
    """Thread-safe Playwright and simulated step trace recorder for AgentSwarm."""

    def __init__(self, artifacts_dir: Optional[Path | str] = None) -> None:
        self._lock = Lock()
        self._traces: dict[str, TraceArtifact] = {}
        self.artifacts_dir: Path = (
            Path(artifacts_dir)
            if artifacts_dir is not None
            else Path(tempfile.gettempdir()) / "agentswarm_traces"
        )

    def start_trace(
        self,
        scenario_id: str,
        scenario_title: str,
        target_url: str,
        run_id: Optional[str] = None,
        context: Optional[Any] = None,
        execution_mode: Literal["playwright", "simulated"] = "simulated",
    ) -> TraceArtifact:
        """Create a new trace artifact and optionally start Playwright context tracing."""
        trace_id = f"trc-{uuid4().hex[:10]}"
        if context is not None and hasattr(context, "tracing"):
            try:
                context.tracing.start(screenshots=True, snapshots=True, sources=True)
            except Exception as exc:
                logger.warning("[TraceManager] Unable to start Playwright context tracing: %s", exc)

        artifact = TraceArtifact(
            trace_id=trace_id,
            run_id=run_id,
            scenario_id=scenario_id,
            scenario_title=scenario_title,
            target_url=target_url,
            execution_mode=execution_mode,
            started_at=datetime.now(timezone.utc),
            overall_status="running",
        )
        with self._lock:
            self._traces[trace_id] = artifact
            return artifact.model_copy(deep=True)

    def record_step(
        self,
        trace_id: str,
        step_number: int,
        action_type: str,
        status: Literal["passed", "failed", "skipped"],
        duration_ms: float = 0.0,
        selector: Optional[str] = None,
        value: Optional[str] = None,
        description: str = "",
        current_url: str = "",
        error_message: Optional[str] = None,
        screenshot_b64: Optional[str] = None,
        target_box: Optional[dict[str, Any]] = None,
    ) -> Optional[TraceEvent]:
        """Append a step event to an existing trace artifact."""
        event = TraceEvent(
            step_number=step_number,
            action_type=action_type,
            selector=selector,
            value=value,
            description=description,
            status=status,
            duration_ms=duration_ms,
            current_url=current_url,
            error_message=error_message,
            screenshot_b64=screenshot_b64,
            target_box=target_box,
            timestamp=datetime.now(timezone.utc),
        )
        with self._lock:
            artifact = self._traces.get(trace_id)
            if artifact is None:
                return None
            artifact.events.append(event)
            return event.model_copy(deep=True)

    def stop_trace(
        self,
        trace_id: str,
        overall_status: Literal["passed", "failed", "stuck"] = "passed",
        context: Optional[Any] = None,
        save_to_disk: bool = False,
        video_url: Optional[str] = None,
    ) -> Optional[TraceArtifact]:
        """Finalize an active trace and optionally flush Playwright trace.zip to disk."""
        zip_path: Optional[str] = None
        if context is not None and hasattr(context, "tracing"):
            try:
                if save_to_disk:
                    self.artifacts_dir.mkdir(parents=True, exist_ok=True)
                    zip_path = str(self.artifacts_dir / f"{trace_id}.zip")
                    context.tracing.stop(path=zip_path)
                else:
                    context.tracing.stop()
            except Exception as exc:
                logger.warning("[TraceManager] Unable to stop Playwright context tracing cleanly: %s", exc)
                zip_path = None

        with self._lock:
            artifact = self._traces.get(trace_id)
            if artifact is None:
                return None
            artifact.completed_at = datetime.now(timezone.utc)
            artifact.overall_status = overall_status
            if zip_path:
                artifact.trace_zip_path = zip_path
            if video_url:
                artifact.video_url = video_url
            return artifact.model_copy(deep=True)

    def get_trace(self, trace_id: str) -> Optional[TraceArtifact]:
        """Retrieve a trace artifact by trace_id."""
        with self._lock:
            artifact = self._traces.get(trace_id)
            return artifact.model_copy(deep=True) if artifact is not None else None

    def list_traces(
        self,
        run_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
    ) -> list[TraceArtifact]:
        """List recorded trace artifacts, optionally filtered by run_id or scenario_id."""
        with self._lock:
            items = [item.model_copy(deep=True) for item in self._traces.values()]
        if run_id is not None:
            items = [item for item in items if item.run_id == run_id]
        if scenario_id is not None:
            items = [item for item in items if item.scenario_id == scenario_id]
        return sorted(items, key=lambda item: item.started_at)

    def clear(self) -> None:
        """Clear all stored trace artifacts."""
        with self._lock:
            self._traces.clear()


_DEFAULT_TRACE_MANAGER = TraceManager()


def get_trace_manager() -> TraceManager:
    """Return the shared AgentSwarm TraceManager singleton."""
    return _DEFAULT_TRACE_MANAGER


class LiveStreamManager:
    """Thread-safe real-time screenshot frame & agent stage stream broker for SSE and polling."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._streams: dict[str, dict[str, Any]] = {}

    def init_stream(
        self,
        stream_id: str,
        target_url: str = "",
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> dict[str, Any]:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            existing = self._streams.get(stream_id)
            state: dict[str, Any] = {
                "stream_id": stream_id,
                "run_id": run_id or (existing or {}).get("run_id"),
                "target_url": target_url or (existing or {}).get("target_url", ""),
                "provider": provider or (existing or {}).get("provider"),
                "model_name": model_name or (existing or {}).get("model_name"),
                "status": "running",
                "current_stage": "planner",
                "stage_index": 0,
                "stage_message": f"Opening {target_url or 'target URL'} in live browser & inspecting DOM...",
                "frames": list((existing or {}).get("frames") or []),
                "logs": list((existing or {}).get("logs") or []),
                "partial_results": dict((existing or {}).get("partial_results") or {}),
                "events": list((existing or {}).get("events") or []),
                "video_urls": list((existing or {}).get("video_urls") or []),
                "started_at": (existing or {}).get("started_at") or now_iso,
                "updated_at": now_iso,
            }
            if not state["events"]:
                state["events"].append(
                    {
                        "type": "stage",
                        "data": {
                            "stage": "planner",
                            "stage_index": 0,
                            "message": state["stage_message"],
                            "timestamp": now_iso,
                        },
                    }
                )
            self._streams[stream_id] = state
            return dict(state)

    def _ensure_stream_locked(self, stream_id: str, now_iso: str) -> dict[str, Any]:
        state = self._streams.get(stream_id)
        if state is None:
            state = {
                "stream_id": stream_id,
                "run_id": None,
                "target_url": "",
                "provider": None,
                "model_name": None,
                "status": "running",
                "current_stage": "planner",
                "stage_index": 0,
                "stage_message": "Running AgentSwarm...",
                "frames": [],
                "logs": [],
                "partial_results": {},
                "events": [],
                "video_urls": [],
                "started_at": now_iso,
                "updated_at": now_iso,
            }
            self._streams[stream_id] = state
        return state

    def update_stage(
        self,
        stream_id: Optional[str],
        stage: str,
        message: str,
        stage_index: int = 0,
    ) -> None:
        if not stream_id:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            state = self._ensure_stream_locked(stream_id, now_iso)
            state["current_stage"] = stage
            state["stage_index"] = stage_index
            state["stage_message"] = message
            state["updated_at"] = now_iso
            state["events"].append(
                {
                    "type": "stage",
                    "data": {
                        "stage": stage,
                        "stage_index": stage_index,
                        "message": message,
                        "timestamp": now_iso,
                    },
                }
            )

    def push_log(
        self,
        stream_id: Optional[str],
        agent: str,
        message: str,
        status: str = "info",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        if not stream_id:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            state = self._ensure_stream_locked(stream_id, now_iso)
            entry = {
                "index": len(state["logs"]),
                "agent": agent,
                "message": message,
                "status": status,
                "details": details or {},
                "timestamp": now_iso,
            }
            state["logs"].append(entry)
            state["updated_at"] = now_iso
            state["events"].append({"type": "agent_log", "data": entry})

    def update_partial(
        self,
        stream_id: Optional[str],
        key: str,
        value: Any,
    ) -> None:
        if not stream_id:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            state = self._ensure_stream_locked(stream_id, now_iso)
            state["partial_results"][key] = value
            state["updated_at"] = now_iso
            state["events"].append(
                {
                    "type": "partial",
                    "data": {
                        "key": key,
                        "value": value,
                        "timestamp": now_iso,
                    },
                }
            )

    def push_frame(self, stream_id: Optional[str], frame: dict[str, Any]) -> None:
        if not stream_id:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            state = self._ensure_stream_locked(stream_id, now_iso)
            enriched = {
                **frame,
                "frame_index": len(state["frames"]),
                "timestamp": frame.get("timestamp") or now_iso,
            }
            state["frames"].append(enriched)
            state["updated_at"] = now_iso
            state["events"].append({"type": "frame", "data": enriched})

    def complete_stream(
        self,
        stream_id: Optional[str],
        run_id: Optional[str] = None,
        status: str = "completed",
        video_urls: Optional[list[str]] = None,
        message: Optional[str] = None,
    ) -> None:
        if not stream_id:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            state = self._streams.get(stream_id)
            if state is None:
                return
            state["status"] = status
            if run_id:
                state["run_id"] = run_id
            if video_urls is not None:
                state["video_urls"] = list(video_urls)
            state["current_stage"] = "done"
            state["stage_index"] = 4
            if message:
                state["stage_message"] = message
            state["updated_at"] = now_iso
            state["events"].append(
                {
                    "type": "complete",
                    "data": {
                        "run_id": state["run_id"],
                        "status": status,
                        "video_urls": state["video_urls"],
                        "total_frames": len(state["frames"]),
                        "timestamp": now_iso,
                    },
                }
            )

    def get_state(self, stream_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            state = self._streams.get(stream_id)
            if state is None:
                return None
            return {
                "stream_id": state["stream_id"],
                "run_id": state["run_id"],
                "target_url": state["target_url"],
                "provider": state["provider"],
                "model_name": state["model_name"],
                "status": state["status"],
                "current_stage": state["current_stage"],
                "stage_index": state["stage_index"],
                "stage_message": state["stage_message"],
                "frames": list(state["frames"]),
                "logs": list(state.get("logs") or []),
                "partial_results": dict(state.get("partial_results") or {}),
                "events_count": len(state["events"]),
                "video_urls": list(state["video_urls"]),
                "started_at": state["started_at"],
                "updated_at": state["updated_at"],
            }

    def get_events_since(self, stream_id: str, cursor: int = 0) -> tuple[list[dict[str, Any]], str]:
        with self._lock:
            state = self._streams.get(stream_id)
            if state is None:
                return [], "unknown"
            events = list(state["events"][cursor:])
            return events, str(state["status"])


_DEFAULT_STREAM_MANAGER = LiveStreamManager()


def get_stream_manager() -> LiveStreamManager:
    """Return the shared AgentSwarm LiveStreamManager singleton."""
    return _DEFAULT_STREAM_MANAGER
