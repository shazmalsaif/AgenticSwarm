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
    status: Literal["passed", "failed", "skipped"]
    duration_ms: float = 0.0
    current_url: str = ""
    error_message: Optional[str] = None
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
    events: list[TraceEvent] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    overall_status: Literal["passed", "failed", "running"] = "running"


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
        current_url: str = "",
        error_message: Optional[str] = None,
    ) -> Optional[TraceEvent]:
        """Append a step event to an existing trace artifact."""
        event = TraceEvent(
            step_number=step_number,
            action_type=action_type,
            selector=selector,
            value=value,
            status=status,
            duration_ms=duration_ms,
            current_url=current_url,
            error_message=error_message,
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
        overall_status: Literal["passed", "failed"] = "passed",
        context: Optional[Any] = None,
        save_to_disk: bool = False,
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
