from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .sentinel import sentinel_exists


class StatusUpdater:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.path = self.project_root / "STATUS.json"
        self._lock = threading.Lock()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"last_updated": self._now(), "phases": {}, "blocking_decisions": []}
        try:
            return json.loads(self.path.read_text())
        except Exception:  # noqa: BLE001
            return {"last_updated": self._now(), "phases": {}, "blocking_decisions": []}

    def _atomic_write(self, payload: dict[str, Any]) -> None:
        # Use PID-specific tmp file to avoid cross-process collision when multiple
        # phase4 scripts run in parallel and all write to the same STATUS.json.
        tmp = self.path.with_name(f"STATUS.{os.getpid()}.json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(self.path)

    def update(
        self,
        phase_name: str,
        status: str,
        progress: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            data = self._read()
            phases = data.setdefault("phases", {})
            phases.setdefault(phase_name, {})
            phases[phase_name]["status"] = status
            if progress is not None:
                phases[phase_name]["progress"] = progress
            phases[phase_name]["sentinel"] = bool(sentinel_exists(phase_name))
            phases[phase_name]["error"] = error
            data["last_updated"] = self._now()
            data.setdefault("blocking_decisions", [])
            self._atomic_write(data)

    def append_blocking_decision(self, decision: dict[str, Any]) -> None:
        with self._lock:
            data = self._read()
            data.setdefault("blocking_decisions", [])
            data["blocking_decisions"].append(decision)
            data["last_updated"] = self._now()
            self._atomic_write(data)

