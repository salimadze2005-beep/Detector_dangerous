from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class ComponentState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    ERROR = "error"
    RECONNECTING = "reconnecting"


@dataclass(frozen=True, slots=True)
class HealthStatus:
    component: str
    state: ComponentState
    message: str
    updated_at: datetime


class HealthRegistry:
    def __init__(self):
        self._statuses: dict[str, HealthStatus] = {}
        self._lock = threading.Lock()

    def update(self, component: str, state: ComponentState, message: str = "") -> HealthStatus:
        status = HealthStatus(component, state, message, datetime.now(timezone.utc))
        with self._lock:
            self._statuses[component] = status
        return status

    def snapshot(self) -> dict[str, HealthStatus]:
        with self._lock:
            return dict(self._statuses)

