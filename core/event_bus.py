from __future__ import annotations

import logging
import queue
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    GUNSHOT_DETECTED = "GUNSHOT_DETECTED"
    KEYWORD_DETECTED = "KEYWORD_DETECTED"
    FALL_DETECTED = "FALL_DETECTED"
    FALL_RECOVERED = "FALL_RECOVERED"
    SPEECH_RECOGNIZED = "SPEECH_RECOGNIZED"
    COMPONENT_STATUS = "COMPONENT_STATUS"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class Event:
    type: EventType | str
    confidence: float = 1.0
    source: str = "unknown"
    severity: Severity = Severity.INFO
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Mapping[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        event_type = self.type if isinstance(self.type, EventType) else EventType(self.type)
        confidence = min(max(float(self.confidence), 0.0), 1.0)
        timestamp = self.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        object.__setattr__(self, "type", event_type)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "confidence": self.confidence,
            "source": self.source,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": dict(self.metadata),
        }


EventHandler = Callable[[Event], None]


@dataclass(slots=True)
class _Barrier:
    completed: threading.Event = field(default_factory=threading.Event)


class EventBus:
    """Thread-safe ordered event dispatcher with explicit lifecycle."""

    _STOP = object()

    def __init__(self, queue_size: int = 1_000):
        self._subscribers: dict[EventType, list[EventHandler]] = {}
        self._all_subscribers: list[EventHandler] = []
        self._queue: queue.Queue[Event | _Barrier | object] = queue.Queue(maxsize=queue_size)
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._running = threading.Event()

    def subscribe(self, event_type: EventType | str, handler: EventHandler) -> None:
        normalized = event_type if isinstance(event_type, EventType) else EventType(event_type)
        with self._lock:
            self._subscribers.setdefault(normalized, []).append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        with self._lock:
            self._all_subscribers.append(handler)

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._running.set()
            self._thread = threading.Thread(target=self._dispatch_loop, name="event-bus", daemon=True)
            self._thread.start()

    def publish(self, event: Event, timeout: float = 1.0) -> None:
        if not self._running.is_set():
            self.start()
        try:
            self._queue.put(event, timeout=timeout)
        except queue.Full as exc:
            raise RuntimeError("Очередь событий переполнена; событие не было принято") from exc

    def flush(self, timeout: float = 5.0) -> bool:
        barrier = _Barrier()
        try:
            self._queue.put(barrier, timeout=timeout)
        except queue.Full:
            return False
        return barrier.completed.wait(timeout)

    def stop(self, drain: bool = True, timeout: float = 5.0) -> bool:
        if not self._thread:
            return True
        if drain:
            self.flush(timeout=timeout)
        self._running.clear()
        try:
            self._queue.put(self._STOP, timeout=timeout)
        except queue.Full:
            return False
        self._thread.join(timeout)
        stopped = not self._thread.is_alive()
        if stopped:
            self._thread = None
        return stopped

    def _dispatch_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                if isinstance(item, _Barrier):
                    item.completed.set()
                    continue
                assert isinstance(item, Event)
                with self._lock:
                    handlers = [*self._all_subscribers, *self._subscribers.get(item.type, [])]
                for handler in handlers:
                    try:
                        handler(item)
                    except Exception:
                        logger.exception("Ошибка обработчика события %s", item.type.value)
            finally:
                self._queue.task_done()
