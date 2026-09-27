from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from core.event_bus import Event, EventType


@dataclass(frozen=True, slots=True)
class OutboxItem:
    event_id: str
    event_type: str
    description: str
    image_path: str | None
    video_path: str | None
    delivery_stage: str
    status: str
    attempts: int
    next_attempt_at: float
    last_error: str | None
    created_at: float
    delivered_at: float | None


class EventStore:
    """Durable append-only event journal backed by SQLite."""

    def __init__(self, path: str):
        self.path = str(Path(path).resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_type_time ON events(type, timestamp DESC)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS alert_outbox (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    image_path TEXT,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at REAL NOT NULL,
                    last_error TEXT,
                    created_at REAL NOT NULL,
                    delivered_at REAL
                )
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(alert_outbox)").fetchall()
            }
            if "video_path" not in columns:
                connection.execute("ALTER TABLE alert_outbox ADD COLUMN video_path TEXT")
            if "delivery_stage" not in columns:
                connection.execute(
                    "ALTER TABLE alert_outbox ADD COLUMN delivery_stage TEXT "
                    "NOT NULL DEFAULT 'trigger'"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_outbox_due "
                "ON alert_outbox(status, next_attempt_at)"
            )

    def record(self, event: Event) -> None:
        payload = json.dumps(dict(event.metadata), ensure_ascii=False, default=str)
        with self._write_lock, self._connection() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO events
                    (id, type, timestamp, confidence, source, severity, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.type.value,
                    event.timestamp.isoformat(),
                    event.confidence,
                    event.source,
                    event.severity.value,
                    payload,
                ),
            )

    def counts(self) -> dict[str, int]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT type, COUNT(*) FROM events GROUP BY type"
            ).fetchall()
        return {str(event_type): int(count) for event_type, count in rows}

    def recent(self, limit: int = 100, event_type: EventType | None = None) -> list[dict]:
        query = (
            "SELECT id, type, timestamp, confidence, source, severity, metadata_json "
            "FROM events"
        )
        params: list[object] = []
        if event_type is not None:
            query += " WHERE type = ?"
            params.append(event_type.value)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(max(1, min(limit, 10_000)))
        with self._connection() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            {
                "id": row[0], "type": row[1], "timestamp": row[2],
                "confidence": row[3], "source": row[4], "severity": row[5],
                "metadata": json.loads(row[6]),
            }
            for row in rows
        ]

    def enqueue_alert(
        self,
        event: Event,
        description: str,
        image_path: str | None,
        now: float | None = None,
        await_video: bool = False,
    ) -> bool:
        created_at = time.time() if now is None else now
        status = "preparing" if await_video else "pending"
        with self._write_lock, self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO alert_outbox
                    (event_id, event_type, description, image_path, status, attempts,
                     next_attempt_at, last_error, created_at, delivered_at,
                     video_path, delivery_stage)
                VALUES (?, ?, ?, ?, ?, 0, ?, NULL, ?, NULL, NULL, 'trigger')
                """,
                (
                    event.id,
                    event.type.value,
                    description,
                    image_path,
                    status,
                    created_at,
                    created_at,
                ),
            )
            return cursor.rowcount == 1

    @staticmethod
    def _outbox_item(row) -> OutboxItem:
        return OutboxItem(
            event_id=row[0], event_type=row[1], description=row[2], image_path=row[3],
            status=row[4], attempts=int(row[5]), next_attempt_at=float(row[6]),
            last_error=row[7], created_at=float(row[8]), delivered_at=row[9],
            video_path=row[10], delivery_stage=str(row[11] or "trigger"),
        )

    def due_alerts(self, now: float | None = None, limit: int = 20) -> list[OutboxItem]:
        current = time.time() if now is None else now
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT event_id, event_type, description, image_path, status, attempts,
                       next_attempt_at, last_error, created_at, delivered_at,
                       video_path, delivery_stage
                FROM alert_outbox
                WHERE status = 'pending' AND next_attempt_at <= ?
                ORDER BY next_attempt_at, created_at
                LIMIT ?
                """,
                (current, max(1, min(limit, 1_000))),
            ).fetchall()
        return [self._outbox_item(row) for row in rows]

    def get_alert(self, event_id: str) -> OutboxItem | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT event_id, event_type, description, image_path, status, attempts,
                       next_attempt_at, last_error, created_at, delivered_at,
                       video_path, delivery_stage
                FROM alert_outbox WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()
        return None if row is None else self._outbox_item(row)

    def attach_delivery_video(
        self,
        event_id: str,
        video_path: str,
        now: float | None = None,
    ) -> bool:
        ready_at = time.time() if now is None else now
        with self._write_lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE alert_outbox
                SET video_path = ?, delivery_stage = 'trigger', status = 'pending',
                    next_attempt_at = ?, last_error = NULL
                WHERE event_id = ? AND status = 'preparing'
                """,
                (video_path, ready_at, event_id),
            )
            return cursor.rowcount == 1

    def fail_video_preparation(self, event_id: str, error_message: str) -> bool:
        with self._write_lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE alert_outbox
                SET status = 'failed', last_error = ?
                WHERE event_id = ? AND status = 'preparing'
                """,
                (error_message[:2_000], event_id),
            )
            return cursor.rowcount == 1

    def advance_to_upload(self, event_id: str, now: float | None = None) -> bool:
        ready_at = time.time() if now is None else now
        with self._write_lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE alert_outbox
                SET delivery_stage = 'upload', attempts = attempts + 1,
                    next_attempt_at = ?, last_error = NULL
                WHERE event_id = ? AND status = 'pending'
                  AND delivery_stage = 'trigger'
                """,
                (ready_at, event_id),
            )
            return cursor.rowcount == 1

    def mark_delivered(self, event_id: str, delivered_at: float | None = None) -> bool:
        delivered = time.time() if delivered_at is None else delivered_at
        with self._write_lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE alert_outbox
                SET status = 'delivered', attempts = attempts + 1,
                    delivered_at = ?, last_error = NULL
                WHERE event_id = ? AND status = 'pending'
                """,
                (delivered, event_id),
            )
            return cursor.rowcount == 1

    def schedule_retry(self, event_id: str, next_attempt_at: float, error_message: str) -> bool:
        with self._write_lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE alert_outbox
                SET status = 'pending', attempts = attempts + 1,
                    next_attempt_at = ?, last_error = ?
                WHERE event_id = ? AND status = 'pending'
                """,
                (next_attempt_at, error_message[:2_000], event_id),
            )
            return cursor.rowcount == 1

    def mark_permanent_failure(self, event_id: str, error_message: str) -> bool:
        with self._write_lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE alert_outbox
                SET status = 'failed', attempts = attempts + 1, last_error = ?
                WHERE event_id = ? AND status = 'pending'
                """,
                (error_message[:2_000], event_id),
            )
            return cursor.rowcount == 1

    def detach_expired_images(self, cutoff: float) -> list[str]:
        """Detach delivered images before deletion; pending and failed evidence is retained."""
        with self._write_lock, self._connection() as connection:
            rows = connection.execute(
                """
                SELECT image_path FROM alert_outbox
                WHERE status = 'delivered' AND delivered_at < ? AND image_path IS NOT NULL
                """,
                (cutoff,),
            ).fetchall()
            connection.execute(
                """
                UPDATE alert_outbox SET image_path = NULL
                WHERE status = 'delivered' AND delivered_at < ? AND image_path IS NOT NULL
                """,
                (cutoff,),
            )
        return [str(row[0]) for row in rows]

    def detach_expired_videos(self, cutoff: float) -> list[str]:
        with self._write_lock, self._connection() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT video_path FROM alert_outbox
                WHERE status = 'delivered' AND delivered_at < ? AND video_path IS NOT NULL
                """,
                (cutoff,),
            ).fetchall()
            connection.execute(
                """
                UPDATE alert_outbox SET video_path = NULL
                WHERE status = 'delivered' AND delivered_at < ? AND video_path IS NOT NULL
                """,
                (cutoff,),
            )
            unused_paths = []
            for row in rows:
                video_path = str(row[0])
                references = connection.execute(
                    "SELECT 1 FROM alert_outbox WHERE video_path = ? LIMIT 1",
                    (video_path,),
                ).fetchone()
                if references is None:
                    unused_paths.append(video_path)
        return unused_paths

