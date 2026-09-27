from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True, slots=True)
class IncidentRecord:
    event_id: str
    event_type: str
    timestamp: str
    confidence: float
    source: str
    severity: str
    description: str
    image_path: str | None
    delivery_status: str | None
    decision: str
    note: str
    reviewed_at: float | None
    clip_dir: str | None
    clip_files: tuple[str, ...]


class IncidentStore:
    """Operator-facing incident history stored next to the durable event journal."""

    CATEGORY_BY_EVENT = {
        "FALL_DETECTED": "fall",
        "GUNSHOT_DETECTED": "gunshot",
        "KEYWORD_DETECTED": "keyword",
    }
    _SELECT = """
        SELECT e.id, e.type, e.timestamp, e.confidence, e.source, e.severity,
               COALESCE(o.description, ''), o.image_path, o.status,
               COALESCE(r.decision, 'unreviewed'), COALESCE(r.note, ''), r.reviewed_at
        FROM events e
        LEFT JOIN alert_outbox o ON o.event_id = e.id
        LEFT JOIN incident_reviews r ON r.event_id = e.id
    """

    def __init__(self, db_path: str, clips_root: str):
        self.db_path = str(Path(db_path).resolve())
        self.clips_root = Path(clips_root).resolve()
        self._write_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10.0)
        connection.execute("PRAGMA journal_mode=WAL")
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
                CREATE TABLE IF NOT EXISTS incident_reviews (
                    event_id TEXT PRIMARY KEY,
                    decision TEXT NOT NULL DEFAULT 'unreviewed',
                    note TEXT NOT NULL DEFAULT '',
                    reviewed_at REAL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_incident_reviews_decision "
                "ON incident_reviews(decision, reviewed_at DESC)"
            )

    def review(self, event_id: str, decision: str, note: str = "") -> None:
        if decision not in {"confirmed", "false_alarm", "unreviewed"}:
            raise ValueError(f"Unsupported incident decision: {decision}")
        reviewed_at = None if decision == "unreviewed" else time.time()
        with self._write_lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO incident_reviews(event_id, decision, note, reviewed_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    decision = excluded.decision,
                    note = excluded.note,
                    reviewed_at = excluded.reviewed_at
                """,
                (event_id, decision, note.strip()[:4000], reviewed_at),
            )
            if decision == "false_alarm":
                connection.execute(
                    """
                    UPDATE alert_outbox
                    SET status = 'cancelled', last_error = 'Отклонено оператором как ложная тревога'
                    WHERE event_id = ? AND status IN ('preparing', 'pending')
                    """,
                    (event_id,),
                )

    def update_note(self, event_id: str, note: str) -> None:
        with self._write_lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO incident_reviews(event_id, decision, note, reviewed_at)
                VALUES (?, 'unreviewed', ?, NULL)
                ON CONFLICT(event_id) DO UPDATE SET note = excluded.note
                """,
                (event_id, note.strip()[:4000]),
            )

    def _record_from_row(self, row, resolve_media: bool) -> IncidentRecord:
        clip_dir, clip_files = (None, ())
        if resolve_media:
            clip_dir, clip_files = self._media_for_event(str(row[0]), str(row[1]), str(row[2]))
        return IncidentRecord(
            event_id=str(row[0]),
            event_type=str(row[1]),
            timestamp=str(row[2]),
            confidence=float(row[3]),
            source=str(row[4]),
            severity=str(row[5]),
            description=str(row[6] or ""),
            image_path=str(row[7]) if row[7] else None,
            delivery_status=str(row[8]) if row[8] else None,
            decision=str(row[9]),
            note=str(row[10] or ""),
            reviewed_at=float(row[11]) if row[11] is not None else None,
            clip_dir=clip_dir,
            clip_files=clip_files,
        )

    def recent(self, limit: int = 200, resolve_media: bool = True) -> list[IncidentRecord]:
        query = self._SELECT + """
            WHERE e.type IN ('FALL_DETECTED', 'GUNSHOT_DETECTED', 'KEYWORD_DETECTED')
            ORDER BY e.timestamp DESC
            LIMIT ?
        """
        with self._connection() as connection:
            rows = connection.execute(query, (max(1, min(int(limit), 5000)),)).fetchall()
        return [self._record_from_row(row, resolve_media) for row in rows]

    def get(self, event_id: str, resolve_media: bool = True) -> IncidentRecord | None:
        query = self._SELECT + """
            WHERE e.id = ?
              AND e.type IN ('FALL_DETECTED', 'GUNSHOT_DETECTED', 'KEYWORD_DETECTED')
            LIMIT 1
        """
        with self._connection() as connection:
            row = connection.execute(query, (event_id,)).fetchone()
        return None if row is None else self._record_from_row(row, resolve_media)

    def _media_for_event(self, event_id: str, event_type: str, timestamp: str) -> tuple[str | None, tuple[str, ...]]:
        category = self.CATEGORY_BY_EVENT.get(event_type)
        if not category:
            return None, ()
        try:
            dt = datetime.fromisoformat(timestamp)
            directory = self.clips_root / category / f"{dt:%Y}" / f"{dt:%m}" / event_id
        except ValueError:
            return None, ()
        if not directory.is_dir():
            return None, ()
        files = tuple(str(path.resolve()) for path in sorted(directory.glob("*.mp4")))
        return str(directory.resolve()), files
